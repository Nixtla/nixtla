#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "diff.h"
#include "expanding.h"
#include "exponentially_weighted.h"
#include "grouped_array_functions.h"
#include "lag.h"
#include "rolling.h"
#include "scalers.h"
#include "seasonal.h"

using namespace pybind11::literals;

// The module is built with -fvisibility=hidden, so anything holding a pybind
// type has to be hidden too; internal linkage is the portable way to say it.
namespace {

// Entries used as offsets into a buffer: a negative one makes a slice start
// before it and a decreasing one makes a slice run backwards.
template <typename T>
inline void RequireOffsets(const T *values, py::ssize_t size) {
  for (py::ssize_t i = 0; i < size; ++i) {
    if (values[i] < 0) {
      throw std::invalid_argument("indptr values must be non-negative");
    }
    if (i > 0 && values[i] < values[i - 1]) {
      throw std::invalid_argument("indptr must be non-decreasing");
    }
  }
}

// One difference order per group, used both as a loop bound and to build the
// tails offsets, so a short array is read past its end.
inline void CheckDs(const CArray<indptr_t> &ds, int n_groups) {
  if (ds.size() != n_groups) {
    throw std::invalid_argument("ds must have one element per group");
  }
  for (py::ssize_t i = 0; i < ds.size(); ++i) {
    RequireNonNegative("d", ds.data()[i]);
  }
}

// Two entries per group, read as stats[2 * i] and stats[2 * i + 1], so anything
// shorter is read past its end.
template <typename T>
inline void CheckStats(const CArray<T> &stats, int n_groups) {
  if (stats.ndim() != 2 || stats.size() != 2 * n_groups) {
    throw std::invalid_argument("stats must have shape (n_groups, 2)");
  }
}

template <typename T> inline void SkipLags(T *out, int n, int lag) {
  int replacements = std::min(lag, n);
  for (int i = 0; i < replacements; ++i) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
}

template <typename T> class GroupedArray {
public:
  const py::array_t<T> data_;
  const py::array_t<indptr_t> indptr_;
  int num_threads_;

  GroupedArray(const py::array_t<T> data, const py::array_t<indptr_t> indptr,
               int num_threads)
      : data_(data), indptr_(indptr), num_threads_(num_threads) {}

  int NumGroups() const noexcept {
    return static_cast<int>(indptr_.size()) - 1;
  }

  py::array_t<T> operator[](int i) const {
    int num_groups = NumGroups();
    if (i >= num_groups) {
      throw std::out_of_range("Index out of range");
    }
    if (i < 0) {
      if (i < -num_groups) {
        throw std::out_of_range("Index out of range");
      }
      i += num_groups;
    }
    indptr_t start = indptr_.data()[i];
    indptr_t end = indptr_.data()[i + 1];
    auto buffer = data_.request();
    return py::array_t<T>({end - start}, {buffer.strides[0]},
                          static_cast<T *>(buffer.ptr) + start, data_);
  }

  py::array_t<T> Take(const CArray<indptr_t> indices) const {
    auto indices_data = indices.data();
    auto indptr_data = indptr_.data();
    const indptr_t num_groups = NumGroups();
    indptr_t out_size = 0;
    for (int i = 0; i < indices.size(); ++i) {
      if (indices_data[i] < 0 || indices_data[i] >= num_groups) {
        throw std::out_of_range("Index out of range");
      }
      indptr_t start = indptr_data[indices_data[i]];
      indptr_t end = indptr_data[indices_data[i] + 1];
      out_size += end - start;
    }
    py::array_t<T> out(out_size);
    size_t j = 0;
    for (int i = 0; i < indices.size(); ++i) {
      indptr_t start = indptr_data[indices_data[i]];
      indptr_t end = indptr_data[indices_data[i] + 1];
      std::copy(data_.data() + start, data_.data() + end,
                out.mutable_data() + j);
      j += end - start;
    }
    return out;
  }

  // The workers only touch raw buffers, never the Python API, so the GIL is
  // released around them. Exceptions can't cross a thread boundary, so each
  // worker stashes its own and we rethrow once everything has been joined.
  template <typename Func> void ForEach(Func f) const {
    const int n_groups = NumGroups();
    const int n_threads = std::clamp(num_threads_, 1, std::max(1, n_groups));
    if (n_threads < 2) {
      py::gil_scoped_release release;
      f(0, n_groups);
      return;
    }
    std::vector<std::exception_ptr> errors(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    const int groups_per_thread = n_groups / n_threads;
    const int remainder = n_groups % n_threads;
    std::exception_ptr spawn_error;
    {
      py::gil_scoped_release release;
      try {
        for (int t = 0; t < n_threads; ++t) {
          int start_group = t * groups_per_thread + std::min(t, remainder);
          int end_group =
              (t + 1) * groups_per_thread + std::min(t + 1, remainder);
          threads.emplace_back([&f, &errors, t, start_group, end_group]() {
            try {
              f(start_group, end_group);
            } catch (...) {
              errors[t] = std::current_exception();
            }
          });
        }
      } catch (...) {
        spawn_error = std::current_exception();
      }
      for (auto &thread : threads) {
        thread.join();
      }
    }
    if (spawn_error) {
      std::rethrow_exception(spawn_error);
    }
    for (auto &error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
  }

  template <typename Func, typename... Args>
  void Reduce(Func f, int n_out, T *out, int lag, Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, n_out, out, lag,
             &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n);
        if (start_idx + lag >= n) {
          for (int j = 0; j < n_out; ++j) {
            out[n_out * i + j] = std::numeric_limits<T>::quiet_NaN();
          }
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx - lag, out + n_out * i,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func, typename... Args>
  void VariableReduce(Func f, const indptr_t *indptr_out, T *out,
                      Args &&...args) const {
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, indptr_out, out,
             &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t out_n = indptr_out[i + 1] - indptr_out[i];
        f(data + start, n, out + indptr_out[i], out_n,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void ScalerTransform(Func f, const T *stats, T *out) const {
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, stats,
             out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        T offset = stats[2 * i];
        T scale = stats[2 * i + 1];
        if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
          scale = static_cast<T>(1.0);
        }
        for (indptr_t j = start; j < end; ++j) {
          out[j] = f(data[j], offset, scale);
        }
      }
    });
  }

  template <typename Func, typename... Args>
  void Transform(Func f, int lag, T *out, Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, lag, out,
             &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        SkipLags(out + start + start_idx, n - start_idx, lag);
        if (start_idx + lag >= n) {
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx - lag, out + start + lag,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void VariableTransform(Func f, const indptr_t *params, T *out) const {
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, params,
             out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        if (start_idx >= n) {
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx, params[i], out + start);
      }
    });
  }

  template <typename Func, typename... Args>
  void TransformAndReduce(Func f, int lag, T *out, int n_agg, T *agg,
                          Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, lag, out, n_agg,
             agg, &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        SkipLags(out + start + start_idx, n - start_idx, lag);
        if (start_idx + lag >= n) {
          // f never runs for an empty or all-NaN group, so its stats row would
          // otherwise be left uninitialised.
          std::fill(agg + i * n_agg, agg + (i + 1) * n_agg,
                    std::numeric_limits<T>::quiet_NaN());
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx - lag, out + start + lag, agg + i * n_agg,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void Zip(Func f, const GroupedArray<T> &other, const indptr_t *out_indptr,
           T *out) const {
    ForEach([data = data_.data(), indptr = indptr_.data(), &f,
             other_data = other.data_.data(),
             other_indptr = other.indptr_.data(), out_indptr,
             out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t other_start = other_indptr[i];
        indptr_t other_end = other_indptr[i + 1];
        indptr_t other_n = other_end - other_start;
        f(data + start, n, other_data + other_start, other_n,
          out + out_indptr[i]);
      }
    });
  }

  std::unique_ptr<GroupedArray<T>> WithData(const CArray<T> new_data) {
    if (new_data.size() != data_.size()) {
      throw std::invalid_argument("Data must have the same size");
    }
    return std::make_unique<GroupedArray<T>>(new_data, indptr_, num_threads_);
  }

  py::array_t<T> IndexFromEnd(int k) {
    RequireNonNegative("k", k);
    py::array_t<T> out(NumGroups());
    Reduce(grouped_array_functions::IndexFromEnd<T>, 1, out.mutable_data(), 0,
           k);
    return out;
  }
  py::array_t<T> Head(int k) {
    RequireNonNegative("k", k);
    py::array_t<T> out(k * NumGroups());
    Reduce(grouped_array_functions::Head<T>, k, out.mutable_data(), 0, k);
    return out;
  }
  py::array_t<T> Tail(int k) {
    RequireNonNegative("k", k);
    py::array_t<T> out(k * NumGroups());
    Reduce(grouped_array_functions::Tail<T>, k, out.mutable_data(), 0, k);
    return out;
  }
  std::unique_ptr<GroupedArray<T>> Append(const GroupedArray<T> &other) {
    if (NumGroups() != other.NumGroups()) {
      throw std::invalid_argument("Number of groups must be the same");
    }
    py::array_t<T> out_data(data_.size() + other.data_.size());
    py::array_t<indptr_t> out_indptr(indptr_.size());
    std::transform(indptr_.data(), indptr_.data() + indptr_.size(),
                   other.indptr_.data(), out_indptr.mutable_data(),
                   std::plus<indptr_t>());
    Zip(grouped_array_functions::Append<T>, other, out_indptr.data(),
        out_data.mutable_data());
    return std::make_unique<GroupedArray<T>>(out_data, out_indptr,
                                             num_threads_);
  }
  py::array_t<T> Tails(const CArray<indptr_t> out_indptr) {
    if (out_indptr.size() != indptr_.size()) {
      throw std::invalid_argument(
          "indptr must have one element per group plus one");
    }
    RequireOffsets(out_indptr.data(), out_indptr.size());
    // the output is sized from the last offset, so a non-zero first one would
    // leave the elements before it uninitialized
    if (out_indptr.data()[0] != 0) {
      throw std::invalid_argument("First element of indptr must be zero");
    }
    py::array_t<T> out(out_indptr.data()[NumGroups()]);
    VariableReduce(grouped_array_functions::Tail<T>, out_indptr.data(),
                   out.mutable_data());
    return out;
  }

  py::array_t<T> LagTransform(int lag) {
    py::array_t<T> out(data_.size());
    Transform(lag::LagTransform<T>, lag, out.mutable_data());
    return out;
  }

  template <typename Func>
  py::array_t<T> RollingTransform(Func transform, int lag, int window_size,
                                  int min_samples, bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(transform, lag, out.mutable_data(), window_size, min_samples,
              skipna);
    return out;
  }
  py::array_t<T> RollingMeanTransform(int lag, int window_size, int min_samples,
                                      bool skipna = false) {
    return RollingTransform(rolling::MeanTransform<T>, lag, window_size,
                            min_samples, skipna);
  }
  py::array_t<T> RollingStdTransform(int lag, int window_size, int min_samples,
                                     bool skipna = false) {
    return RollingTransform(rolling::StdTransform<T>, lag, window_size,
                            min_samples, skipna);
  }
  py::array_t<T> RollingMinTransform(int lag, int window_size, int min_samples,
                                     bool skipna = false) {
    return RollingTransform(rolling::MinTransform<T>, lag, window_size,
                            min_samples, skipna);
  }
  py::array_t<T> RollingMaxTransform(int lag, int window_size, int min_samples,
                                     bool skipna = false) {
    return RollingTransform(rolling::MaxTransform<T>, lag, window_size,
                            min_samples, skipna);
  }
  py::array_t<T> RollingQuantileTransform(int lag, T p, int window_size,
                                          int min_samples,
                                          bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(data_.size());
    Transform(rolling::QuantileTransform<T>, lag, out.mutable_data(),
              window_size, min_samples, p, skipna);
    return out;
  }

  template <typename Func>
  py::array_t<T> RollingUpdate(Func transform, int lag, int window_size,
                               int min_samples, bool skipna = false) {
    py::array_t<T> out(NumGroups());
    Reduce(transform, 1, out.mutable_data(), lag, window_size, min_samples,
           skipna);
    return out;
  }
  py::array_t<T> RollingMeanUpdate(int lag, int window_size, int min_samples,
                                   bool skipna = false) {
    return RollingUpdate(rolling::MeanUpdate<T>, lag, window_size, min_samples,
                         skipna);
  }
  py::array_t<T> RollingStdUpdate(int lag, int window_size, int min_samples,
                                  bool skipna = false) {
    return RollingUpdate(rolling::StdUpdate<T>, lag, window_size, min_samples,
                         skipna);
  }
  py::array_t<T> RollingMaxUpdate(int lag, int window_size, int min_samples,
                                  bool skipna = false) {
    return RollingUpdate(rolling::MaxUpdate<T>, lag, window_size, min_samples,
                         skipna);
  }
  py::array_t<T> RollingMinUpdate(int lag, int window_size, int min_samples,
                                  bool skipna = false) {
    return RollingUpdate(rolling::MinUpdate<T>, lag, window_size, min_samples,
                         skipna);
  }
  py::array_t<T> RollingQuantileUpdate(int lag, T p, int window_size,
                                       int min_samples, bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(NumGroups());
    Reduce(rolling::QuantileUpdate<T>, 1, out.mutable_data(), lag, window_size,
           min_samples, p, skipna);
    return out;
  }

  template <typename Func>
  py::array_t<T> SeasonalRollingTransform(Func transform, int lag,
                                          int season_length, int window_size,
                                          int min_samples,
                                          bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(transform, lag, out.mutable_data(), season_length, window_size,
              min_samples, skipna);
    return out;
  }
  py::array_t<T> SeasonalRollingMeanTransform(int lag, int season_length,
                                              int window_size, int min_samples,
                                              bool skipna = false) {
    return SeasonalRollingTransform(rolling::SeasonalMeanTransform<T>, lag,
                                    season_length, window_size, min_samples,
                                    skipna);
  }
  py::array_t<T> SeasonalRollingStdTransform(int lag, int season_length,
                                             int window_size, int min_samples,
                                             bool skipna = false) {
    return SeasonalRollingTransform(rolling::SeasonalStdTransform<T>, lag,
                                    season_length, window_size, min_samples,
                                    skipna);
  }
  py::array_t<T> SeasonalRollingMinTransform(int lag, int season_length,
                                             int window_size, int min_samples,
                                             bool skipna = false) {
    return SeasonalRollingTransform(rolling::SeasonalMinTransform<T>, lag,
                                    season_length, window_size, min_samples,
                                    skipna);
  }
  py::array_t<T> SeasonalRollingMaxTransform(int lag, int season_length,
                                             int window_size, int min_samples,
                                             bool skipna = false) {
    return SeasonalRollingTransform(rolling::SeasonalMaxTransform<T>, lag,
                                    season_length, window_size, min_samples,
                                    skipna);
  }
  py::array_t<T> SeasonalRollingQuantileTransform(int lag, T p,
                                                  int season_length,
                                                  int window_size,
                                                  int min_samples,
                                                  bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(data_.size());
    Transform(rolling::SeasonalQuantileTransform<T>, lag, out.mutable_data(),
              season_length, window_size, min_samples, p, skipna);
    return out;
  }

  template <typename Func>
  py::array_t<T> SeasonalRollingUpdate(Func transform, int lag,
                                       int season_length, int window_size,
                                       int min_samples, bool skipna = false) {
    py::array_t<T> out(NumGroups());
    Reduce(transform, 1, out.mutable_data(), lag, season_length, window_size,
           min_samples, skipna);
    return out;
  }
  py::array_t<T> SeasonalRollingMeanUpdate(int lag, int season_length,
                                           int window_size, int min_samples,
                                           bool skipna = false) {
    return SeasonalRollingUpdate(rolling::SeasonalMeanUpdate<T>, lag,
                                 season_length, window_size, min_samples,
                                 skipna);
  }
  py::array_t<T> SeasonalRollingStdUpdate(int lag, int season_length,
                                          int window_size, int min_samples,
                                          bool skipna = false) {
    return SeasonalRollingUpdate(rolling::SeasonalStdUpdate<T>, lag,
                                 season_length, window_size, min_samples,
                                 skipna);
  }
  py::array_t<T> SeasonalRollingMinUpdate(int lag, int season_length,
                                          int window_size, int min_samples,
                                          bool skipna = false) {
    return SeasonalRollingUpdate(rolling::SeasonalMinUpdate<T>, lag,
                                 season_length, window_size, min_samples,
                                 skipna);
  }
  py::array_t<T> SeasonalRollingMaxUpdate(int lag, int season_length,
                                          int window_size, int min_samples,
                                          bool skipna = false) {
    return SeasonalRollingUpdate(rolling::SeasonalMaxUpdate<T>, lag,
                                 season_length, window_size, min_samples,
                                 skipna);
  }
  py::array_t<T> SeasonalRollingQuantileUpdate(int lag, T p, int season_length,
                                               int window_size, int min_samples,
                                               bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(NumGroups());
    Reduce(rolling::SeasonalQuantileUpdate<T>, 1, out.mutable_data(), lag,
           season_length, window_size, min_samples, p, skipna);
    return out;
  }

  std::tuple<py::array_t<T>, py::array_t<T>>
  ExpandingMeanTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg(NumGroups());
    TransformAndReduce(expanding::MeanTransform<T>, lag, out.mutable_data(), 1,
                       agg.mutable_data(), skipna);
    return std::make_tuple(out, agg);
  }
  std::tuple<py::array_t<T>, py::array_t<T>>
  ExpandingStdTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg({static_cast<int>(NumGroups()), 3});
    TransformAndReduce(expanding::StdTransform<T>, lag, out.mutable_data(), 3,
                       agg.mutable_data(), skipna);
    return std::make_tuple(out, agg);
  }
  py::array_t<T> ExpandingMinTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MinTransform<T>, lag, out.mutable_data(), skipna);
    return out;
  }
  py::array_t<T> ExpandingMaxTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MaxTransform<T>, lag, out.mutable_data(), skipna);
    return out;
  }
  py::array_t<T> ExpandingQuantileTransform(int lag, T p, bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(data_.size());
    Transform(expanding::QuantileTransform<T>, lag, out.mutable_data(), p,
              skipna);
    return out;
  }
  py::array_t<T> ExpandingQuantileUpdate(int lag, T p, bool skipna = false) {
    rolling::RequireProbability("p", p);
    py::array_t<T> out(NumGroups());
    Reduce(expanding::QuantileUpdate<T>, 1, out.mutable_data(), lag, p, skipna);
    return out;
  }

  py::array_t<T> ExponentiallyWeightedMeanTransform(int lag, T alpha,
                                                    bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(exponentially_weighted::MeanTransform<T>, lag, out.mutable_data(),
              alpha, skipna);
    return out;
  }

  template <typename Func>
  py::array_t<T> ScalerStats(Func func, bool skipna = false) {
    py::array_t<T> out({static_cast<int>(NumGroups()), 2});
    Reduce(func, 2, out.mutable_data(), 0, skipna);
    return out;
  }
  py::array_t<T> MinMaxScalerStats(bool skipna = false) {
    return ScalerStats(scalers::MinMaxScalerStats<T>, skipna);
  }
  py::array_t<T> StandardScalerStats(bool skipna = false) {
    return ScalerStats(scalers::StandardScalerStats<T>, skipna);
  }
  py::array_t<T> RobustIqrScalerStats(bool skipna = false) {
    return ScalerStats(scalers::RobustScalerIqrStats<T>, skipna);
  }
  py::array_t<T> RobustMadScalerStats(bool skipna = false) {
    return ScalerStats(scalers::RobustScalerMadStats<T>, skipna);
  }
  py::array_t<T> ApplyScaler(const CArray<T> stats) {
    CheckStats(stats, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::CommonScalerTransform<T>, stats.data(),
                    out.mutable_data());
    return out;
  }
  py::array_t<T> InvertScaler(const CArray<T> stats) {
    CheckStats(stats, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::CommonScalerInverseTransform<T>, stats.data(),
                    out.mutable_data());
    return out;
  }
  py::array_t<T> BoxCoxLambdaGuerrero(int period, T lower, T upper) {
    py::array_t<T> out({NumGroups(), 2});
    Reduce(scalers::BoxCoxLambdaGuerrero<T>, 2, out.mutable_data(), 0, period,
           lower, upper);
    return out;
  }
  py::array_t<T> BoxCoxLambdaLogLik(T lower, T upper) {
    py::array_t<T> out({NumGroups(), 2});
    Reduce(scalers::BoxCoxLambdaLogLik<T>, 2, out.mutable_data(), 0, lower,
           upper);
    return out;
  }
  py::array_t<T> BoxCoxTransform(const CArray<T> lambdas) {
    CheckStats(lambdas, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::BoxCoxTransform<T>, lambdas.data(),
                    out.mutable_data());
    return out;
  }
  py::array_t<T> BoxCoxInverseTransform(const CArray<T> lambdas) {
    CheckStats(lambdas, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::BoxCoxInverseTransform<T>, lambdas.data(),
                    out.mutable_data());
    return out;
  }

  py::array_t<T> NumDiffs(int max_d) {
    py::array_t<T> out(NumGroups());
    Reduce(diff::NumDiffs<T>, 1, out.mutable_data(), 0, max_d);
    return out;
  }
  py::array_t<T> NumSeasDiffs(int period, int max_d) {
    py::array_t<T> out(NumGroups());
    Reduce(diff::NumSeasDiffs<T>, 1, out.mutable_data(), 0, period, max_d);
    return out;
  }
  py::array_t<T> NumSeasDiffsPeriods(int max_d, const CArray<T> periods) {
    if (periods.size() != NumGroups()) {
      throw std::invalid_argument("periods must have one element per group");
    }
    py::array_t<T> periods_and_out({static_cast<int>(NumGroups()), 2});
    auto periods_ptr = periods.data();
    auto periods_and_out_ptr = periods_and_out.mutable_data();
    for (py::ssize_t i = 0; i < periods.size(); ++i) {
      periods_and_out_ptr[2 * i] = periods_ptr[i];
    }
    Reduce(diff::NumSeasDiffsPeriods<T>, 2, periods_and_out.mutable_data(), 0,
           max_d);
    py::array_t<T> out(NumGroups());
    auto out_ptr = out.mutable_data();
    for (py::ssize_t i = 0; i < periods.size(); ++i) {
      out_ptr[i] = periods_and_out_ptr[2 * i + 1];
    }
    return out;
  }
  py::array_t<T> Periods(size_t max_lag) {
    py::array_t<T> out(NumGroups());
    Reduce(seasonal::GreatestAutocovariance<T>, 1, out.mutable_data(), 0,
           max_lag);
    return out;
  }
  py::array_t<T> Difference(int d) {
    RequireNonNegative("d", d);
    py::array_t<T> out(data_.size());
    Transform(seasonal::Difference<T>, 0, out.mutable_data(), d);
    return out;
  }
  py::array_t<T> Differences(const CArray<indptr_t> ds) {
    CheckDs(ds, NumGroups());
    py::array_t<T> out(data_.size());
    VariableTransform(diff::Differences<T>, ds.data(), out.mutable_data());
    return out;
  }
  py::array_t<T> InvertDifference(int d, const CArray<T> tails) {
    py::array_t<indptr_t> ds(NumGroups());
    std::fill(ds.mutable_data(), ds.mutable_data() + ds.size(), d);
    return InvertDifferences(ds, tails);
  }
  py::array_t<T> InvertDifferences(const CArray<indptr_t> ds,
                                   const CArray<T> tails) {
    CheckDs(ds, NumGroups());
    py::array_t<indptr_t> tails_indptr(indptr_.size());
    auto ds_data = ds.data();
    auto tails_indptr_data = tails_indptr.mutable_data();
    tails_indptr_data[0] = 0;
    for (int i = 1; i < tails_indptr.size(); ++i) {
      tails_indptr_data[i] = tails_indptr_data[i - 1] + ds_data[i - 1];
    }
    // tails_ga is built here rather than through the factory, so the check the
    // factory would have done on its last offset has to happen here.
    if (static_cast<py::ssize_t>(tails_indptr_data[tails_indptr.size() - 1]) !=
        tails.size()) {
      throw std::invalid_argument(
          "tails must have as many elements as the sum of ds");
    }
    auto tails_ga = GroupedArray<T>(tails, tails_indptr, num_threads_);
    py::array_t<T> out(data_.size());
    Zip(diff::InvertDifference<T>, tails_ga, indptr_.data(),
        out.mutable_data());
    return out;
  }
};

// Validates indptr at the Python boundary and returns it as indptr_t. Every
// entry is used as an offset into data, so all of them have to be
// representable and describe a non-decreasing range, not just the last one:
// casting straight to indptr_t would wrap a too-large value into one that can
// still compare equal to the size of data. Takes py::object so that lists,
// tuples and Series keep working, which py::array would reject.
inline py::array_t<indptr_t> CheckedIndptr(const py::object &indptr,
                                           py::ssize_t data_size) {
  auto indptr64 = py::array_t < int64_t,
       py::array::c_style | py::array::forcecast > ::ensure(indptr);
  if (!indptr64) {
    throw std::invalid_argument("indptr must be an integer array");
  }
  if (indptr64.ndim() != 1) {
    throw std::invalid_argument("indptr must be a 1d array");
  }
  if (indptr64.size() < 1) {
    throw std::invalid_argument("indptr must have at least one element");
  }
  const int64_t *values = indptr64.data();
  for (py::ssize_t i = 0; i < indptr64.size(); ++i) {
    if (values[i] > std::numeric_limits<indptr_t>::max()) {
      throw std::invalid_argument(
          "indptr values must be representable with 32-bit integers");
    }
  }
  RequireOffsets(values, indptr64.size());
  if (data_size != values[indptr64.size() - 1]) {
    throw std::invalid_argument(
        "Last element of indptr must be equal to the size of data");
  }
  return py::array_t < indptr_t,
         py::array::c_style | py::array::forcecast > (indptr64);
}

template <typename T> void bind_ga(py::module &m, const std::string &name) {
  py::class_<GroupedArray<T>>(m, name.c_str())
      .def(py::init(
          [](const CArray<T> &data, const py::object &indptr, int num_threads) {
            return std::make_unique<GroupedArray<T>>(
                data, CheckedIndptr(indptr, data.size()), num_threads);
          }))
      .def_readonly("data", &GroupedArray<T>::data_)
      .def_readonly("indptr", &GroupedArray<T>::indptr_)
      .def_readwrite("num_threads", &GroupedArray<T>::num_threads_)
      .def("__getitem__", &GroupedArray<T>::operator[])
      .def("__len__", &GroupedArray<T>::NumGroups)
      .def("_with_data", &GroupedArray<T>::WithData)
      .def("_index_from_end", &GroupedArray<T>::IndexFromEnd)
      .def("_take", &GroupedArray<T>::Take)
      .def("_head", &GroupedArray<T>::Head)
      .def("_tail", &GroupedArray<T>::Tail)
      .def("_tails", &GroupedArray<T>::Tails)
      .def("_append", &GroupedArray<T>::Append)
      .def("_lag", &GroupedArray<T>::LagTransform)
      .def("_rolling_mean", &GroupedArray<T>::RollingMeanTransform,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_std", &GroupedArray<T>::RollingStdTransform,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_min", &GroupedArray<T>::RollingMinTransform,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_max", &GroupedArray<T>::RollingMaxTransform,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_quantile", &GroupedArray<T>::RollingQuantileTransform,
           py::arg("lag"), py::arg("p"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_rolling_mean_update", &GroupedArray<T>::RollingMeanUpdate,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_std_update", &GroupedArray<T>::RollingStdUpdate,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_min_update", &GroupedArray<T>::RollingMinUpdate,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_max_update", &GroupedArray<T>::RollingMaxUpdate,
           py::arg("lag"), py::arg("window_size"), py::arg("min_samples"),
           py::arg("skipna") = false)
      .def("_rolling_quantile_update", &GroupedArray<T>::RollingQuantileUpdate,
           py::arg("lag"), py::arg("p"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_mean",
           &GroupedArray<T>::SeasonalRollingMeanTransform, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_std",
           &GroupedArray<T>::SeasonalRollingStdTransform, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_min",
           &GroupedArray<T>::SeasonalRollingMinTransform, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_max",
           &GroupedArray<T>::SeasonalRollingMaxTransform, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_quantile",
           &GroupedArray<T>::SeasonalRollingQuantileTransform, py::arg("lag"),
           py::arg("p"), py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_mean_update",
           &GroupedArray<T>::SeasonalRollingMeanUpdate, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_std_update",
           &GroupedArray<T>::SeasonalRollingStdUpdate, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_min_update",
           &GroupedArray<T>::SeasonalRollingMinUpdate, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_max_update",
           &GroupedArray<T>::SeasonalRollingMaxUpdate, py::arg("lag"),
           py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_seasonal_rolling_quantile_update",
           &GroupedArray<T>::SeasonalRollingQuantileUpdate, py::arg("lag"),
           py::arg("p"), py::arg("season_length"), py::arg("window_size"),
           py::arg("min_samples"), py::arg("skipna") = false)
      .def("_expanding_mean", &GroupedArray<T>::ExpandingMeanTransform,
           py::arg("lag"), py::arg("skipna") = false)
      .def("_expanding_std", &GroupedArray<T>::ExpandingStdTransform,
           py::arg("lag"), py::arg("skipna") = false)
      .def("_expanding_min", &GroupedArray<T>::ExpandingMinTransform,
           py::arg("lag"), py::arg("skipna") = false)
      .def("_expanding_max", &GroupedArray<T>::ExpandingMaxTransform,
           py::arg("lag"), py::arg("skipna") = false)
      .def("_expanding_quantile", &GroupedArray<T>::ExpandingQuantileTransform,
           py::arg("lag"), py::arg("p"), py::arg("skipna") = false)
      .def("_expanding_quantile_update",
           &GroupedArray<T>::ExpandingQuantileUpdate, py::arg("lag"),
           py::arg("p"), py::arg("skipna") = false)
      .def("_exponentially_weighted_mean",
           &GroupedArray<T>::ExponentiallyWeightedMeanTransform, py::arg("lag"),
           py::arg("alpha"), py::arg("skipna") = false)
      .def("_minmax_stats", &GroupedArray<T>::MinMaxScalerStats,
           py::arg("skipna") = false)
      .def("_standard_stats", &GroupedArray<T>::StandardScalerStats,
           py::arg("skipna") = false)
      .def("_robust_iqr_stats", &GroupedArray<T>::RobustIqrScalerStats,
           py::arg("skipna") = false)
      .def("_robust_mad_stats", &GroupedArray<T>::RobustMadScalerStats,
           py::arg("skipna") = false)
      .def("_scaler_transform", &GroupedArray<T>::ApplyScaler)
      .def("_scaler_inverse_transform", &GroupedArray<T>::InvertScaler)
      .def("_boxcox_guerrero", &GroupedArray<T>::BoxCoxLambdaGuerrero)
      .def("_boxcox_loglik", &GroupedArray<T>::BoxCoxLambdaLogLik)
      .def("_boxcox", &GroupedArray<T>::BoxCoxTransform)
      .def("_inv_boxcox", &GroupedArray<T>::BoxCoxInverseTransform)
      .def("_num_diffs", &GroupedArray<T>::NumDiffs)
      .def("_num_seas_diffs", &GroupedArray<T>::NumSeasDiffs)
      .def("_num_seas_diffs_periods", &GroupedArray<T>::NumSeasDiffsPeriods)
      .def("_periods", &GroupedArray<T>::Periods)
      .def("_diff", &GroupedArray<T>::Difference)
      .def("_diffs", &GroupedArray<T>::Differences)
      .def("_inv_diff", &GroupedArray<T>::InvertDifference)
      .def("_inv_diffs", &GroupedArray<T>::InvertDifferences);
}

} // namespace

void init_ga(py::module_ &m) {
  py::module_ ga = m.def_submodule("grouped_array");
  bind_ga<float>(ga, "_GroupedArrayFloat32");
  bind_ga<double>(ga, "_GroupedArrayFloat64");
  ga.def(
      "GroupedArray",
      [](py::array data, const py::object &indptr_arg,
         int num_threads) -> py::object {
        if (data.ndim() != 1) {
          throw std::invalid_argument("data must be a 1d array");
        }
        auto indptr = CheckedIndptr(indptr_arg, data.size());
        data = py::array::ensure(data, py::array::c_style);
        if (data.dtype().kind() != 'f') {
          data = data.attr("astype")("float64");
        }
        if (py::isinstance<py::array_t<float>>(data)) {
          return py::cast(std::make_unique<GroupedArray<float>>(
              data.cast<py::array_t<float>>(), indptr, num_threads));
        } else if (py::isinstance<py::array_t<double>>(data)) {
          return py::cast(std::make_unique<GroupedArray<double>>(
              data.cast<py::array_t<double>>(), indptr, num_threads));
        } else {
          throw py::type_error("Unsupported dtype");
        }
      },
      "data"_a, "indptr"_a, "num_threads"_a = 1);
}
