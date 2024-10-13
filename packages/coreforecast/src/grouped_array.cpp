#include <limits>

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

  py::array_t<T> Take(const py::array_t<indptr_t> indices) const {
    auto indices_data = indices.data();
    auto indptr_data = indptr_.data();
    indptr_t out_size = 0;
    for (int i = 0; i < indices.size(); ++i) {
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

  template <typename Func> void ForEach(Func f) const noexcept {
    if (num_threads_ < 2) {
      f(0, NumGroups());
      return;
    }
    std::vector<std::thread> threads;
    threads.reserve(num_threads_);
    int groups_per_thread = NumGroups() / num_threads_;
    int remainder = NumGroups() % num_threads_;
    for (int t = 0; t < num_threads_; ++t) {
      int start_group = t * groups_per_thread + std::min(t, remainder);
      int end_group = (t + 1) * groups_per_thread + std::min(t + 1, remainder);
      threads.emplace_back(f, start_group, end_group);
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

  template <typename Func, typename... Args>
  void Reduce(Func f, int n_out, T *out, int lag,
              Args &&...args) const noexcept {
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
                      Args &&...args) const noexcept {
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
  void ScalerTransform(Func f, const T *stats, T *out) const noexcept {
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
  void Transform(Func f, int lag, T *out, Args &&...args) const noexcept {
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
  void VariableTransform(Func f, const indptr_t *params,
                         T *out) const noexcept {
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
                          Args &&...args) const noexcept {
    ForEach([data = data_.data(), indptr = indptr_.data(), &f, lag, out, n_agg,
             agg, &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        SkipLags(out + start + start_idx, n - start_idx, lag);
        if (start_idx + lag >= n)
          continue;
        start += start_idx;
        f(data + start, n - start_idx - lag, out + start + lag, agg + i * n_agg,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void Zip(Func f, const GroupedArray<T> &other, const indptr_t *out_indptr,
           T *out) const noexcept {
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

  std::unique_ptr<GroupedArray<T>> WithData(const py::array_t<T> new_data) {
    if (new_data.size() != data_.size()) {
      throw std::invalid_argument("Data must have the same size");
    }
    return std::make_unique<GroupedArray<T>>(new_data, indptr_, num_threads_);
  }

  py::array_t<T> IndexFromEnd(int k) {
    py::array_t<T> out(NumGroups());
    Reduce(grouped_array_functions::IndexFromEnd<T>, 1, out.mutable_data(), 0,
           k);
    return out;
  }
  py::array_t<T> Head(int k) {
    py::array_t<T> out(k * NumGroups());
    Reduce(grouped_array_functions::Head<T>, k, out.mutable_data(), 0, k);
    return out;
  }
  py::array_t<T> Tail(int k) {
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
  py::array_t<T> Tails(py::array_t<indptr_t> out_indptr) {
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
                                  int min_samples) {
    py::array_t<T> out(data_.size());
    Transform(transform, lag, out.mutable_data(), window_size, min_samples);
    return out;
  }
  py::array_t<T> RollingMeanTransform(int lag, int window_size,
                                      int min_samples) {
    return RollingTransform(rolling::MeanTransform<T>, lag, window_size,
                            min_samples);
  }
  py::array_t<T> RollingStdTransform(int lag, int window_size,
                                     int min_samples) {
    return RollingTransform(rolling::StdTransform<T>, lag, window_size,
                            min_samples);
  }
  py::array_t<T> RollingMinTransform(int lag, int window_size,
                                     int min_samples) {
    return RollingTransform(rolling::MinTransform<T>, lag, window_size,
                            min_samples);
  }
  py::array_t<T> RollingMaxTransform(int lag, int window_size,
                                     int min_samples) {
    return RollingTransform(rolling::MaxTransform<T>, lag, window_size,
                            min_samples);
  }
  py::array_t<T> RollingQuantileTransform(int lag, T p, int window_size,
                                          int min_samples) {
    py::array_t<T> out(data_.size());
    Transform(rolling::QuantileTransform<T>, lag, out.mutable_data(),
              window_size, min_samples, p);
    return out;
  }

  template <typename Func>
  py::array_t<T> RollingUpdate(Func transform, int lag, int window_size,
                               int min_samples) {
    py::array_t<T> out(NumGroups());
    Reduce(transform, 1, out.mutable_data(), lag, window_size, min_samples);
    return out;
  }
  py::array_t<T> RollingMeanUpdate(int lag, int window_size, int min_samples) {
    return RollingUpdate(rolling::MeanUpdate<T>, lag, window_size, min_samples);
  }
  py::array_t<T> RollingStdUpdate(int lag, int window_size, int min_samples) {
    return RollingUpdate(rolling::StdUpdate<T>, lag, window_size, min_samples);
  }
  py::array_t<T> RollingMaxUpdate(int lag, int window_size, int min_samples) {
    return RollingUpdate(rolling::MaxUpdate<T>, lag, window_size, min_samples);
  }
  py::array_t<T> RollingMinUpdate(int lag, int window_size, int min_samples) {
    return RollingUpdate(rolling::MinUpdate<T>, lag, window_size, min_samples);
  }
  py::array_t<T> RollingQuantileUpdate(int lag, T p, int window_size,
                                       int min_samples) {
    py::array_t<T> out(NumGroups());
    Reduce(rolling::QuantileUpdate<T>, 1, out.mutable_data(), lag, window_size,
           min_samples, p);
    return out;
  }

  template <typename Func>
  py::array_t<T> SeasonalRollingTransform(Func transform, int lag,
                                          int season_length, int window_size,
                                          int min_samples) {
    py::array_t<T> out(data_.size());
    Transform(transform, lag, out.mutable_data(), season_length, window_size,
              min_samples);
    return out;
  }
  py::array_t<T> SeasonalRollingMeanTransform(int lag, int season_length,
                                              int window_size,
                                              int min_samples) {
    return SeasonalRollingTransform(rolling::SeasonalMeanTransform<T>, lag,
                                    season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingStdTransform(int lag, int season_length,
                                             int window_size, int min_samples) {
    return SeasonalRollingTransform(rolling::SeasonalStdTransform<T>, lag,
                                    season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingMinTransform(int lag, int season_length,
                                             int window_size, int min_samples) {
    return SeasonalRollingTransform(rolling::SeasonalMinTransform<T>, lag,
                                    season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingMaxTransform(int lag, int season_length,
                                             int window_size, int min_samples) {
    return SeasonalRollingTransform(rolling::SeasonalMaxTransform<T>, lag,
                                    season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingQuantileTransform(int lag, T p,
                                                  int season_length,
                                                  int window_size,
                                                  int min_samples) {
    py::array_t<T> out(data_.size());
    Transform(rolling::SeasonalQuantileTransform<T>, lag, out.mutable_data(),
              season_length, window_size, min_samples, p);
    return out;
  }

  template <typename Func>
  py::array_t<T> SeasonalRollingUpdate(Func transform, int lag,
                                       int season_length, int window_size,
                                       int min_samples) {
    py::array_t<T> out(NumGroups());
    Reduce(transform, 1, out.mutable_data(), lag, season_length, window_size,
           min_samples);
    return out;
  }
  py::array_t<T> SeasonalRollingMeanUpdate(int lag, int season_length,
                                           int window_size, int min_samples) {
    return SeasonalRollingUpdate(rolling::SeasonalMeanUpdate<T>, lag,
                                 season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingStdUpdate(int lag, int season_length,
                                          int window_size, int min_samples) {
    return SeasonalRollingUpdate(rolling::SeasonalStdUpdate<T>, lag,
                                 season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingMinUpdate(int lag, int season_length,
                                          int window_size, int min_samples) {
    return SeasonalRollingUpdate(rolling::SeasonalMinUpdate<T>, lag,
                                 season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingMaxUpdate(int lag, int season_length,
                                          int window_size, int min_samples) {
    return SeasonalRollingUpdate(rolling::SeasonalMaxUpdate<T>, lag,
                                 season_length, window_size, min_samples);
  }
  py::array_t<T> SeasonalRollingQuantileUpdate(int lag, T p, int season_length,
                                               int window_size,
                                               int min_samples) {
    py::array_t<T> out(NumGroups());
    Reduce(rolling::SeasonalQuantileUpdate<T>, 1, out.mutable_data(), lag,
           season_length, window_size, min_samples, p);
    return out;
  }

  std::tuple<py::array_t<T>, py::array_t<T>> ExpandingMeanTransform(int lag) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg(NumGroups());
    TransformAndReduce(expanding::MeanTransform<T>, lag, out.mutable_data(), 1,
                       agg.mutable_data());
    return std::make_tuple(out, agg);
  }
  std::tuple<py::array_t<T>, py::array_t<T>> ExpandingStdTransform(int lag) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg({static_cast<int>(NumGroups()), 3});
    TransformAndReduce(expanding::StdTransform<T>, lag, out.mutable_data(), 3,
                       agg.mutable_data());
    return std::make_tuple(out, agg);
  }
  py::array_t<T> ExpandingMinTransform(int lag) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MinTransform<T>, lag, out.mutable_data());
    return out;
  }
  py::array_t<T> ExpandingMaxTransform(int lag) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MaxTransform<T>, lag, out.mutable_data());
    return out;
  }
  py::array_t<T> ExpandingQuantileTransform(int lag, T p) {
    py::array_t<T> out(data_.size());
    Transform(expanding::QuantileTransform<T>, lag, out.mutable_data(), p);
    return out;
  }
  py::array_t<T> ExpandingQuantileUpdate(int lag, T p) {
    py::array_t<T> out(NumGroups());
    Reduce(expanding::QuantileUpdate<T>, 1, out.mutable_data(), lag, p);
    return out;
  }

  py::array_t<T> ExponentiallyWeightedMeanTransform(int lag, T alpha) {
    py::array_t<T> out(data_.size());
    Transform(exponentially_weighted::MeanTransform<T>, lag, out.mutable_data(),
              alpha);
    return out;
  }

  template <typename Func> py::array_t<T> ScalerStats(Func func) {
    py::array_t<T> out({static_cast<int>(NumGroups()), 2});
    Reduce(func, 2, out.mutable_data(), 0);
    return out;
  }
  py::array_t<T> MinMaxScalerStats() {
    return ScalerStats(scalers::MinMaxScalerStats<T>);
  }
  py::array_t<T> StandardScalerStats() {
    return ScalerStats(scalers::StandardScalerStats<T>);
  }
  py::array_t<T> RobustIqrScalerStats() {
    return ScalerStats(scalers::RobustScalerIqrStats<T>);
  }
  py::array_t<T> RobustMadScalerStats() {
    return ScalerStats(scalers::RobustScalerMadStats<T>);
  }
  py::array_t<T> ApplyScaler(const py::array_t<T> stats) {
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::CommonScalerTransform<T>, stats.data(),
                    out.mutable_data());
    return out;
  }
  py::array_t<T> InvertScaler(const py::array_t<T> stats) {
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
  py::array_t<T> BoxCoxTransform(const py::array_t<T> lambdas) {
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::BoxCoxTransform<T>, lambdas.data(),
                    out.mutable_data());
    return out;
  }
  py::array_t<T> BoxCoxInverseTransform(const py::array_t<T> lambdas) {
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
  py::array_t<T> NumSeasDiffsPeriods(int max_d, py::array_t<T> periods) {
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
    py::array_t<T> out(data_.size());
    Transform(seasonal::Difference<T>, 0, out.mutable_data(), d);
    return out;
  }
  py::array_t<T> Differences(const py::array_t<indptr_t> ds) {
    py::array_t<T> out(data_.size());
    VariableTransform(diff::Differences<T>, ds.data(), out.mutable_data());
    return out;
  }
  py::array_t<T> InvertDifference(int d, const py::array_t<T> tails) {
    py::array_t<indptr_t> ds(NumGroups());
    std::fill(ds.mutable_data(), ds.mutable_data() + ds.size(), d);
    return InvertDifferences(ds, tails);
  }
  py::array_t<T> InvertDifferences(const py::array_t<indptr_t> ds,
                                   const py::array_t<T> tails) {
    py::array_t<indptr_t> tails_indptr(indptr_.size());
    auto ds_data = ds.data();
    auto tails_indptr_data = tails_indptr.mutable_data();
    tails_indptr_data[0] = 0;
    for (int i = 1; i < tails_indptr.size(); ++i) {
      tails_indptr_data[i] = tails_indptr_data[i - 1] + ds_data[i - 1];
    }
    auto tails_ga = GroupedArray<T>(tails, tails_indptr, num_threads_);
    py::array_t<T> out(data_.size());
    Zip(diff::InvertDifference<T>, tails_ga, indptr_.data(),
        out.mutable_data());
    return out;
  }
};

template <typename T> void bind_ga(py::module &m, const std::string &name) {
  py::class_<GroupedArray<T>>(m, name.c_str())
      .def(py::init<const py::array_t<T> &, const py::array_t<indptr_t> &,
                    int>())
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
      .def("_rolling_mean", &GroupedArray<T>::RollingMeanTransform)
      .def("_rolling_std", &GroupedArray<T>::RollingStdTransform)
      .def("_rolling_min", &GroupedArray<T>::RollingMinTransform)
      .def("_rolling_max", &GroupedArray<T>::RollingMaxTransform)
      .def("_rolling_quantile", &GroupedArray<T>::RollingQuantileTransform)
      .def("_rolling_mean_update", &GroupedArray<T>::RollingMeanUpdate)
      .def("_rolling_std_update", &GroupedArray<T>::RollingStdUpdate)
      .def("_rolling_min_update", &GroupedArray<T>::RollingMinUpdate)
      .def("_rolling_max_update", &GroupedArray<T>::RollingMaxUpdate)
      .def("_rolling_quantile_update", &GroupedArray<T>::RollingQuantileUpdate)
      .def("_seasonal_rolling_mean",
           &GroupedArray<T>::SeasonalRollingMeanTransform)
      .def("_seasonal_rolling_std",
           &GroupedArray<T>::SeasonalRollingStdTransform)
      .def("_seasonal_rolling_min",
           &GroupedArray<T>::SeasonalRollingMinTransform)
      .def("_seasonal_rolling_max",
           &GroupedArray<T>::SeasonalRollingMaxTransform)
      .def("_seasonal_rolling_quantile",
           &GroupedArray<T>::SeasonalRollingQuantileTransform)
      .def("_seasonal_rolling_mean_update",
           &GroupedArray<T>::SeasonalRollingMeanUpdate)
      .def("_seasonal_rolling_std_update",
           &GroupedArray<T>::SeasonalRollingStdUpdate)
      .def("_seasonal_rolling_min_update",
           &GroupedArray<T>::SeasonalRollingMinUpdate)
      .def("_seasonal_rolling_max_update",
           &GroupedArray<T>::SeasonalRollingMaxUpdate)
      .def("_seasonal_rolling_quantile_update",
           &GroupedArray<T>::SeasonalRollingQuantileUpdate)
      .def("_expanding_mean", &GroupedArray<T>::ExpandingMeanTransform)
      .def("_expanding_std", &GroupedArray<T>::ExpandingStdTransform)
      .def("_expanding_min", &GroupedArray<T>::ExpandingMinTransform)
      .def("_expanding_max", &GroupedArray<T>::ExpandingMaxTransform)
      .def("_expanding_quantile", &GroupedArray<T>::ExpandingQuantileTransform)
      .def("_expanding_quantile_update",
           &GroupedArray<T>::ExpandingQuantileUpdate)
      .def("_exponentially_weighted_mean",
           &GroupedArray<T>::ExponentiallyWeightedMeanTransform)
      .def("_minmax_stats", &GroupedArray<T>::MinMaxScalerStats)
      .def("_standard_stats", &GroupedArray<T>::StandardScalerStats)
      .def("_robust_iqr_stats", &GroupedArray<T>::RobustIqrScalerStats)
      .def("_robust_mad_stats", &GroupedArray<T>::RobustMadScalerStats)
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

void init_ga(py::module_ &m) {
  py::module_ ga = m.def_submodule("grouped_array");
  bind_ga<float>(ga, "_GroupedArrayFloat32");
  bind_ga<double>(ga, "_GroupedArrayFloat64");
  ga.def(
      "GroupedArray",
      [](py::array data,
         py::array_t<indptr_t, py::array::c_style | py::array::forcecast>
             indptr,
         int num_threads) -> py::object {
        if (data.ndim() != 1) {
          throw std::invalid_argument("data must be a 1d array");
        }
        if (indptr.ndim() != 1) {
          throw std::invalid_argument("indptr must be a 1d array");
        }
        indptr_t last_indptr = indptr.data()[indptr.size() - 1];
        if (data.size() != last_indptr) {
          throw std::invalid_argument(
              "Last element of indptr must be equal to the size of data");
        }
        data = py::array::ensure(data, py::array::c_style);
        if (data.dtype().kind() != 'f') {
          data = data.attr("astype")("float32");
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
