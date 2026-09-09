#include <algorithm>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bindings.h"
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
template <typename T> inline void RequireOffsets(std::span<const T> values) {
  for (index_t i = 0; i < std::ssize(values); ++i) {
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
inline void CheckDs(const CArray<index_t> &ds, index_t n_groups) {
  if (ds.size() != n_groups) {
    throw std::invalid_argument("ds must have one element per group");
  }
  for (index_t d : View(ds)) {
    RequireNonNegative("d", d);
  }
}

// Two entries per group, read as stats[2 * i] and stats[2 * i + 1], so anything
// shorter is read past its end.
template <typename T>
inline void CheckStats(const CArray<T> &stats, index_t n_groups) {
  if (stats.ndim() != 2 || stats.size() != 2 * n_groups) {
    throw std::invalid_argument("stats must have shape (n_groups, 2)");
  }
}

// NaN over the first `lag` positions of a group's output (or all of it).
template <typename T> inline void SkipLags(std::span<T> out, index_t lag) {
  FillNaN(out.first(std::min<index_t>(lag, std::ssize(out))));
}

template <typename T> class GroupedArray {
public:
  const py::array_t<T> data_;
  const py::array_t<index_t> indptr_;
  int num_threads_;

  GroupedArray(const py::array_t<T> data, const py::array_t<index_t> indptr,
               int num_threads)
      : data_(data), indptr_(indptr), num_threads_(num_threads) {}

  index_t NumGroups() const noexcept { return indptr_.size() - 1; }

  std::span<const T> Data() const { return View(data_); }
  std::span<const index_t> Indptr() const { return View(indptr_); }

  // The elements of group i.
  static std::span<const T> Group(std::span<const T> data,
                                  std::span<const index_t> indptr, index_t i) {
    return data.subspan(indptr[i], indptr[i + 1] - indptr[i]);
  }
  static std::span<T> Group(std::span<T> data, std::span<const index_t> indptr,
                            index_t i) {
    return data.subspan(indptr[i], indptr[i + 1] - indptr[i]);
  }

  py::array_t<T> operator[](int i) const {
    const index_t num_groups = NumGroups();
    if (i >= num_groups) {
      throw std::out_of_range("Index out of range");
    }
    if (i < 0) {
      if (i < -num_groups) {
        throw std::out_of_range("Index out of range");
      }
      i += num_groups;
    }
    const index_t start = indptr_.data()[i];
    const index_t end = indptr_.data()[i + 1];
    auto buffer = data_.request();
    return py::array_t<T>({end - start}, {buffer.strides[0]},
                          static_cast<T *>(buffer.ptr) + start, data_);
  }

  py::array_t<T> Take(const CArray<index_t> indices_arg) const {
    const auto indices = View(indices_arg);
    const auto data = Data();
    const auto indptr = Indptr();
    const index_t num_groups = NumGroups();
    index_t out_size = 0;
    for (index_t idx : indices) {
      if (idx < 0 || idx >= num_groups) {
        throw std::out_of_range("Index out of range");
      }
      out_size += indptr[idx + 1] - indptr[idx];
    }
    py::array_t<T> out(out_size);
    auto out_view = MutableView(out);
    index_t j = 0;
    for (index_t idx : indices) {
      const auto group = Group(data, indptr, idx);
      std::copy(group.begin(), group.end(), out_view.begin() + j);
      j += std::ssize(group);
    }
    return out;
  }

  // The workers only touch raw buffers, never the Python API, so the GIL is
  // released around them. Exceptions can't cross a thread boundary, so each
  // worker stashes its own and we rethrow once everything has been joined.
  template <typename Func> void ForEach(Func f) const {
    const index_t n_groups = NumGroups();
    const int n_threads = static_cast<int>(
        std::clamp<index_t>(num_threads_, 1, std::max<index_t>(1, n_groups)));
    if (n_threads < 2) {
      py::gil_scoped_release release;
      f(0, n_groups);
      return;
    }
    std::vector<std::exception_ptr> errors(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    const index_t groups_per_thread = n_groups / n_threads;
    const index_t remainder = n_groups % n_threads;
    std::exception_ptr spawn_error;
    {
      py::gil_scoped_release release;
      try {
        for (int t = 0; t < n_threads; ++t) {
          const index_t start_group =
              t * groups_per_thread + std::min<index_t>(t, remainder);
          const index_t end_group =
              (t + 1) * groups_per_thread + std::min<index_t>(t + 1, remainder);
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

  // One row of n_out results per group, from the group minus its leading NaN
  // run and its last `lag` elements. An empty remainder gets a NaN row.
  template <typename Func, typename... Args>
  void Reduce(Func f, index_t n_out, std::span<T> out, int lag,
              Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = Data(), indptr = Indptr(), &f, n_out, out, lag,
             &args...](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        const auto group = Group(data, indptr, i);
        const auto row = out.subspan(n_out * i, n_out);
        const index_t n = std::ssize(group);
        const index_t start_idx = FirstNotNaN(group);
        if (start_idx + lag >= n) {
          FillNaN(row);
          continue;
        }
        f(group.subspan(start_idx, n - start_idx - lag), row,
          std::forward<Args>(args)...);
      }
    });
  }

  // Like Reduce with a per-group output size given by indptr_out, and no NaN
  // or lag handling.
  template <typename Func, typename... Args>
  void VariableReduce(Func f, std::span<const index_t> indptr_out,
                      std::span<T> out, Args &&...args) const {
    ForEach([data = Data(), indptr = Indptr(), &f, indptr_out, out,
             &args...](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        f(Group(data, indptr, i), Group(out, indptr_out, i),
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void ScalerTransform(Func f, std::span<const T> stats,
                       std::span<T> out) const {
    ForEach([data = Data(), indptr = Indptr(), &f, stats,
             out](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        const T offset = stats[2 * i];
        T scale = stats[2 * i + 1];
        if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
          scale = static_cast<T>(1.0);
        }
        for (index_t j = indptr[i]; j < indptr[i + 1]; ++j) {
          out[j] = f(data[j], offset, scale);
        }
      }
    });
  }

  // Same-size output per group. The leading NaN run and the first `lag`
  // positions after it are NaN; f fills the rest from the group without its
  // last `lag` elements.
  template <typename Func, typename... Args>
  void Transform(Func f, int lag, std::span<T> out, Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = Data(), indptr = Indptr(), &f, lag, out,
             &args...](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        const auto group = Group(data, indptr, i);
        const auto group_out = Group(out, indptr, i);
        const index_t n = std::ssize(group);
        const index_t start_idx = FirstNotNaN(group, group_out);
        SkipLags(group_out.subspan(start_idx), lag);
        if (start_idx + lag >= n) {
          continue;
        }
        f(group.subspan(start_idx, n - start_idx - lag),
          group_out.subspan(start_idx + lag), std::forward<Args>(args)...);
      }
    });
  }

  // Transform with one parameter per group and no lag.
  template <typename Func>
  void VariableTransform(Func f, std::span<const index_t> params,
                         std::span<T> out) const {
    ForEach([data = Data(), indptr = Indptr(), &f, params,
             out](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        const auto group = Group(data, indptr, i);
        const auto group_out = Group(out, indptr, i);
        const index_t start_idx = FirstNotNaN(group, group_out);
        if (start_idx >= std::ssize(group)) {
          continue;
        }
        f(group.subspan(start_idx), group_out.subspan(start_idx), params[i]);
      }
    });
  }

  // Transform that also leaves n_agg stats per group in agg.
  template <typename Func, typename... Args>
  void TransformAndReduce(Func f, int lag, std::span<T> out, index_t n_agg,
                          std::span<T> agg, Args &&...args) const {
    RequireNonNegative("lag", lag);
    ForEach([data = Data(), indptr = Indptr(), &f, lag, out, n_agg, agg,
             &args...](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        const auto group = Group(data, indptr, i);
        const auto group_out = Group(out, indptr, i);
        const auto agg_row = agg.subspan(i * n_agg, n_agg);
        const index_t n = std::ssize(group);
        const index_t start_idx = FirstNotNaN(group, group_out);
        SkipLags(group_out.subspan(start_idx), lag);
        if (start_idx + lag >= n) {
          // f never runs for an empty or all-NaN group, so its stats row would
          // otherwise be left uninitialised.
          FillNaN(agg_row);
          continue;
        }
        f(group.subspan(start_idx, n - start_idx - lag),
          group_out.subspan(start_idx + lag), agg_row,
          std::forward<Args>(args)...);
      }
    });
  }

  // f(group, other's group, output group), with the output groups delimited by
  // out_indptr.
  template <typename Func>
  void Zip(Func f, const GroupedArray<T> &other,
           std::span<const index_t> out_indptr, std::span<T> out) const {
    ForEach([data = Data(), indptr = Indptr(), &f, other_data = other.Data(),
             other_indptr = other.Indptr(), out_indptr,
             out](index_t start_group, index_t end_group) {
      for (index_t i = start_group; i < end_group; ++i) {
        f(Group(data, indptr, i), Group(other_data, other_indptr, i),
          Group(out, out_indptr, i));
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
    Reduce(grouped_array_functions::IndexFromEnd<T>, 1, MutableView(out), 0, k);
    return out;
  }
  py::array_t<T> Head(int k) {
    RequireNonNegative("k", k);
    py::array_t<T> out(k * NumGroups());
    Reduce(grouped_array_functions::Head<T>, k, MutableView(out), 0, k);
    return out;
  }
  py::array_t<T> Tail(int k) {
    RequireNonNegative("k", k);
    py::array_t<T> out(k * NumGroups());
    Reduce(grouped_array_functions::Tail<T>, k, MutableView(out), 0, k);
    return out;
  }
  std::unique_ptr<GroupedArray<T>> Append(const GroupedArray<T> &other) {
    if (NumGroups() != other.NumGroups()) {
      throw std::invalid_argument("Number of groups must be the same");
    }
    py::array_t<T> out_data(data_.size() + other.data_.size());
    py::array_t<index_t> out_indptr(indptr_.size());
    std::transform(indptr_.data(), indptr_.data() + indptr_.size(),
                   other.indptr_.data(), out_indptr.mutable_data(),
                   std::plus<index_t>());
    Zip(grouped_array_functions::Append<T>, other, View(out_indptr),
        MutableView(out_data));
    return std::make_unique<GroupedArray<T>>(out_data, out_indptr,
                                             num_threads_);
  }
  py::array_t<T> Tails(const CArray<index_t> out_indptr) {
    if (out_indptr.size() != indptr_.size()) {
      throw std::invalid_argument(
          "indptr must have one element per group plus one");
    }
    const auto offsets = View(out_indptr);
    RequireOffsets(offsets);
    // the output is sized from the last offset, so a non-zero first one would
    // leave the elements before it uninitialized
    if (offsets[0] != 0) {
      throw std::invalid_argument("First element of indptr must be zero");
    }
    py::array_t<T> out(offsets[NumGroups()]);
    // each group's tail is as long as its output slot
    VariableReduce(
        [](std::span<const T> in, std::span<T> tail) {
          grouped_array_functions::Tail(in, tail, std::ssize(tail));
        },
        offsets, MutableView(out));
    return out;
  }

  py::array_t<T> LagTransform(int lag) {
    py::array_t<T> out(data_.size());
    Transform(lag::LagTransform<T>, lag, MutableView(out));
    return out;
  }

  std::tuple<py::array_t<T>, py::array_t<T>>
  ExpandingMeanTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg(NumGroups());
    TransformAndReduce(expanding::MeanTransform<T>, lag, MutableView(out), 1,
                       MutableView(agg), skipna);
    return std::make_tuple(out, agg);
  }
  std::tuple<py::array_t<T>, py::array_t<T>>
  ExpandingStdTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    py::array_t<T> agg({NumGroups(), index_t{3}});
    TransformAndReduce(expanding::StdTransform<T>, lag, MutableView(out), 3,
                       MutableView(agg), skipna);
    return std::make_tuple(out, agg);
  }
  py::array_t<T> ExpandingMinTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MinTransform<T>, lag, MutableView(out), skipna);
    return out;
  }
  py::array_t<T> ExpandingMaxTransform(int lag, bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(expanding::MaxTransform<T>, lag, MutableView(out), skipna);
    return out;
  }
  py::array_t<T> ExpandingQuantileTransform(int lag, T p, bool skipna = false) {
    RequireProbability("p", p);
    py::array_t<T> out(data_.size());
    Transform(expanding::QuantileTransform<T>, lag, MutableView(out), p,
              skipna);
    return out;
  }
  py::array_t<T> ExpandingQuantileUpdate(int lag, T p, bool skipna = false) {
    RequireProbability("p", p);
    py::array_t<T> out(NumGroups());
    Reduce(expanding::QuantileUpdate<T>, 1, MutableView(out), lag, p, skipna);
    return out;
  }

  py::array_t<T> ExponentiallyWeightedMeanTransform(int lag, T alpha,
                                                    bool skipna = false) {
    py::array_t<T> out(data_.size());
    Transform(exponentially_weighted::MeanTransform<T>, lag, MutableView(out),
              alpha, skipna);
    return out;
  }

  template <typename Func>
  py::array_t<T> ScalerStats(Func func, bool skipna = false) {
    py::array_t<T> out({NumGroups(), index_t{2}});
    Reduce(func, 2, MutableView(out), 0, skipna);
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
    ScalerTransform(scalers::CommonScalerTransform<T>, View(stats),
                    MutableView(out));
    return out;
  }
  py::array_t<T> InvertScaler(const CArray<T> stats) {
    CheckStats(stats, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::CommonScalerInverseTransform<T>, View(stats),
                    MutableView(out));
    return out;
  }
  // The lambda kernels produce one value per group; the second column is
  // padding so box-cox stats share the (n_groups, 2) layout of the other
  // scalers, which is what CheckStats and the Python take/stack code expect.
  // Zeroed here because the kernel only writes column 0.
  py::array_t<T> LambdaStats() const {
    py::array_t<T> out({NumGroups(), index_t{2}});
    std::fill_n(out.mutable_data(), out.size(), T{0});
    return out;
  }
  py::array_t<T> BoxCoxLambdaGuerrero(int period, T lower, T upper) {
    RequirePositive("season_length", period);
    py::array_t<T> out = LambdaStats();
    Reduce(scalers::BoxCoxLambdaGuerrero<T>, 2, MutableView(out), 0, period,
           lower, upper);
    return out;
  }
  py::array_t<T> BoxCoxLambdaLogLik(T lower, T upper) {
    py::array_t<T> out = LambdaStats();
    Reduce(scalers::BoxCoxLambdaLogLik<T>, 2, MutableView(out), 0, lower,
           upper);
    return out;
  }
  py::array_t<T> BoxCoxTransform(const CArray<T> lambdas) {
    CheckStats(lambdas, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::BoxCoxTransform<T>, View(lambdas),
                    MutableView(out));
    return out;
  }
  py::array_t<T> BoxCoxInverseTransform(const CArray<T> lambdas) {
    CheckStats(lambdas, NumGroups());
    py::array_t<T> out(data_.size());
    ScalerTransform(scalers::BoxCoxInverseTransform<T>, View(lambdas),
                    MutableView(out));
    return out;
  }

  py::array_t<T> NumDiffs(int max_d) {
    py::array_t<T> out(NumGroups());
    Reduce(diff::NumDiffs<T>, 1, MutableView(out), 0, max_d);
    return out;
  }
  py::array_t<T> NumSeasDiffs(int period, int max_d) {
    RequireNonNegative("season_length", period);
    py::array_t<T> out(NumGroups());
    Reduce(diff::NumSeasDiffs<T>, 1, MutableView(out), 0, period, max_d);
    return out;
  }
  py::array_t<T> NumSeasDiffsPeriods(int max_d, const CArray<T> periods) {
    if (periods.size() != NumGroups()) {
      throw std::invalid_argument("periods must have one element per group");
    }
    // the period rides along in column 0 of the output row so the kernel can
    // see its group's value. It is checked as a float rather than cast: a NaN
    // period (from a group Periods could not size) passes and the kernel
    // turns it into a NaN count.
    py::array_t<T> periods_and_out({NumGroups(), index_t{2}});
    const auto rows = MutableView(periods_and_out);
    const auto periods_view = View(periods);
    for (index_t i = 0; i < NumGroups(); ++i) {
      RequireNonNegative("season_length", periods_view[i]);
      rows[2 * i] = periods_view[i];
    }
    Reduce(diff::NumSeasDiffsPeriods<T>, 2, rows, 0, max_d);
    py::array_t<T> out(NumGroups());
    const auto out_view = MutableView(out);
    for (index_t i = 0; i < NumGroups(); ++i) {
      out_view[i] = rows[2 * i + 1];
    }
    return out;
  }
  py::array_t<T> Periods(int max_lag) {
    RequireNonNegative("max_season_length", max_lag);
    py::array_t<T> out(NumGroups());
    Reduce(seasonal::GreatestAutocovariance<T>, 1, MutableView(out), 0,
           max_lag);
    return out;
  }
  py::array_t<T> Difference(int d) {
    RequireNonNegative("d", d);
    py::array_t<T> out(data_.size());
    Transform(seasonal::Difference<T>, 0, MutableView(out), d);
    return out;
  }
  py::array_t<T> Differences(const CArray<index_t> ds) {
    CheckDs(ds, NumGroups());
    py::array_t<T> out(data_.size());
    VariableTransform(diff::Differences<T>, View(ds), MutableView(out));
    return out;
  }
  py::array_t<T> InvertDifference(int d, const CArray<T> tails) {
    py::array_t<index_t> ds(NumGroups());
    std::fill(ds.mutable_data(), ds.mutable_data() + ds.size(), d);
    return InvertDifferences(ds, tails);
  }
  py::array_t<T> InvertDifferences(const CArray<index_t> ds,
                                   const CArray<T> tails) {
    CheckDs(ds, NumGroups());
    py::array_t<index_t> tails_indptr(indptr_.size());
    const auto ds_view = View(ds);
    const auto tails_offsets = MutableView(tails_indptr);
    tails_offsets[0] = 0;
    for (index_t i = 1; i < std::ssize(tails_offsets); ++i) {
      tails_offsets[i] = tails_offsets[i - 1] + ds_view[i - 1];
    }
    // tails_ga is built here rather than through the factory, so the check the
    // factory would have done on its last offset has to happen here.
    if (tails_offsets[NumGroups()] != tails.size()) {
      throw std::invalid_argument(
          "tails must have as many elements as the sum of ds");
    }
    auto tails_ga = GroupedArray<T>(tails, tails_indptr, num_threads_);
    py::array_t<T> out(data_.size());
    Zip(diff::InvertDifference<T>, tails_ga, Indptr(), MutableView(out));
    return out;
  }
};

// Validates indptr at the Python boundary. Every entry is used as an offset
// into data, so all of them have to describe a non-decreasing range, not just
// the last one. Takes py::object so that lists, tuples and Series keep
// working, which py::array would reject.
inline py::array_t<index_t> CheckedIndptr(const py::object &indptr,
                                          py::ssize_t data_size) {
  auto out = CArray<index_t>::ensure(indptr);
  if (!out) {
    throw std::invalid_argument("indptr must be an integer array");
  }
  if (out.ndim() != 1) {
    throw std::invalid_argument("indptr must be a 1d array");
  }
  if (out.size() < 1) {
    throw std::invalid_argument("indptr must have at least one element");
  }
  const auto values = View(out);
  RequireOffsets(values);
  if (data_size != values.back()) {
    throw std::invalid_argument(
        "Last element of indptr must be equal to the size of data");
  }
  return out;
}

template <typename T>
py::class_<GroupedArray<T>> BindGroupedArray(py::module_ &m,
                                             const std::string &name) {
  return py::class_<GroupedArray<T>>(m, name.c_str())
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

// ---------------------------------------------------------------------------
// Rolling family: one transform kernel per statistic; update, seasonal and
// seasonal update are derived from it through the combinators in rolling.h.
// Each statistic gets two free functions and four GroupedArray methods, for
// each dtype, from the two binders below.
// ---------------------------------------------------------------------------

template <typename T> py::array_t<T> Alloc(const GroupedArray<T> &ga) {
  return py::array_t<T>(ga.data_.size());
}
template <typename T> py::array_t<T> AllocPerGroup(const GroupedArray<T> &ga) {
  return py::array_t<T>(ga.NumGroups());
}

template <typename T, typename Tfm>
void BindRolling(py::module_ &roll, py::class_<GroupedArray<T>> &ga,
                 const std::string &stat, Tfm transform) {
  using In = std::span<const T>;
  using Out = std::span<T>;
  auto update = [transform](In in, Out out, const Window &w) {
    rolling::Update(transform, in, out, w);
  };
  auto seasonal = [transform](In in, Out out, const SeasonalWindow &sw) {
    rolling::SeasonalTransform(transform, in, out, sw);
  };
  auto seasonal_update = [update](In in, Out out, const SeasonalWindow &sw) {
    rolling::SeasonalUpdate(update, in, out, sw);
  };

  roll.def(("rolling_" + stat).c_str(),
           [transform](const py::array_t<T> &data, int window_size,
                       int min_samples, bool skipna) {
             const Window w = Window::Checked(window_size, min_samples, skipna);
             return SkipLeadingNaN<T>(
                 data, [&](In in, Out out) { transform(in, out, w); });
           },
           "data"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  roll.def(("seasonal_rolling_" + stat).c_str(),
           [seasonal](const py::array_t<T> &data, int season_length,
                      int window_size, int min_samples, bool skipna) {
             const SeasonalWindow sw = SeasonalWindow::Checked(
                 season_length, window_size, min_samples, skipna);
             return SkipLeadingNaN<T>(
                 data, [&](In in, Out out) { seasonal(in, out, sw); });
           },
           "data"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
           "skipna"_a = false);

  // The window is checked before the output is allocated, so a bad one costs
  // nothing and is rejected even when no group would run.
  ga.def(("_rolling_" + stat).c_str(),
         [transform](const GroupedArray<T> &self, int lag, int window_size,
                     int min_samples, bool skipna) {
           const Window w = Window::Checked(window_size, min_samples, skipna);
           auto out = Alloc(self);
           self.Transform(transform, lag, MutableView(out), w);
           return out;
         },
         "lag"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  ga.def(("_rolling_" + stat + "_update").c_str(),
         [update](const GroupedArray<T> &self, int lag, int window_size,
                  int min_samples, bool skipna) {
           const Window w = Window::Checked(window_size, min_samples, skipna);
           auto out = AllocPerGroup(self);
           self.Reduce(update, 1, MutableView(out), lag, w);
           return out;
         },
         "lag"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  ga.def(("_seasonal_rolling_" + stat).c_str(),
         [seasonal](const GroupedArray<T> &self, int lag, int season_length,
                    int window_size, int min_samples, bool skipna) {
           const SeasonalWindow sw = SeasonalWindow::Checked(
               season_length, window_size, min_samples, skipna);
           auto out = Alloc(self);
           self.Transform(seasonal, lag, MutableView(out), sw);
           return out;
         },
         "lag"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
         "skipna"_a = false);
  ga.def(("_seasonal_rolling_" + stat + "_update").c_str(),
         [seasonal_update](const GroupedArray<T> &self, int lag,
                           int season_length, int window_size, int min_samples,
                           bool skipna) {
           const SeasonalWindow sw = SeasonalWindow::Checked(
               season_length, window_size, min_samples, skipna);
           auto out = AllocPerGroup(self);
           self.Reduce(seasonal_update, 1, MutableView(out), lag, sw);
           return out;
         },
         "lag"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
         "skipna"_a = false);
}

// Same six registrations for the quantile, whose kernel also takes p. It comes
// right after data / lag, the order the public Python functions use.
template <typename T>
void BindRollingQuantile(py::module_ &roll, py::class_<GroupedArray<T>> &ga) {
  using In = std::span<const T>;
  using Out = std::span<T>;
  auto transform = rolling::QuantileTransform<T>;
  auto update = [](In in, Out out, const Window &w, T p) {
    rolling::Update(rolling::QuantileTransform<T>, in, out, w, p);
  };
  auto seasonal = [](In in, Out out, const SeasonalWindow &sw, T p) {
    rolling::SeasonalTransform(rolling::QuantileTransform<T>, in, out, sw, p);
  };
  auto seasonal_update = [update](In in, Out out, const SeasonalWindow &sw,
                                  T p) {
    rolling::SeasonalUpdate(update, in, out, sw, p);
  };

  roll.def(
      "rolling_quantile",
      [transform](const py::array_t<T> &data, T p, int window_size,
                  int min_samples, bool skipna) {
        RequireProbability("p", p);
        const Window w = Window::Checked(window_size, min_samples, skipna);
        return SkipLeadingNaN<T>(
            data, [&](In in, Out out) { transform(in, out, w, p); });
      },
      "data"_a, "p"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  roll.def(
      "seasonal_rolling_quantile",
      [seasonal](const py::array_t<T> &data, T p, int season_length,
                 int window_size, int min_samples, bool skipna) {
        RequireProbability("p", p);
        const SeasonalWindow sw = SeasonalWindow::Checked(
            season_length, window_size, min_samples, skipna);
        return SkipLeadingNaN<T>(
            data, [&](In in, Out out) { seasonal(in, out, sw, p); });
      },
      "data"_a, "p"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
      "skipna"_a = false);

  ga.def(
      "_rolling_quantile",
      [transform](const GroupedArray<T> &self, int lag, T p, int window_size,
                  int min_samples, bool skipna) {
        RequireProbability("p", p);
        const Window w = Window::Checked(window_size, min_samples, skipna);
        auto out = Alloc(self);
        self.Transform(transform, lag, MutableView(out), w, p);
        return out;
      },
      "lag"_a, "p"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  ga.def(
      "_rolling_quantile_update",
      [update](const GroupedArray<T> &self, int lag, T p, int window_size,
               int min_samples, bool skipna) {
        RequireProbability("p", p);
        const Window w = Window::Checked(window_size, min_samples, skipna);
        auto out = AllocPerGroup(self);
        self.Reduce(update, 1, MutableView(out), lag, w, p);
        return out;
      },
      "lag"_a, "p"_a, "window_size"_a, "min_samples"_a, "skipna"_a = false);
  ga.def(
      "_seasonal_rolling_quantile",
      [seasonal](const GroupedArray<T> &self, int lag, T p, int season_length,
                 int window_size, int min_samples, bool skipna) {
        RequireProbability("p", p);
        const SeasonalWindow sw = SeasonalWindow::Checked(
            season_length, window_size, min_samples, skipna);
        auto out = Alloc(self);
        self.Transform(seasonal, lag, MutableView(out), sw, p);
        return out;
      },
      "lag"_a, "p"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
      "skipna"_a = false);
  ga.def(
      "_seasonal_rolling_quantile_update",
      [seasonal_update](const GroupedArray<T> &self, int lag, T p,
                        int season_length, int window_size, int min_samples,
                        bool skipna) {
        RequireProbability("p", p);
        const SeasonalWindow sw = SeasonalWindow::Checked(
            season_length, window_size, min_samples, skipna);
        auto out = AllocPerGroup(self);
        self.Reduce(seasonal_update, 1, MutableView(out), lag, sw, p);
        return out;
      },
      "lag"_a, "p"_a, "season_length"_a, "window_size"_a, "min_samples"_a,
      "skipna"_a = false);
}

template <typename T>
void BindRollingFamily(py::module_ &roll, py::class_<GroupedArray<T>> &ga) {
  BindRolling<T>(roll, ga, "mean", rolling::MeanTransform<T>);
  BindRolling<T>(roll, ga, "std", rolling::StdTransform<T>);
  BindRolling<T>(roll, ga, "min", rolling::MinTransform<T>);
  BindRolling<T>(roll, ga, "max", rolling::MaxTransform<T>);
  BindRollingQuantile<T>(roll, ga);
}

// ---------------------------------------------------------------------------
// Free functions over a single array for the remaining families. These are
// irregular (tuple returns, scalar results) so they stay explicit.
// ---------------------------------------------------------------------------

template <typename T, typename Func, typename... Args>
py::array_t<T> ExpandingOp(Func f, const py::array_t<T> data, Args... args) {
  return SkipLeadingNaN<T>(data, [&](std::span<const T> in, std::span<T> out) {
    f(in, out, std::forward<Args>(args)...);
  });
}

// The single-array versions have nothing to resume from, so they pass an empty
// agg and the kernels leave the stats out.
template <typename T>
py::array_t<T> ExpandingMean(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::MeanTransform<T>, data, std::span<T>{}, skipna);
}

template <typename T>
py::array_t<T> ExpandingStd(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::StdTransform<T>, data, std::span<T>{}, skipna);
}

template <typename T>
py::array_t<T> ExpandingMin(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::MinTransform<T>, data, skipna);
}

template <typename T>
py::array_t<T> ExpandingMax(const py::array_t<T> data, bool skipna = false) {
  return ExpandingOp(expanding::MaxTransform<T>, data, skipna);
}

template <typename T>
py::array_t<T> ExpandingQuantile(const py::array_t<T> data, T p,
                                 bool skipna = false) {
  RequireProbability("p", p);
  return ExpandingOp(expanding::QuantileTransform<T>, data, p, skipna);
}

template <typename T> void BindExpanding(py::module_ &m) {
  m.def("expanding_mean", &ExpandingMean<T>, "data"_a, "skipna"_a = false);
  m.def("expanding_std", &ExpandingStd<T>, "data"_a, "skipna"_a = false);
  m.def("expanding_min", &ExpandingMin<T>, "data"_a, "skipna"_a = false);
  m.def("expanding_max", &ExpandingMax<T>, "data"_a, "skipna"_a = false);
  m.def("expanding_quantile", &ExpandingQuantile<T>, "data"_a, "p"_a,
        "skipna"_a = false);
}

template <typename T>
py::array_t<T> ExponentiallyWeightedMean(const py::array_t<T> data, T alpha,
                                         bool skipna = false) {
  return SkipLeadingNaN<T>(data, [&](std::span<const T> in, std::span<T> out) {
    exponentially_weighted::MeanTransform<T>(in, out, alpha, skipna);
  });
}

template <typename T> void BindExponentiallyWeighted(py::module_ &m) {
  m.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<T>, "data"_a,
        "alpha"_a, "skipna"_a = false);
}

template <typename T>
T BoxCoxLambdaGuerrero(py::array_t<T> data, int period, T lower, T upper) {
  RequirePositive("season_length", period);
  T out;
  const auto x = AsContiguous(data);
  scalers::BoxCoxLambdaGuerrero(View(x), std::span<T>{&out, 1}, period, lower,
                                upper);
  return out;
}

template <typename T>
T BoxCoxLambdaLogLik(py::array_t<T> data, T lower, T upper) {
  T out;
  const auto x = AsContiguous(data);
  scalers::BoxCoxLambdaLogLik(View(x), std::span<T>{&out, 1}, lower, upper);
  return out;
}

template <typename T>
py::array_t<T> BoxCoxTransform(py::array_t<T> data, T lambda) {
  const auto in = AsContiguous(data);
  py::array_t<T> out(in.size());
  std::transform(
      in.data(), in.data() + in.size(), out.mutable_data(),
      [lambda](T x) { return scalers::BoxCoxTransform<T>(x, lambda, 0.0); });
  return out;
}

template <typename T>
py::array_t<T> BoxCoxInverseTransform(const py::array_t<T> data, T lambda) {
  const auto in = AsContiguous(data);
  py::array_t<T> out(in.size());
  std::transform(in.data(), in.data() + in.size(), out.mutable_data(),
                 [lambda](T x) {
                   return scalers::BoxCoxInverseTransform<T>(x, lambda, 0.0);
                 });
  return out;
}

template <typename T> void BindScalers(py::module_ &m) {
  m.def("boxcox_lambda_guerrero", &BoxCoxLambdaGuerrero<T>);
  m.def("boxcox_lambda_loglik", &BoxCoxLambdaLogLik<T>);
  m.def("boxcox", &BoxCoxTransform<T>);
  m.def("inv_boxcox", &BoxCoxInverseTransform<T>);
}

template <typename T> int NumDiffs(const py::array_t<T> data, int max_d) {
  T out;
  const auto x = AsContiguous(data);
  diff::NumDiffs(View(x), std::span<T>{&out, 1}, max_d);
  return static_cast<int>(out);
}

template <typename T>
int NumSeasDiffs(const py::array_t<T> data, int period, int max_d) {
  RequireNonNegative("season_length", period);
  T out;
  const auto x = AsContiguous(data);
  diff::NumSeasDiffs(View(x), std::span<T>{&out, 1}, period, max_d);
  return static_cast<int>(out);
}

template <typename T>
py::array_t<T> Difference(const py::array_t<T> data, int d) {
  RequireNonNegative("d", d);
  const auto x = AsContiguous(data);
  py::array_t<T> out(x.size());
  seasonal::Difference(View(x), MutableView(out), d);
  return out;
}

template <typename T> void BindDifferences(py::module_ &m) {
  m.def("num_diffs", &NumDiffs<T>);
  m.def("num_seas_diffs", &NumSeasDiffs<T>);
  m.def("diff", &Difference<T>);
}

template <typename T> int Period(const py::array_t<T> data, int max_lag) {
  RequireNonNegative("max_season_length", max_lag);
  T out;
  const auto x = AsContiguous(data);
  seasonal::GreatestAutocovariance(View(x), std::span<T>{&out, 1}, max_lag);
  return static_cast<int>(out);
}

template <typename T> void BindSeasonal(py::module_ &m) {
  m.def("period", &Period<T>);
}

} // namespace

// Everything is registered for double before float: a non-float input matches
// no overload without conversion, so pybind11 converts it to the first one
// registered, and that has to be float64 (see test_dtype_coercion.py).
PYBIND11_MODULE(_lib, m) {
  py::module_ ga_mod = m.def_submodule("grouped_array");
  py::module_ roll = m.def_submodule("rolling");
  py::module_ exp = m.def_submodule("expanding");
  py::module_ ew = m.def_submodule("exponentially_weighted");
  py::module_ sc = m.def_submodule("scalers");
  py::module_ diffs = m.def_submodule("differences");
  py::module_ seas = m.def_submodule("seasonal");

  auto ga64 = BindGroupedArray<double>(ga_mod, "_GroupedArrayFloat64");
  auto ga32 = BindGroupedArray<float>(ga_mod, "_GroupedArrayFloat32");
  BindRollingFamily<double>(roll, ga64);
  BindRollingFamily<float>(roll, ga32);
  BindExpanding<double>(exp);
  BindExpanding<float>(exp);
  BindExponentiallyWeighted<double>(ew);
  BindExponentiallyWeighted<float>(ew);
  BindScalers<double>(sc);
  BindScalers<float>(sc);
  BindDifferences<double>(diffs);
  BindDifferences<float>(diffs);
  BindSeasonal<double>(seas);
  BindSeasonal<float>(seas);

  ga_mod.def(
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
