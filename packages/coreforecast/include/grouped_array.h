#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

using indptr_t = int32_t;

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
    ++i;
  }
  return i;
}

template <typename T> inline void SkipLags(T *out, int n, int lag) {
  int replacements = std::min(lag, n);
  for (int i = 0; i < replacements; ++i) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
}

template <class T> class GroupedArray {
private:
  const T *data_;
  const indptr_t *indptr_;
  int n_groups_;
  int num_threads_;

public:
  GroupedArray(const T *data, const indptr_t *indptr, int n_indptr,
               int num_threads)
      : data_(data), indptr_(indptr), n_groups_(n_indptr - 1),
        num_threads_(num_threads) {}

  template <typename Func> void Parallelize(Func f) const noexcept {
    std::vector<std::thread> threads;
    int groups_per_thread = n_groups_ / num_threads_;
    int remainder = n_groups_ % num_threads_;
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
    Parallelize([data = data_, indptr = indptr_, &f, n_out, out, lag,
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
    Parallelize([data = data_, indptr = indptr_, &f, indptr_out, out,
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
    Parallelize([data = data_, indptr = indptr_, &f, stats,
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
    Parallelize([data = data_, indptr = indptr_, &f, lag, out,
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
    Parallelize([data = data_, indptr = indptr_, &f, params,
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
    Parallelize([data = data_, indptr = indptr_, &f, lag, out, n_agg, agg,
                 &args...](int start_group, int end_group) {
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
    Parallelize([data = data_, indptr = indptr_, &f, other_data = other.data_,
                 other_indptr = other.indptr_, out_indptr,
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
};
