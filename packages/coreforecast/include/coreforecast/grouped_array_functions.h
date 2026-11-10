#pragma once

#include <algorithm>
#include <span>

#include "common.h"

namespace grouped_array_functions {
template <typename T>
inline void IndexFromEnd(std::span<const T> data, std::span<T> out, index_t k) {
  const index_t n = std::ssize(data);
  out[0] = k >= n ? kNaN<T> : data[n - 1 - k];
}

// out has k elements; the ones data can't fill are NaN
template <typename T>
inline void Head(std::span<const T> data, std::span<T> out, index_t k) {
  const index_t m = std::min<index_t>(k, std::ssize(data));
  std::copy_n(data.begin(), m, out.begin());
  FillNaN(out.subspan(m));
}

template <typename T>
inline void Tail(std::span<const T> data, std::span<T> out, index_t k) {
  const index_t m = std::min<index_t>(k, std::ssize(data));
  FillNaN(out.first(k - m));
  std::copy_n(data.end() - m, m, out.begin() + (k - m));
}

template <typename T>
inline void Append(std::span<const T> data, std::span<const T> other,
                   std::span<T> out) {
  std::copy(data.begin(), data.end(), out.begin());
  std::copy(other.begin(), other.end(), out.begin() + std::ssize(data));
}
} // namespace grouped_array_functions
