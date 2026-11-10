#pragma once

#include <algorithm>
#include <span>

namespace lag {
template <typename T>
inline void LagTransform(std::span<const T> data, std::span<T> out) {
  std::copy(data.begin(), data.end(), out.begin());
}
} // namespace lag
