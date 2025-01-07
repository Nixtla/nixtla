#pragma once

#include "SkipList.h"
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace stats {
template <std::contiguous_iterator Iterator>
auto Quantile(Iterator begin, Iterator end, std::iter_value_t<Iterator> p) {
  using T = std::iter_value_t<Iterator>;
  auto n = std::distance(begin, end);

  T i_plus_g = p * (n - 1);
  auto i = static_cast<size_t>(i_plus_g);
  T g = i_plus_g - i;

  auto nth = begin + i;
  std::nth_element(begin, nth, end);
  T out = *nth;

  if (g > T{0}) {
    auto min = std::min_element(nth + 1, end);
    out += g * (*min - out);
  }
  return out;
}

template <typename T>
auto SortedQuantile(OrderedStructs::SkipList::HeadNode<T> &data, T p,
                    size_t n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<size_t>(i_plus_g);
  T g = i_plus_g - i;
  T out = data.at(i);
  if (g > T{0.0}) {
    out += g * (data.at(i + 1) - out);
  }
  return out;
}
} // namespace stats
