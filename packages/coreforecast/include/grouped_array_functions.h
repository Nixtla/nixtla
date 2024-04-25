#include "grouped_array.h"

template <typename T>
inline void IndexFromEnd(const T *data, int n, T *out, int k) {
  if (k > n) {
    *out = std::numeric_limits<T>::quiet_NaN();
  } else {
    *out = data[n - 1 - k];
  }
}

template <typename T> inline void Head(const T *data, int n, T *out, int k) {
  int m = std::min(k, n);
  std::copy(data, data + m, out);
  std::fill(out + m, out + k, std::numeric_limits<T>::quiet_NaN());
}

template <typename T> inline void Tail(const T *data, int n, T *out, int k) {
  int m = std::min(k, n);
  std::fill(out, out + k - m, std::numeric_limits<T>::quiet_NaN());
  std::copy(data + n - m, data + n, out + k - m);
}

template <typename T>
inline void Append(const T *data, int n, const T *other_data, int other_n,
                   T *out) {
  std::copy(data, data + n, out);
  std::copy(other_data, other_data + other_n, out + n);
}
