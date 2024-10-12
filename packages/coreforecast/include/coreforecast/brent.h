#pragma once

#include <cmath>

constexpr double GOLDEN_RATIO = 0.3819660;

template <typename Function, typename... Args>
double Brent(Function f, double a, double b, double tol, Args &&...args) {
  double d, e, p, q, r, u, v, w, x;
  double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

  eps = std::numeric_limits<double>::epsilon();
  tol1 = eps + 1.0;
  eps = std::sqrt(eps);

  v = a + GOLDEN_RATIO * (b - a);
  w = v;
  x = v;

  d = 0.0;
  e = 0.0;
  fx = f(x, std::forward<Args>(args)...);
  fv = fx;
  fw = fx;
  tol3 = tol / 3.0;

  while (true) {
    xm = (a + b) * 0.5;
    tol1 = eps * std::abs(x) + tol3;
    t2 = tol1 * 2.0;

    if (std::abs(x - xm) <= t2 - (b - a) * 0.5) {
      break;
    }

    p = 0.0;
    q = 0.0;
    r = 0.0;
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.0;
      if (q > 0.0) {
        p = -p;
      } else {
        q = -q;
      }
      r = e;
      e = d;
    }

    if (std::abs(p) >= std::abs(q * 0.5 * r) || p <= q * (a - x) ||
        p >= q * (b - x)) {
      if (x < xm) {
        e = b - x;
      } else {
        e = a - x;
      }
      d = GOLDEN_RATIO * e;
    } else {
      d = p / q;
      u = x + d;
      if (u - a < t2 || b - u < t2) {
        d = tol1;
        if (x >= xm) {
          d = -d;
        }
      }
    }

    if (std::abs(d) >= tol1) {
      u = x + d;
    } else if (d > 0.0) {
      u = x + tol1;
    } else {
      u = x - tol1;
    }

    fu = f(u, std::forward<Args>(args)...);

    if (fu <= fx) {
      if (u < x) {
        b = x;
      } else {
        a = x;
      }
      v = w;
      w = x;
      x = u;
      fv = fw;
      fw = fx;
      fx = fu;
    } else {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  return x;
}
