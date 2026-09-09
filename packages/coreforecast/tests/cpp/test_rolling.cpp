#include "expanding.h"
#include "rolling.h"

#include "helpers.h"

using namespace helpers;

namespace {

// O(n * w) reference. `gate_on_count` mirrors a difference between the
// kernels that the tests pin rather than hide: with skipna, mean, min, max and
// quantile emit NaN for the first min_samples - 1 positions and then use
// whatever valid values the window holds, while std requires min_samples valid
// values in the window.
template <typename T, typename Stat>
std::vector<T> RefRolling(const std::vector<T> &data, int w, int ms,
                          bool skipna, bool gate_on_count, Stat stat) {
  const int n = static_cast<int>(data.size());
  std::vector<T> out(n, NaN<T>);
  if (n < ms) {
    return out;
  }
  w = std::min(w, n);
  ms = std::min(ms, w);
  for (int i = 0; i < n; ++i) {
    const int lo = std::max(0, i - w + 1);
    const auto vals = Valid(data, lo, i + 1, skipna);
    const bool ok =
        gate_on_count ? static_cast<int>(vals.size()) >= ms : i + 1 >= ms;
    if (ok && !vals.empty()) {
      out[i] = stat(vals);
    }
  }
  return out;
}

template <typename T> T Min(const std::vector<T> &v) {
  return *std::min_element(v.begin(), v.end());
}
template <typename T> T Max(const std::vector<T> &v) {
  return *std::max_element(v.begin(), v.end());
}

struct Shape {
  int size;
  int min_samples;
};
constexpr Shape kWindows[] = {{5, 5}, {5, 3}, {5, 1}, {3, 3}, {10, 2}};

} // namespace

TEST_CASE_TEMPLATE("rolling transforms match an O(n*w) reference", T, float,
                   double) {
  const int n = 24;
  const T p = static_cast<T>(0.3);
  for (bool skipna : {false, true}) {
    // interior NaN is only defined behaviour with skipna
    const auto data =
        skipna ? WithNaN(Random<T>(n), {0, 1, 7, 8, 9, 20}) : Random<T>(n);
    for (const auto [w, ms] : kWindows) {
      CAPTURE(skipna);
      CAPTURE(w);
      CAPTURE(ms);
      std::vector<T> out(n);

      rolling::MeanTransform(In(data), Out(out), Window{w, ms, skipna});
      CheckClose(out, RefRolling(data, w, ms, skipna, false, Mean<T>));

      rolling::StdTransform(In(data), Out(out), Window{w, ms, skipna});
      CheckClose(out, RefRolling(data, w, ms, skipna, true, Std<T>));

      rolling::MinTransform(In(data), Out(out), Window{w, ms, skipna});
      CheckClose(out, RefRolling(data, w, ms, skipna, false, Min<T>));

      rolling::MaxTransform(In(data), Out(out), Window{w, ms, skipna});
      CheckClose(out, RefRolling(data, w, ms, skipna, false, Max<T>));

      rolling::QuantileTransform(In(data), Out(out), Window{w, ms, skipna}, p);
      CheckClose(out,
                 RefRolling(data, w, ms, skipna, false,
                            [p](const auto &v) { return Quantile(v, p); }));
    }
  }
}

TEST_CASE_TEMPLATE(
    "rolling transforms on fewer samples than min_samples are NaN", T, float,
    double) {
  const auto data = Random<T>(2);
  std::vector<T> out(2, T{0});
  rolling::MeanTransform(In(data), Out(out), Window{5, 3, false});
  CheckClose(out, {NaN<T>, NaN<T>});
  rolling::QuantileTransform(In(data), Out(out), Window{5, 3, true}, T{0.5});
  CheckClose(out, {NaN<T>, NaN<T>});
}

TEST_CASE_TEMPLATE("min and max without skipna are NaN from the first NaN on",
                   T, float, double) {
  const std::vector<T> data = {1, 2, NaN<T>, 4, 5, 6};
  std::vector<T> out(6);
  const std::vector<T> want = {1, 1, NaN<T>, NaN<T>, NaN<T>, NaN<T>};
  rolling::MinTransform(In(data), Out(out), Window{2, 1, false});
  CheckClose(out, want);
  const std::vector<T> want_max = {1, 2, NaN<T>, NaN<T>, NaN<T>, NaN<T>};
  rolling::MaxTransform(In(data), Out(out), Window{2, 1, false});
  CheckClose(out, want_max);
}

TEST_CASE_TEMPLATE("update equals the last element of transform", T, float,
                   double) {
  const int n = 17;
  const T p = static_cast<T>(0.8);
  for (bool skipna : {false, true}) {
    const auto data =
        skipna ? WithNaN(Random<T>(n, 3), {0, 5, 6, 16}) : Random<T>(n, 3);
    for (const auto [w, ms] : kWindows) {
      CAPTURE(skipna);
      CAPTURE(w);
      CAPTURE(ms);
      std::vector<T> full(n);
      // NaN so a kernel that stops writing fails instead of comparing garbage
      T last = NaN<T>;

      rolling::MeanTransform(In(data), Out(full), Window{w, ms, skipna});
      rolling::Update(rolling::MeanTransform<T>, In(data), Out(last),
                      Window{w, ms, skipna});
      CheckClose(last, full[n - 1]);

      rolling::StdTransform(In(data), Out(full), Window{w, ms, skipna});
      rolling::Update(rolling::StdTransform<T>, In(data), Out(last),
                      Window{w, ms, skipna});
      CheckClose(last, full[n - 1]);

      rolling::MinTransform(In(data), Out(full), Window{w, ms, skipna});
      rolling::Update(rolling::MinTransform<T>, In(data), Out(last),
                      Window{w, ms, skipna});
      CheckClose(last, full[n - 1]);

      rolling::MaxTransform(In(data), Out(full), Window{w, ms, skipna});
      rolling::Update(rolling::MaxTransform<T>, In(data), Out(last),
                      Window{w, ms, skipna});
      CheckClose(last, full[n - 1]);

      rolling::QuantileTransform(In(data), Out(full), Window{w, ms, skipna}, p);
      rolling::Update(rolling::QuantileTransform<T>, In(data), Out(last),
                      Window{w, ms, skipna}, p);
      CheckClose(last, full[n - 1]);
    }
  }
}

TEST_CASE_TEMPLATE("seasonal rolling equals rolling applied per phase", T,
                   float, double) {
  const int n = 29;
  const int season = 4;
  const auto data = WithNaN(Random<T>(n, 11), {2, 3, 9});
  for (const auto [w, ms] : kWindows) {
    CAPTURE(w);
    CAPTURE(ms);
    std::vector<T> want(n);
    for (int phase = 0; phase < season; ++phase) {
      std::vector<T> sub;
      for (int i = phase; i < n; i += season) {
        sub.push_back(data[i]);
      }
      std::vector<T> sub_out(sub.size());
      rolling::MeanTransform(In(sub), Out(sub_out), Window{w, ms, true});
      for (size_t j = 0; j < sub.size(); ++j) {
        want[phase + j * season] = sub_out[j];
      }
    }
    std::vector<T> out(n);
    rolling::SeasonalTransform(rolling::MeanTransform<T>, In(data), Out(out),
                               SeasonalWindow{season, {w, ms, true}});
    CheckClose(out, want);

    T last = NaN<T>;
    auto update = [](std::span<const T> in, std::span<T> out, const Window &w) {
      rolling::Update(rolling::MeanTransform<T>, in, out, w);
    };
    rolling::SeasonalUpdate(update, In(data), Out(last),
                            SeasonalWindow{season, {w, ms, true}});
    CheckClose(last, want[n - 1]);
  }
}

TEST_CASE_TEMPLATE("expanding transforms are cumulative statistics", T, float,
                   double) {
  const int n = 15;
  for (bool skipna : {false, true}) {
    const auto data =
        skipna ? WithNaN(Random<T>(n, 5), {0, 4, 5}) : Random<T>(n, 5);
    CAPTURE(skipna);
    std::vector<T> out(n);
    std::vector<T> want(n);
    std::vector<T> agg(3);

    expanding::MeanTransform(In(data), Out(out), Out(agg), skipna);
    for (int i = 0; i < n; ++i) {
      const auto v = Valid(data, 0, i + 1, skipna);
      want[i] = v.empty() ? NaN<T> : Mean(v);
    }
    CheckClose(out, want);
    CHECK(agg[0] == static_cast<T>(Valid(data, 0, n, skipna).size()));
    // the single-array entry points pass no agg; the result is the same
    std::vector<T> without_agg(n);
    expanding::MeanTransform(In(data), Out(without_agg), std::span<T>{}, skipna);
    CheckClose(without_agg, out);

    expanding::StdTransform(In(data), Out(out), Out(agg), skipna);
    for (int i = 0; i < n; ++i) {
      want[i] = Std(Valid(data, 0, i + 1, skipna));
    }
    CheckClose(out, want);
    expanding::StdTransform(In(data), Out(without_agg), std::span<T>{}, skipna);
    CheckClose(without_agg, out);

    expanding::MinTransform(In(data), Out(out), skipna);
    for (int i = 0; i < n; ++i) {
      const auto v = Valid(data, 0, i + 1, skipna);
      want[i] = v.empty() ? NaN<T> : Min(v);
    }
    CheckClose(out, want);

    const T p = static_cast<T>(0.5);
    expanding::QuantileTransform(In(data), Out(out), p, skipna);
    for (int i = 0; i < n; ++i) {
      const auto v = Valid(data, 0, i + 1, skipna);
      want[i] = v.empty() ? NaN<T> : Quantile(v, p);
    }
    CheckClose(out, want);

    T last = NaN<T>;
    expanding::QuantileUpdate(In(data), Out(last), p, skipna);
    CheckClose(last, want[n - 1]);
  }
}
