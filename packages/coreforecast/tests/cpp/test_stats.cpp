#include "brent.h"
#include "common.h"
#include "diff.h"
#include "kpss.h"
#include "scalers.h"
#include "seasonal.h"
#include "stats.h"

#include "helpers.h"

using namespace helpers;

TEST_CASE("FirstNotNaN finds the end of the leading run") {
  const std::vector<double> lead = {NaN<double>, NaN<double>, 1.0, NaN<double>};
  CHECK(FirstNotNaN(In(lead).first(4)) == 2);
  const std::vector<double> none = {1.0, 2.0};
  CHECK(FirstNotNaN(In(none).first(2)) == 0);
  const std::vector<double> all = {NaN<double>, NaN<double>};
  CHECK(FirstNotNaN(In(all).first(2)) == 2);
  CHECK(FirstNotNaN(In(all).first(0)) == 0);
  // the overload with an output writes NaN over the run it skips
  std::vector<double> out(4, 0.0);
  CHECK(FirstNotNaN(In(lead), Out(out)) == 2);
  CheckClose(out, {NaN<double>, NaN<double>, 0.0, 0.0});
}

TEST_CASE_TEMPLATE("Difference and InvertDifference round-trip", T, float,
                   double) {
  const int n = 20;
  const auto x = Random<T>(n, 9);
  for (int d : {1, 2, 7}) {
    CAPTURE(d);
    std::vector<T> diffs(n);
    seasonal::Difference(In(x), Out(diffs), d);
    for (int i = 0; i < d; ++i) {
      CHECK(std::isnan(diffs[i]));
    }
    // the inverse takes the differenced tail and the d values that preceded it
    std::vector<T> restored(n - d);
    diff::InvertDifference(In(diffs).subspan(d), In(x).first(d), Out(restored));
    CheckClose(restored, std::vector<T>(x.begin() + d, x.end()));
  }
  // d = 0 is a copy; d > n is all NaN
  std::vector<T> out(n);
  seasonal::Difference(In(x), Out(out), 0);
  CheckClose(out, x);
  seasonal::Difference(In(x), Out(out), n + 1);
  CheckClose(out, std::vector<T>(n, NaN<T>));
}

TEST_CASE_TEMPLATE("NumSeasDiffsPeriods reads the period as a float", T, float,
                   double) {
  // five seasons of a period-12 pattern on a trend
  std::vector<T> x(60);
  for (int i = 0; i < 60; ++i) {
    x[i] = static_cast<T>(i % 12 + i);
  }
  T want;
  diff::NumSeasDiffs(In(x), Out(want), 12, 1);
  CHECK(want == 1);
  std::vector<T> row = {12, 0};
  diff::NumSeasDiffsPeriods(In(x), Out(row), 1);
  CHECK(row[1] == want);
  // no period for the group gives no count, not a cast of NaN
  row = {NaN<T>, 0};
  diff::NumSeasDiffsPeriods(In(x), Out(row), 1);
  CHECK(std::isnan(row[1]));
  // a period the group cannot hold twice is zero without being cast
  for (T period : {T{61}, std::numeric_limits<T>::max(),
                   std::numeric_limits<T>::infinity()}) {
    CAPTURE(period);
    row = {period, NaN<T>};
    diff::NumSeasDiffsPeriods(In(x), Out(row), 1);
    CHECK(row[1] == 0);
  }
}

TEST_CASE("KPSS matches statsmodels") {
  // np.cumsum(np.random.default_rng(42).normal(size=64)); the expected values
  // are statsmodels.tsa.stattools.kpss(x, regression="c", nlags=k)[0]
  const std::vector<double> x = {
      0.30471707975443135, -0.7352670264860641, 0.015184169320393126,
      0.955748885711607,   -0.9952863029422294, -2.2974658098045477,
      -2.1696254066372624, -2.4858679989808445, -2.5026691564851333,
      -3.3557130840587135, -2.476315109195885,  -1.6985231737669366,
      -1.6324924762057207, -0.5052512692376878, -0.037741926985642216,
      -0.8970343898688804, -0.5282836057863816, -1.4871662066153806,
      -0.608715905308108,  -0.6586418162943609, -0.8435041798396214,
      -1.5244337242435628, -0.3018923855695326, -0.4564218676383347,
      -0.884749689801442,  -1.2368832402896714, -0.7045740547363227,
      -0.3391299903722444, 0.073602621223744,   0.5044236242316267,
      2.646071225102088,   2.2396562087174727,  1.7274134796459353,
      0.9136407513980576,  1.5296201739735533,  2.658592466694445,
      2.54464500903957,    1.704488532077042,   0.8800073163858024,
      1.5306001042105035,  2.273854275413946,   2.817008543719141,
      2.1514988364304464,  2.383660159497166,   2.500345968637894,
      2.7190345653669072,  3.590463343315097,   3.814058892089779,
      4.492972455161674,   4.560551524650565,   4.8496709233405495,
      5.48095914917909,    4.023803329323424,   3.704132112966122,
      3.233759458673327,   2.5948816104299848,  2.319739359203301,
      3.814680670437697,   2.948849554744454,   3.917127909335935,
      2.23425813772013,    1.8993731077343552,  2.0621261728393607,
      2.6483485041986388,
  };
  CHECK(KPSS(In(x), 0) == doctest::Approx(5.063007444851195).epsilon(1e-9));
  CHECK(KPSS(In(x), 2) == doctest::Approx(1.7993721714555229).epsilon(1e-9));
  CHECK(KPSS(In(x), 5) == doctest::Approx(0.9572458516286745).epsilon(1e-9));
  // lags past n - 1 have no pairs to sum and are clamped to it
  CHECK(KPSS(In(x), 1000) == KPSS(In(x), 63));
}

TEST_CASE("Quantile interpolates linearly like numpy") {
  std::vector<double> v = {4.0, 1.0, 3.0, 2.0};
  CHECK(stats::Quantile(Out(v), 0.5) == doctest::Approx(2.5));
  v = {4.0, 1.0, 3.0, 2.0};
  CHECK(stats::Quantile(Out(v), 0.25) == doctest::Approx(1.75));
  v = {4.0, 1.0, 3.0, 2.0};
  CHECK(stats::Quantile(Out(v), 0.0) == doctest::Approx(1.0));
  v = {4.0, 1.0, 3.0, 2.0};
  CHECK(stats::Quantile(Out(v), 1.0) == doctest::Approx(4.0));
  v = {4.0};
  CHECK(stats::Quantile(Out(v), 0.7) == doctest::Approx(4.0));
}

TEST_CASE("Quantile and SortedQuantile agree") {
  const auto data = Random<double>(31, 13);
  OrderedStructs::SkipList::HeadNode<double> list;
  for (auto x : data) {
    list.insert(x);
  }
  for (double p : {0.0, 0.1, 0.5, 0.9, 1.0}) {
    auto copy = data;
    CHECK(stats::SortedQuantile(list, p, data.size()) ==
          doctest::Approx(stats::Quantile(Out(copy), p)));
  }
}

TEST_CASE("Brent finds the minimum of a parabola") {
  auto f = [](double x, double c) { return (x - c) * (x - c); };
  CHECK(Brent(f, 0.0, 5.0, 1e-8, 2.0) == doctest::Approx(2.0).epsilon(1e-6));
  CHECK(Brent(f, -3.0, 0.0, 1e-8, -1.0) == doctest::Approx(-1.0).epsilon(1e-6));
}

TEST_CASE_TEMPLATE("BoxCox transforms round-trip", T, float, double) {
  const std::vector<T> xs = {0.5, 1.0, 2.0, 7.5};
  for (T lambda : {T{-0.5}, T{0}, T{0.5}, T{1}}) {
    for (T x : xs) {
      CAPTURE(lambda);
      CAPTURE(x);
      const T y = scalers::BoxCoxTransform<T>(x, lambda, T{0});
      const T back = scalers::BoxCoxInverseTransform<T>(y, lambda, T{0});
      CHECK(back ==
            doctest::Approx(x).epsilon(std::is_same_v<T, float> ? 1e-4 : 1e-9));
    }
  }
  // negative input with a negative lambda has no real transform
  CHECK(std::isnan(scalers::BoxCoxTransform<T>(T{-1}, T{-0.5}, T{0})));
  // lambda 0 is the log. static_cast rather than T{}: Clang doesn't fold
  // std::exp, so the braces would be a narrowing error for float.
  CHECK(scalers::BoxCoxTransform<T>(static_cast<T>(std::exp(1.0)), T{0},
                                    T{0}) == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE_TEMPLATE("scaler stats with skipna ignore NaN", T, float, double) {
  const std::vector<T> x = {NaN<T>, 1, 2, 3, 4, NaN<T>, 10};
  std::vector<T> stats(2);
  scalers::MinMaxScalerStats(In(x), Out(stats), true);
  CHECK(stats[0] == T{1});
  CHECK(stats[1] == T{9});
  scalers::StandardScalerStats(In(x), Out(stats), true);
  CHECK(stats[0] == doctest::Approx(4.0));
  CHECK(stats[1] == doctest::Approx(std::sqrt(10.0))); // population std
  scalers::RobustScalerIqrStats(In(x), Out(stats), true);
  CHECK(stats[0] == doctest::Approx(3.0));
  CHECK(stats[1] == doctest::Approx(2.0)); // q3 - q1 of {1,2,3,4,10}
  // nothing valid: both stats are NaN rather than left unwritten
  const std::vector<T> none = {NaN<T>, NaN<T>};
  stats[0] = stats[1] = T{0};
  scalers::MinMaxScalerStats(In(none), Out(stats), true);
  CHECK(std::isnan(stats[0]));
  CHECK(std::isnan(stats[1]));
}
