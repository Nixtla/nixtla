#include <vector>

#include "SkipList.h"

#include "doctest.h"

using OrderedStructs::SkipList::seedRand;
using OrderedStructs::SkipList::tossCoin;

namespace {

std::vector<bool> Tosses(int n) {
  std::vector<bool> out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = tossCoin();
  }
  return out;
}

} // namespace

TEST_CASE("tossCoin is a fair coin") {
  // A coin stuck on one side would still give correct quantiles: node heights
  // only decide how the skip list is searched, so it would silently degrade
  // into a linked list and no other test would notice.
  seedRand(7);
  const int n = 100'000;
  int heads = 0;
  for (bool b : Tosses(n)) {
    heads += b;
  }
  // five standard deviations around n / 2
  CHECK(heads > n / 2 - 800);
  CHECK(heads < n / 2 + 800);
}

TEST_CASE("seedRand restarts the stream and distinct seeds differ") {
  seedRand(3);
  const auto first = Tosses(64);
  seedRand(3);
  CHECK(Tosses(64) == first);
  // 0 and 1 collided when the state was seed times a constant
  seedRand(0);
  const auto zero = Tosses(64);
  seedRand(1);
  CHECK(Tosses(64) != zero);
}
