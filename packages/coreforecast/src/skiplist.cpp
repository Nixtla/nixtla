// The vendored skip list declares these three free functions in its header and
// defines them in a translation unit we don't build, so that we can provide
// them here. Its tossCoin() draws from libc's rand, which glibc serialises on
// a process-wide lock (and the BSD libc doesn't guard at all), so every worker
// running a rolling quantile contended on it.

#include <cstdint>
#include <string>

#include "SkipList.h"

namespace OrderedStructs {
namespace SkipList {
namespace {
// Any non-zero value; xorshift is stuck at zero.
constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;

// Per thread, so node heights don't depend on what the other threads are doing
// and no two threads share the state.
thread_local std::uint64_t state = kDefaultSeed;

// xorshift64*
std::uint64_t Next() {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}
} // namespace

bool tossCoin() { return (Next() >> 63) != 0; }

// Seeds the calling thread only: the state is thread_local, so a seed set on
// the main thread doesn't reach the workers a multi-threaded rolling quantile
// spawns. Nothing calls this today.
void seedRand(unsigned seed) {
  // splitmix64, so neighbouring seeds don't start from neighbouring states
  // (seed * kDefaultSeed gave 0 and 1 the same stream). It's a bijection whose
  // one preimage of zero doesn't fit in an unsigned, so no seed sticks.
  std::uint64_t z = seed + kDefaultSeed;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  state = z ^ (z >> 31);
}

void _throw_exceeds_size(size_t /* index */) {
  throw IndexError("Index out of range.");
}

} // namespace SkipList
} // namespace OrderedStructs
