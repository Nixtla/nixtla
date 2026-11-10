#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <new>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "parallel.h"
#include "rolling.h"

#include "helpers.h"

using namespace helpers;

namespace {

// How many more allocations this thread makes before one throws
// std::bad_alloc, or -1 for none. Only the thread that sets it is affected.
thread_local int allocs_before_failure = -1;
thread_local bool alloc_failed = false;

} // namespace

void *operator new(std::size_t size) {
  if (allocs_before_failure == 0) {
    allocs_before_failure = -1;
    alloc_failed = true;
    throw std::bad_alloc();
  }
  if (allocs_before_failure > 0) {
    --allocs_before_failure;
  }
  if (void *p = std::malloc(size == 0 ? 1 : size)) {
    return p;
  }
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

namespace {

// n_groups groups of one element each, which makes every chunk weigh the same.
std::vector<index_t> FlatIndptr(index_t n_groups) {
  std::vector<index_t> indptr(n_groups + 1);
  std::iota(indptr.begin(), indptr.end(), index_t{0});
  return indptr;
}

// One counter per group, so a group left out or run twice shows up as a count
// other than one.
std::vector<int> VisitCounts(index_t n_groups, int n_threads) {
  std::vector<std::atomic<int>> visits(n_groups);
  const auto indptr = FlatIndptr(n_groups);
  parallel::ForEach(indptr, n_threads,
                    [&visits](index_t start_group, index_t end_group) {
                      for (index_t i = start_group; i < end_group; ++i) {
                        visits[i].fetch_add(1);
                      }
                    });
  return {visits.begin(), visits.end()};
}

void CheckAllVisitedOnce(index_t n_groups, int n_threads) {
  INFO("n_groups ", n_groups, ", n_threads ", n_threads);
  const auto visits = VisitCounts(n_groups, n_threads);
  REQUIRE(std::ssize(visits) == n_groups);
  CHECK(std::all_of(visits.begin(), visits.end(),
                    [](int count) { return count == 1; }));
}

// The threads that ran the groups, each group writing its own slot like an
// output. `park` runs first on every chunk.
template <typename Park>
std::set<std::thread::id> Runners(index_t n_groups, int n_threads, Park park) {
  std::vector<std::thread::id> ids(n_groups);
  const auto indptr = FlatIndptr(n_groups);
  parallel::ForEach(indptr, n_threads,
                    [&ids, &park](index_t start_group, index_t end_group) {
                      park();
                      for (index_t i = start_group; i < end_group; ++i) {
                        ids[i] = std::this_thread::get_id();
                      }
                    });
  return {ids.begin(), ids.end()};
}

} // namespace

TEST_CASE("every group is visited exactly once") {
  for (index_t n_groups : {0, 1, 7, 1000}) {
    for (int n_threads : {1, 2, 3, 8}) {
      CheckAllVisitedOnce(n_groups, n_threads);
    }
  }
}

TEST_CASE("no thread runs without a group of its own") {
  const auto here = std::this_thread::get_id();
  const auto nothing = [] {};
  // one group runs on the calling thread however many threads were asked for
  CHECK(Runners(1, 8, nothing) == std::set<std::thread::id>{here});
  CHECK(Runners(0, 8, nothing).empty());
  CHECK(Runners(3, 8, nothing).size() <= 3);
  // an empty indptr is malformed, but it means no groups rather than -1 of them
  parallel::ForEach(std::span<const index_t>{}, 8,
                    [](index_t start_group, index_t end_group) {
                      CHECK(start_group == end_group);
                    });
}

TEST_CASE("the calling thread is one of the workers") {
  constexpr index_t kGroups = 64;
  constexpr int kThreads = 4;
  std::atomic<int> arrived{0};
  // Every chunk holds its thread until kThreads of them are inside f, which
  // takes every thread including the calling one; a thread that doesn't turn
  // up is a timeout here rather than a hang.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  const auto ids = Runners(kGroups, kThreads, [&arrived, deadline] {
    arrived.fetch_add(1);
    while (arrived.load() < kThreads &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  });
  CHECK(ids.size() == kThreads);
  CHECK(ids.count(std::this_thread::get_id()) == 1);
}

TEST_CASE("equal chunks keep their index order") {
  const auto indptr = FlatIndptr(10);
  CHECK(parallel::ChunkStarts(indptr, 1) ==
        std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // 4 doesn't divide 10, so the last chunk is short and lighter than the rest;
  // it still has to come last rather than be dropped
  CHECK(parallel::ChunkStarts(indptr, 4) == std::vector<index_t>{0, 4, 8});
  CHECK(parallel::ChunkStarts(indptr, 100) == std::vector<index_t>{0});
  CHECK(parallel::ChunkStarts(FlatIndptr(0), 1).empty());
  CHECK(parallel::ChunkStarts(std::span<const index_t>{}, 1).empty());
}

TEST_CASE("the heaviest chunks are handed out first") {
  // group 7 holds almost everything and sits near the end, where going in
  // index order would start it last and leave nothing to overlap it with
  std::vector<index_t> indptr = FlatIndptr(10);
  for (index_t i = 8; i < std::ssize(indptr); ++i) {
    indptr[i] += 1000;
  }
  CHECK(parallel::ChunkStarts(indptr, 1) ==
        std::vector<index_t>{7, 0, 1, 2, 3, 4, 5, 6, 8, 9});
  // chunk 6 covers groups 6 and 7, so it is the heavy one now
  CHECK(parallel::ChunkStarts(indptr, 2) ==
        std::vector<index_t>{6, 0, 2, 4, 8});
}

TEST_CASE("a throwing chunk surfaces once the other workers have joined") {
  constexpr index_t kGroups = 64;
  constexpr int kThreads = 4;
  std::atomic<int> order{0};
  std::atomic<bool> threw{false};
  std::atomic<int> in_flight{0};
  const auto indptr = FlatIndptr(kGroups);
  // Whichever chunk runs first throws, and the rest park until it has, so they
  // are inside f while the exception is stashed. Returning without joining
  // them would leave workers reading the scheduler's locals after it has gone,
  // a use after return for ASan and a race for TSan, and in_flight above zero
  // here. Keying on the running order rather than on a group index keeps the
  // parked workers from waiting on a chunk that the scheduler hasn't handed
  // out yet.
  auto run_and_throw = [&order, &threw, &in_flight, &indptr] {
    parallel::ForEach(indptr, kThreads,
                      [&order, &threw, &in_flight](index_t, index_t) {
                        if (order.fetch_add(1) == 0) {
                          threw.store(true);
                          throw std::runtime_error("boom");
                        }
                        in_flight.fetch_add(1);
                        while (!threw.load()) {
                          std::this_thread::yield();
                        }
                        in_flight.fetch_sub(1);
                      });
  };
  CHECK_THROWS_AS(run_and_throw(), std::runtime_error);
  CHECK(in_flight.load() == 0);

  CheckAllVisitedOnce(kGroups, kThreads);
}

TEST_CASE("a spawn that runs out of memory only costs a thread") {
  constexpr index_t kGroups = 64;
  constexpr int kThreads = 4;
  const auto indptr = FlatIndptr(kGroups);
  bool finished_after_failure = false;
  // Fails each allocation the calling thread makes inside ForEach in turn,
  // spawns included, until there are none left to fail. std::thread allocates
  // its state with operator new, so it can throw std::bad_alloc rather than
  // std::system_error, and doing so with a worker already running must not
  // unwind past it.
  for (int n_allocs = 0;; ++n_allocs) {
    REQUIRE(n_allocs < 64);
    INFO("failing allocation ", n_allocs);
    std::vector<std::atomic<int>> visits(kGroups);
    bool threw = false;
    alloc_failed = false;
    allocs_before_failure = n_allocs;
    try {
      parallel::ForEach(indptr, kThreads,
                        [&visits](index_t start_group, index_t end_group) {
                          for (index_t i = start_group; i < end_group; ++i) {
                            visits[i].fetch_add(1);
                          }
                        });
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    allocs_before_failure = -1;
    if (!alloc_failed) {
      break;
    }
    if (!threw) {
      finished_after_failure = true;
      CHECK(std::all_of(
          visits.begin(), visits.end(),
          [](const std::atomic<int> &count) { return count.load() == 1; }));
    }
  }
  CHECK(finished_after_failure);
}

TEST_CASE("rolling quantile over threads equals the serial result") {
  constexpr index_t kGroups = 200;
  constexpr index_t kSize = 120;
  const auto data = Random<double>(kGroups * kSize);
  const auto w = Window::Checked(10, 1, false);
  std::vector<index_t> indptr(kGroups + 1);
  for (index_t i = 0; i <= kGroups; ++i) {
    indptr[i] = i * kSize;
  }
  auto run = [&data, &w, &indptr](std::vector<double> &out, int n_threads) {
    parallel::ForEach(
        indptr, n_threads,
        [&data, &w, &out](index_t start_group, index_t end_group) {
          for (index_t i = start_group; i < end_group; ++i) {
            rolling::QuantileTransform<double>(
                In(data).subspan(i * kSize, kSize),
                Out(out).subspan(i * kSize, kSize), w, 0.5);
          }
        });
  };
  std::vector<double> serial(data.size());
  std::vector<double> threaded(data.size());
  run(serial, 1);
  run(threaded, 4);
  // Every group owns its own slice of the output, so neither the split across
  // threads nor the order the chunks go out in may change a bit of it
  for (index_t i = 0; i < std::ssize(serial); ++i) {
    INFO("index ", i);
    REQUIRE(threaded[i] == serial[i]);
  }
}
