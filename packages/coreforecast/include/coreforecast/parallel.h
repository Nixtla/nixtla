#pragma once

// The scheduler the grouped operations run through. Nothing here may depend on
// Python or pybind11: the GIL is released on the binding side, so this file is
// compiled into the C++ tests on its own and the sanitizers see the parallel
// path without an interpreter in the way.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include "common.h"

namespace parallel {

// Chunks per thread. Enough of them that a thread which drew cheap chunks takes
// more, few enough that the atomic fetch and the ordering below cost nothing
// next to the groups they hand out.
inline constexpr index_t kChunksPerThread = 16;

// indptr.size() - 1, except that an empty indptr has no groups rather than -1
// of them.
inline index_t NumGroups(std::span<const index_t> indptr) {
  return indptr.empty() ? 0 : std::ssize(indptr) - 1;
}

inline index_t ChunkSize(index_t n_groups, int n_threads) {
  return std::max<index_t>(1, n_groups / (kChunksPerThread * n_threads));
}

// The start offset of every chunk of [0, n_groups), heaviest chunk first,
// where indptr[i + 1] - indptr[i] is how many elements group i holds.
//
// The order matters because a group is never split: a thread that starts the
// longest one last has nothing left to overlap it with, and the call can't
// finish before that group does. Chunks of equal weight keep their index
// order, so evenly sized groups are walked exactly as they are laid out.
inline std::vector<index_t> ChunkStarts(std::span<const index_t> indptr,
                                        index_t chunk) {
  assert(chunk > 0);
  const index_t n_groups = NumGroups(indptr);
  std::vector<index_t> starts;
  starts.reserve(static_cast<size_t>((n_groups + chunk - 1) / chunk));
  for (index_t start = 0; start < n_groups; start += chunk) {
    starts.push_back(start);
  }
  const auto elements = [indptr, chunk, n_groups](index_t start) {
    return indptr[std::min(start + chunk, n_groups)] - indptr[start];
  };
  std::sort(starts.begin(), starts.end(), [&elements](index_t a, index_t b) {
    const index_t wa = elements(a);
    const index_t wb = elements(b);
    return wa != wb ? wa > wb : a < b;
  });
  return starts;
}

// Calls f(start_group, end_group) over the chunks of [0, n_groups) until they
// run out. Chunks go out off a shared counter in the order ChunkStarts puts
// them in, so a thread that drew short groups takes more of them instead of
// idling while another works through the long ones. The calling thread is
// worker 0 and at most one thread per group runs, so no thread ever starts
// without a chunk to run and a spawn that fails only costs a thread.
//
// Every group writes only its own slice of the output, so neither the thread
// count nor the order the chunks go out in changes the result. Exceptions
// can't cross a thread boundary, so a worker that throws stashes its own and
// stops the counter, the rest hand back after the chunk they are on, and it
// is rethrown once everything has been joined.
template <typename Func>
inline void ForEach(std::span<const index_t> indptr, int n_threads, Func f) {
  const index_t n_groups = NumGroups(indptr);
  n_threads = static_cast<int>(std::min<index_t>(n_threads, n_groups));
  if (n_threads < 2) {
    f(0, n_groups);
    return;
  }
  const index_t chunk = ChunkSize(n_groups, n_threads);
  const std::vector<index_t> starts = ChunkStarts(indptr, chunk);
  std::atomic<size_t> next_chunk{0};
  std::vector<std::exception_ptr> errors(n_threads);
  const auto worker = [&f, &errors, &next_chunk, &starts, chunk,
                       n_groups](int t) {
    try {
      for (size_t i = next_chunk.fetch_add(1); i < starts.size();
           i = next_chunk.fetch_add(1)) {
        f(starts[i], std::min(starts[i] + chunk, n_groups));
      }
    } catch (...) {
      errors[t] = std::current_exception();
      next_chunk.store(starts.size());
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(n_threads - 1);
  for (int t = 1; t < n_threads; ++t) {
    try {
      threads.emplace_back(worker, t);
    } catch (...) {
      break;
    }
  }
  worker(0);
  for (auto &thread : threads) {
    thread.join();
  }
  for (auto &error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

} // namespace parallel
