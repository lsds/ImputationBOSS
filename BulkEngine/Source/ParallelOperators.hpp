#pragma once

#include "Bulk.hpp"
#include "BulkExpression.hpp"
#include "BulkUtilities.hpp"

#include <condition_variable>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace boss::engines::bulk {

template <typename... ArgumentTypes>
class ParallelDispatch : public boss::engines::bulk::Operator<ParallelDispatch, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr>>;
  using Operator<ParallelDispatch, ArgumentTypes...>::Operator;
  void operator()(Table::PartitionPtr const& partition) {
    // init: retrieve thread info
    if(threadIndex < 0) {
      auto& symbolRegistry = DefaultSymbolRegistry::instance();
      auto const* threadIndexSymbol = symbolRegistry.findSymbol("threadIndex"_);
      auto const* threadMaxIndexSymbol = symbolRegistry.findSymbol("threadMaxIndex"_);
      auto* retrievedThreadIndex = std::get_if<int64_t>(threadIndexSymbol);
      if(retrievedThreadIndex == nullptr) {
        throw std::runtime_error("ParallelDispatch: could not retrieve threadIndex");
      }
      auto* retrievedThreadMaxIndex = std::get_if<int64_t>(threadMaxIndexSymbol);
      if(retrievedThreadMaxIndex == nullptr) {
        throw std::runtime_error("ParallelDispatch: could not retrieve threadMaxIndex");
      }
      threadIndex = *retrievedThreadIndex;
      maxIndex = *retrievedThreadMaxIndex;
    }
    // push only partitions dispatched for this thread
    if(currentIndex == threadIndex) {
      this->pushUp(partition);
    }
    currentIndex = (currentIndex + 1) % maxIndex;
  }
  void close() override {
    threadIndex = -1;
    maxIndex = -1;
    currentIndex = 0;
  }

private:
  int64_t threadIndex = -1;
  int64_t maxIndex = -1;
  int64_t currentIndex = 0;
};
namespace {
boss::engines::bulk::Engine::Register<ParallelDispatch> const p01("ParallelDispatch"); // NOLINT
}

template <typename... ArgumentTypes>
class ParallelCombine : public boss::engines::bulk::Operator<ParallelCombine, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Expression>>;
  using Operator<ParallelCombine, ArgumentTypes...>::Operator;
  void operator()(Expression const& subQuery) {
    std::vector<std::thread> threads;
    std::atomic<int> numRunningThreads = std::thread::hardware_concurrency();
    expressionsToCollect.reserve(numRunningThreads);
    // run subquery in parallel
    threads.reserve(numRunningThreads);
    for(int i = 0; i < numRunningThreads; ++i) {
      threads.emplace_back(
          [this, &subQuery, &numRunningThreads](int64_t threadIndex, int64_t maxThreadIndex) {
            // register thread info
            auto& symbolRegistry = DefaultSymbolRegistry::instance();
            symbolRegistry.registerSymbol("threadIndex"_, threadIndex);
            symbolRegistry.registerSymbol("threadMaxIndex"_, maxThreadIndex);
            // evaluate sub-query
            try {
              Engine::evaluateInternal(subQuery, [this](auto&& e, bool evaluated) {
                std::unique_lock lk(m);
                if(evaluated) {
                  if constexpr(std::is_lvalue_reference_v<decltype(e)>) {
                    expressionsToCollect.emplace_back(e.clone());
                  } else {
                    expressionsToCollect.emplace_back(std::move(e));
                  }
                }
                lk.unlock();
              });
            } catch(...) {
              std::lock_guard lk(m);
              threadExceptions.emplace_back(std::current_exception());
            }
            --numRunningThreads;
            cv.notify_one();
          },
          i, (int64_t)numRunningThreads);
    }
    // collect and push outputs
    std::unique_lock lk(m);
    do {
      cv.wait(lk, [this, &numRunningThreads]() {
        return !expressionsToCollect.empty() || !threadExceptions.empty() || numRunningThreads == 0;
      });
      if(!threadExceptions.empty()) {
        break;
      }
      for(auto&& e : expressionsToCollect) {
        this->pushUp(std::move(e));
      }
      expressionsToCollect.clear();
    } while(numRunningThreads > 0);
    lk.unlock();
    // wait for all other threads to finish
    for(auto& thread : threads) {
      thread.join();
    }
    // catch and rethrow thread exceptions
    for(auto&& e : threadExceptions) {
      std::rethrow_exception(e);
    }
  }

private:
  std::mutex m;
  std::condition_variable cv;
  std::vector<Expression> expressionsToCollect;
  std::vector<std::exception_ptr> threadExceptions;
};
namespace {
boss::engines::bulk::Engine::Register<ParallelCombine> const p02("ParallelCombine"); // NOLINT
}

} // namespace boss::engines::bulk
