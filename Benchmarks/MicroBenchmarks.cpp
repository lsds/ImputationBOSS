#include "ITTNotifySupport.hpp"
#include "zipf_distribution.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
using namespace std;

static auto const vtune = VTuneAPIInterface{"Microbenchmarks"};

struct Partition {
  std::vector<int> col0; // use the first column to store the indexes
  std::vector<int> col1;
  std::vector<int> col2;
  std::vector<int> col3;
  explicit Partition(size_t size) : col0(size), col1(size), col2(size), col3(size){};
};
using PartitionPtr = std::unique_ptr<Partition>;

struct pqNode {
  Partition *ptr;
  int index;
};

struct pqComp {
  bool operator()(const pqNode &p1, const pqNode &p2) {
    return p1.ptr->col0[p1.index] < p2.ptr->col0[p2.index];
  }
};

struct PartitionRLE {
  std::vector<std::pair<int, int>> col0; // use the first column to store the indexes
  std::vector<int> col1;
  std::vector<int> col2;
  std::vector<int> col3;
  explicit PartitionRLE(Partition &p) : col1(std::move(p.col1)), col2(col1.size()), col3(col1.size()){};
};
using PartitionRLEPtr = std::unique_ptr<PartitionRLE>;

struct pqNodeRLE {
  PartitionRLE *ptr;
  int index;
  int rleIndex;
};

struct pqCompRLE {
  bool operator()(const pqNodeRLE &p1, const pqNodeRLE &p2) {
    return p1.ptr->col0[p1.rleIndex] < p2.ptr->col0[p2.rleIndex];
  }
};

struct indexNode {
  int partition;
  int index;
};

struct indexNodeRLE {
  int partition;
  int index;
  int cnt;
};

/*void PrintOnce(long size, std::vector<indexNode> &globalIndex, std::vector<indexNodeRLE> &globalIndexRLE) {
  static long callSetup = 0;
  if (callSetup != size) {
    std::cout << "[DBG] size before " << globalIndex.size() * sizeof(indexNode) <<
        " size after " << globalIndexRLE.size() * sizeof(indexNodeRLE) <<
        " ratio " << (double)(globalIndex.size() * sizeof(indexNode)) / (double)(globalIndexRLE.size() * sizeof(indexNodeRLE))  << std::endl;

    callSetup = size;
  }
}*/

std::vector<double> alphas = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 16}; //NOLINT

void generateIntegers(std::vector<PartitionPtr>& pars, size_t size, long ratio, long clustering, size_t alphaPos = 0, //NOLINT
                      std::vector<indexNode>* globalIndex = nullptr) {
  size_t totalSize = 0;
  int pIndex = 0;
  std::vector<size_t> idxs(pars.size(), 0);
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 1000); // NOLINT
    for(auto& par : pars) {
      size_t parSize = (totalSize == 0) ? size : size / ratio;
      totalSize += parSize;
      idxs[pIndex++] = parSize;

      par = std::make_unique<Partition>(parSize);
      for(auto i = 0UL; i < parSize; i++) {
        par->col1[i] = distr(gen);
      }
    }
  }

  {
    std::vector<size_t> sIdx(pars.size(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 1000); // NOLINT
    rand_val(distr(gen));
    //std::uniform_int_distribution<> distr(0, (int)pars.size() - 1);
    //zipf_distribution distr((int)pars.size(), 1.0); // NOLINT
    std::vector<int> cntrs (pars.size(), 0);
    if (globalIndex != nullptr) {
      globalIndex->resize(totalSize);
      for(auto i = 0UL; i < totalSize;) {
        int nextPar = zipf(alphas[alphaPos], (int)pars.size()) - 1; //distr(gen) - 1;
        cntrs[nextPar]++;
        (*globalIndex)[i++] = {nextPar, static_cast<int>(sIdx[nextPar]++)};
        if (sIdx[nextPar] >= idxs[nextPar]) {
          pars[nextPar]->col1.push_back(distr(gen));
          idxs[nextPar]++;
        }
        /*long elems = 0;
        while(sIdx[nextPar] < idxs[nextPar] && elems < clustering) {
          (*globalIndex)[i++] = {nextPar, static_cast<int>(sIdx[nextPar]++)};
          elems++;
        }*/
        //std::cout << "[DBG]: added " + std::to_string(elems) + " to par_" + std::to_string(nextPar) << std::endl;
      }
    } else {
      for(auto i = 0UL; i < totalSize;) {
        int nextPar = zipf(alphas[alphaPos], (int)pars.size()) - 1; //distr(gen) - 1;
        cntrs[nextPar]++;
        if (sIdx[nextPar] >= idxs[nextPar]) {
          pars[nextPar]->col1.push_back(distr(gen));
          idxs[nextPar]++;
        }
        if (pars[nextPar]->col0.size() <= sIdx[nextPar]) {
          pars[nextPar]->col0.push_back((int)i);
          sIdx[nextPar]++;
        } else {
          pars[nextPar]->col0[sIdx[nextPar]] = (int)i;
          sIdx[nextPar]++;
        }
        i++;

        /*long elems = 0;
        while(sIdx[nextPar] < idxs[nextPar] && elems < clustering) {
          pars[nextPar]->col0[sIdx[nextPar]++] = (int)i;
          i++;
          elems++;
        }*/
        //std::cout << "[DBG]: added " + std::to_string(elems) + " to par_" + std::to_string(nextPar) << std::endl;
      }
    }

    /*std::cout << "[DBG]: ";
    for (const auto &c: cntrs) {
      std::cout << c << " ";
    }
    std::cout << std::endl;*/
  }

  /*std::cout << "[DBG]: {size " + std::to_string(size) + " ratio " + std::to_string(ratio) +
  " clustering " + std::to_string(clustering) + "} ";
  for (auto i = 0UL; i < pars.size(); i++) {
    std::cout << " par_" + std::to_string(i) + ": " + std::to_string(idxs[i]);
  }
  std::cout << std::endl;*/
}

static void PartitionIndexes(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    std::vector<PartitionPtr> partitions(numOfPartitions);
    generateIntegers(partitions, size, ratio, clustering, alphaPos, nullptr);
    size_t sum = 0;
    size_t index = 0;
    std::priority_queue<pqNode, std::vector<pqNode>, pqComp> pq;
    for(auto& par : partitions) {
      pq.push({par.get(), 0});
    }

    // vtune.startSampling("PartitionIndexes - Microbenchmark");
    state.ResumeTiming();
    while(!pq.empty()) {
      auto nextPart = pq.top();
      pq.pop();
      sum += nextPart.ptr->col1[nextPart.index++];
      index++;
      if(nextPart.index < (int)nextPart.ptr->col1.size()) {
        pq.push(nextPart);
      }
    }
    benchmark::DoNotOptimize(sum);
    // vtune.stopSampling();
  }
}

static void PartitionIndexesUnrolled(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    std::vector<PartitionPtr> partitions(numOfPartitions);
    generateIntegers(partitions, size, ratio, clustering, alphaPos, nullptr);
    size_t sum = 0;
    size_t index = 0;
    std::priority_queue<pqNode, std::vector<pqNode>, pqComp> pq;
    for(auto& par : partitions) {
      pq.push({par.get(), 0});
    }

    // vtune.startSampling("PartitionIndexes - Microbenchmark");
    state.ResumeTiming();
    while(!pq.empty()) {
      auto nextPart = pq.top();
      pq.pop();
      sum += nextPart.ptr->col1[nextPart.index++];
      index++;
      while(nextPart.index < (int)nextPart.ptr->col1.size() && nextPart.ptr->col0[nextPart.index] == (int)index) {
        sum += nextPart.ptr->col1[nextPart.index++];
        index++;
      }
      if(nextPart.index < (int)nextPart.ptr->col1.size()) {
        pq.push(nextPart);
      }
    }
    benchmark::DoNotOptimize(sum);
    // vtune.stopSampling();
  }
}

static void PartitionIndexesUnrolledAndCompressed(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    std::vector<PartitionPtr> partitions(numOfPartitions);
    generateIntegers(partitions, size, ratio, clustering, alphaPos, nullptr);

    size_t sum = 0;
    size_t index = 0;
    std::priority_queue<pqNodeRLE, std::vector<pqNodeRLE>, pqCompRLE> pq;
    std::vector<PartitionRLEPtr> partitionsRLE;
    for(auto& par : partitions) {
      partitionsRLE.push_back(std::make_unique<PartitionRLE>(*par));

      int cnt = 0;
      int startIndex = par->col0[0];
      int prevIndex = par->col0[0]-1;
      for (auto &n: par->col0) {
        if (n == prevIndex+1) {
          cnt++;
        } else {
          partitionsRLE.back()->col0.emplace_back(startIndex, cnt);
          startIndex = n;
          cnt = 1;
        }
        prevIndex = n;
      }
      if (cnt != 0) {
        partitionsRLE.back()->col0.emplace_back(startIndex, cnt);
      }
      pq.push({partitionsRLE.back().get(), 0, 0});
    }
    //std::cout << "Finished compression!" << std::endl;

    // vtune.startSampling("PartitionIndexes - Microbenchmark");
    state.ResumeTiming();
    while(!pq.empty()) {
      auto nextPart = pq.top();
      pq.pop();
      auto const cnt = nextPart.ptr->col0[nextPart.rleIndex].second;
      auto const limit = index + cnt;
      while (index < limit) {
        sum += nextPart.ptr->col1[nextPart.index++];
        index++;
      }
      nextPart.rleIndex++;
      if(nextPart.rleIndex < (int)nextPart.ptr->col0.size()) {
        pq.push(nextPart);
      }
    }
    benchmark::DoNotOptimize(sum);
    // vtune.stopSampling();
  }
}

static void TwoPartitionIndexesUnrolled(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    if (numOfPartitions == 2) {
      std::vector<PartitionPtr> partitions(numOfPartitions);
      generateIntegers(partitions, size, ratio, clustering, alphaPos, nullptr);
      size_t sum = 0;
      size_t index = 0;
      size_t totalSize = partitions[0]->col0.size() + partitions[1]->col0.size();
      pqNode p1 = {partitions[0].get(), 0};
      pqNode p2 = {partitions[1].get(), 0};

      // vtune.startSampling("PartitionIndexes - Microbenchmark");
      state.ResumeTiming();
      while(index < totalSize) {
        pqNode nextPart{nullptr, 0};
        if(p1.index >= (int)p1.ptr->col1.size()) {
          nextPart = p2;
        } else if(p2.index >= (int)p2.ptr->col1.size()) {
          nextPart = p1;
        } else {
          nextPart = (p1.ptr->col0[p1.index] < p2.ptr->col0[p2.index]) ? p1 : p2;
        }

        sum += nextPart.ptr->col1[nextPart.index++];
        index++;
        while(nextPart.index < (int)nextPart.ptr->col1.size() &&
              nextPart.ptr->col0[nextPart.index] == (int)index) {
          sum += nextPart.ptr->col1[nextPart.index++];
          index++;
        }
      }
      benchmark::DoNotOptimize(sum);
      // vtune.stopSampling();
    } else {
      state.ResumeTiming();
    }
  }
}

static void GlobalIndex(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    std::vector<PartitionPtr> partitions(numOfPartitions);
    std::vector<indexNode> globalIndex;
    generateIntegers(partitions, size, ratio, clustering, alphaPos, &globalIndex);
    size_t sum = 0;
    size_t index = 0;

    //vtune.startSampling("GlobalIndex - Microbenchmark");
    state.ResumeTiming();
    while (true) {
      sum += partitions[globalIndex[index].partition]->col1[globalIndex[index].index];
      index++;
      if(index >= globalIndex.size()) {
        break;
      }
    }
    benchmark::DoNotOptimize(sum);
    //vtune.stopSampling();
  }
}

static void CompressedGlobalIndex(benchmark::State& state) {
  for (auto _ : state) { // NOLINT
    state.PauseTiming();
    auto size = state.range(0);
    auto numOfPartitions = state.range(1);
    auto ratio = state.range(2);
    auto clustering = state.range(3);
    auto alphaPos = state.range(4);

    std::vector<PartitionPtr> partitions(numOfPartitions);
    std::vector<indexNode> globalIndex;
    generateIntegers(partitions, size, ratio, clustering, alphaPos, &globalIndex);
    size_t sum = 0;
    size_t index = 0;
    size_t indexRLE = 0;

    // Compress the index with RLE
    std::vector<indexNodeRLE> globalIndexRLE;
    int cnt = 0;
    int prev = globalIndex[0].partition;
    int prevIndex = globalIndex[0].index;
    for (auto &n: globalIndex) {
      if (prev == n.partition) {
        cnt++;
      } else {
        globalIndexRLE.push_back({prev, prevIndex, cnt});
        prev = n.partition;
        prevIndex = n.index;
        cnt = 1;
      }
    }
    if (cnt != 0) {
      globalIndexRLE.push_back({prev, prevIndex, cnt});
    }
    //PrintOnce(size, globalIndex, globalIndexRLE);

    //vtune.startSampling("CompressedGlobalIndex - Microbenchmark");
    state.ResumeTiming();
    while (true) {
      const auto part = globalIndexRLE[indexRLE].partition;
      const auto start = globalIndexRLE[indexRLE].index;
      const auto limit = globalIndexRLE[indexRLE].index+globalIndexRLE[indexRLE].cnt;
      for (auto i = start; i < limit; i++) {
        sum += partitions[part]->col1[i];
        index++;
      }
      indexRLE++;
      if(index >= globalIndex.size() || indexRLE >= globalIndexRLE.size()) {
        break;
      }
    }
    benchmark::DoNotOptimize(sum);
    //vtune.stopSampling();
  }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  static int const sizePerPartition = 1U << 20U; // 1M
  static int const ratio = 1;
  static int const clustering = 1;
  for(auto numPartitions : std::vector<int>{4, 16, 64}) {
    for(auto alphaPos = 0; alphaPos < alphas.size(); alphaPos++) {
      b->Args({sizePerPartition*numPartitions, numPartitions, ratio, clustering, alphaPos});
    }
  }
}

BENCHMARK(PartitionIndexes)->Apply(CustomArguments); // NOLINT
BENCHMARK(PartitionIndexesUnrolled)->Apply(CustomArguments); // NOLINT
BENCHMARK(PartitionIndexesUnrolledAndCompressed)->Apply(CustomArguments); // NOLINT
BENCHMARK(TwoPartitionIndexesUnrolled)->Apply(CustomArguments); // NOLINT
BENCHMARK(GlobalIndex)->Apply(CustomArguments); // NOLINT
BENCHMARK(CompressedGlobalIndex)->Apply(CustomArguments); // NOLINT

BENCHMARK_MAIN();
