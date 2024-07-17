#pragma once

#include "ArrowExtensions/ComplexExpressionBuilder.hpp"
#include "BulkExpression.hpp"
// #include "CompareExpression.hpp"
#include "Table.hpp"

#include "ITT/ITTNotifySupport.hpp"

#include <Expression.hpp>

#include <functional>
#include <iterator>
#include <map>
#include <sstream>

// for debug info
#include <iostream>

#ifndef NDEBUG
#define DEBUG_OUTPUT_OPS
#endif

// hash implementation
#if defined(USE_ABSL_HASHING)
#include <absl/hash/hash.h>
template <typename T> class Hash : public absl::Hash<T> {};
#elif defined(USE_ROBIN_HASHING)
#include <robin_hood.h>
template <typename T> class Hash : public robin_hood::hash<T> {};
#elif defined(USE_FOLLY_HASHING)
#include <folly/hash/Hash.h>
template <typename T> class Hash : public folly::hasher<T> {};
#else // STD
template <typename T> class Hash : public std::hash<T> {};
#endif

// equal_to implementation
template <typename T> class Equal : public std::equal_to<T> {};

// map implementation
#if defined(USE_TSL_HASHMAP)
#include <tsl/sparse_map.h>
template <typename K, typename V> using Map = tsl::sparse_map<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_ROBIN_HASHMAP)
#include <robin_hood.h>
template <typename K, typename V>
using Map = robin_hood::unordered_flat_map<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_ABSL_HASHMAP)
#include <absl/container/flat_hash_map.h>
template <typename K, typename V> using Map = absl::flat_hash_map<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_FOLLY_HASHMAP)
#include <folly/container/F14Map.h>
template <typename K, typename V> using Map = folly::F14ValueMap<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_ROBIN_NODE_HASHMAP)
#include <robin_hood.h>
template <typename K, typename V>
using Map = robin_hood::unordered_node_map<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_ABSL_NODE_HASHMAP)
#include <absl/container/node_hash_map.h>
template <typename K, typename V> using Map = absl::node_hash_map<K, V, Hash<K>, Equal<K>>;
#elif defined(USE_FOLLY_NODE_HASHMAP)
#include <folly/container/F14Map.h>
template <typename K, typename V> using Map = folly::F14NodeMap<K, V, Hash<K>, Equal<K>>;
#else // STD
#include <unordered_map>
template <typename K, typename V> using Map = std::unordered_map<K, V, Hash<K>, Equal<K>>;
#endif

// set implementation
#if defined(USE_TSL_SET)
#include <tsl/sparse_set.h>
template <typename K> using Set = tsl::sparse_set<K>;
#elif defined(USE_ROBIN_SET)
#include <robin_hood.h>
template <typename K> using Set = robin_hood::unordered_set<K>;
#elif defined(USE_ABSL_SET)
#include <absl/container/flat_hash_set.h
template <typename K> using Set = absl::flat_hash_set<K>;
#else // STD
#include <unordered_set>
template <typename K> using Set = std::unordered_set<K>;
#endif

static auto& ITT() {
  thread_local auto const vtune = VTuneAPIInterface{"RelationalOps"};
  return vtune;
}
static auto& ITTSelectTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Select"};
  return subtask;
}
static auto& ITTProjectTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Project"};
  return subtask;
}
static auto& ITTGroupTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Group"};
  return subtask;
}
static auto& ITTSortTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Sort"};
  return subtask;
}
static auto& ITTTopTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Top"};
  return subtask;
}
static auto& ITTJoinTask() {
  thread_local auto subtask = VTuneSubtask{ITT(), "Join"};
  return subtask;
}

template <size_t Size>
class MultipleNumericalKeys : public std::pair<int64_t, std::array<int64_t, Size>> {
public:
  using ArrayType = std::array<int64_t, Size>;
  template <typename... Args>
  explicit MultipleNumericalKeys(Args... args)
      : std::pair<int64_t, ArrayType>(sizeof...(Args), ArrayType{args...}) {}
  size_t size() const { return this->first; }
  void resize(size_t size) { this->first = size; }
  template <typename T, std::enable_if_t<sizeof(T) == sizeof(int64_t), bool> = true>
  T get(size_t index) const {
    return reinterpret_cast<T const&>(this->second.at(index));
  }
  template <typename T, std::enable_if_t<sizeof(T) == sizeof(int32_t), bool> = true>
  T get(size_t index) const {
    auto value32 = static_cast<int32_t>(this->second.at(index));
    return reinterpret_cast<T&>(value32);
  }
  template <typename T, std::enable_if_t<sizeof(T) == sizeof(int64_t), bool> = true>
  void set(size_t index, T value) {
    this->second.at(index) = reinterpret_cast<int64_t&>(value);
  }
  template <typename T, std::enable_if_t<sizeof(T) == sizeof(int32_t), bool> = true>
  void set(size_t index, T value) {
    this->second.at(index) = static_cast<int64_t>(reinterpret_cast<int32_t&>(value));
  }
  void set(size_t index, void const* dataPtr, int64_t size) {
    auto mask = (uint64_t)(int64_t(-1))
                << std::min((uint64_t)sizeof(int64_t) * CHAR_BIT - 1U, (uint64_t)size * CHAR_BIT)
                << (size >= sizeof(int64_t) * CHAR_BIT ? 1U : 0U);
    this->second.at(index) = (uint64_t)~mask & (uint64_t) * static_cast<int64_t const*>(dataPtr);
  }

  template <typename H> friend H AbslHashValue(H h, MultipleNumericalKeys const& ints);
};

constexpr static int MAX_MULTIPLE_KEY_SIZE = 10;
using MultipleKeys = MultipleNumericalKeys<MAX_MULTIPLE_KEY_SIZE>;

#if defined(USE_ABSL_HASHING)
template <typename H> H AbslHashValue(H h, MultipleKeys const& ints) {
  return H::combine_contiguous(std::move(h), &ints.second[0], ints.first);
}
#else
template <std::size_t Size> class Hash<MultipleNumericalKeys<Size>> {
public:
  size_t operator()(MultipleNumericalKeys<Size> const& ints) const {
    return hashing(ints, std::make_index_sequence<Size>{});
  }

private:
  template <std::size_t I0, std::size_t... Is>
  size_t hashing(MultipleNumericalKeys<Size> const& ints,
                 std::index_sequence<I0, Is...> /*unused*/) const {
#if defined(USE_FOLLY_HASHING)
    return folly::hash_combine_generic(
        Hash<MultipleNumericalKeys<Size>>{},
        reinterpret_cast<int64_t const&>(std::get<Is>(ints.second))...);
#else
    auto h = std::hash<int64_t>{}(reinterpret_cast<int64_t const&>(std::get<I0>(ints.second)));
    if(ints.first > 1) {
      (void)((hash_combine(h, std::hash<int64_t>{}(
                                  reinterpret_cast<int64_t const&>(std::get<Is>(ints.second)))),
              ints.first > Is + 1) &&
             ...);
    }
    return h;
#endif
  };

  static void hash_combine(size_t& seed, size_t value) {
    static constexpr auto boostHashMagicNumber1 = 0x9e3779b9U;
    static constexpr auto boostHashMagicNumber2 = 6U;
    static constexpr auto boostHashMagicNumber3 = 2U;
    seed ^= value + boostHashMagicNumber1 + (seed << boostHashMagicNumber2) +
            (seed >> boostHashMagicNumber3);
  }
};
#endif

template <size_t Size> class Equal<MultipleNumericalKeys<Size>> {
public:
  bool operator()(MultipleNumericalKeys<Size> const& lhs,
                  MultipleNumericalKeys<Size> const& rhs) const {
    return operator()(lhs, rhs, std::make_index_sequence<Size>{});
  }

private:
  template <std::size_t... Is>
  bool operator()(MultipleNumericalKeys<Size> const& lhs, MultipleNumericalKeys<Size> const& rhs,
                  std::index_sequence<Is...> /*unused*/) const {
    bool equal = true;
    (void)(((equal &= bool(std::get<Is>(lhs.second) == std::get<Is>(rhs.second))),
            lhs.first > Is + 1) &&
           ...);
    return equal;
  };
};

namespace boss::engines::bulk {

#ifdef DEBUG_OUTPUT_OPS
static int DEBUG_OUTPUT_RELATION_OPS_DEPTH = 0; // NOLINT
static auto& debugPartitions() {
  static std::vector<Table::PartitionPtr> partitions;
  return partitions;
}
static void outputDebugPartition(Table::PartitionPtr const& partition) {
  if(!partition) {
    std::cerr << "[]";
    return;
  }
  auto foundIt = std::find(debugPartitions().begin(), debugPartitions().end(), partition);
  if(foundIt == debugPartitions().end()) {
    debugPartitions().emplace_back(partition);
    std::cerr << "[" << debugPartitions().size() << "(rows" << partition->length() << "cols"
              << partition->num_fields() << ")]";
    return;
  }
  std::cerr << "[" << std::distance(debugPartitions().begin(), foundIt) + 1 << "(rows"
            << (*foundIt)->length() << "cols" << (*foundIt)->num_fields() << ")]";
}
static void outputDebugPartitions(Table::PartitionVectorPtr const& partitions) {
  if(!partitions) {
    std::cerr << "[]" << std::endl;
    return;
  }
  auto length = std::accumulate(
      partitions->begin(), partitions->end(), 0,
      [](auto count, auto const& partition) { return count + partition->length(); });
  if(length == 0) {
    std::cerr << "[]" << std::endl;
    return;
  }
  std::cerr << "["
            << "rows" << length << "cols" << (*partitions)[0]->getSchema().size() << "]";
}
static void clearDebugPartitions() { debugPartitions().clear(); }
#endif // DEBUG_OUTPUT_OPS

class RegisterColumnSymbols {
public:
  explicit RegisterColumnSymbols(Table::PartitionPtr const& partitionPtr)
      : partitionPtr(partitionPtr) {
    if(partitionPtr) {
      partitionPtr->registerSymbols();
    }
  }
  ~RegisterColumnSymbols() {
    try {
      if(partitionPtr) {
        partitionPtr->unregisterSymbols();
      }
    } catch(std::exception const& /*e*/) {
    }
  }
  RegisterColumnSymbols(RegisterColumnSymbols&) = delete;
  RegisterColumnSymbols& operator=(RegisterColumnSymbols&) = delete;
  RegisterColumnSymbols(RegisterColumnSymbols&&) = default;
  RegisterColumnSymbols& operator=(RegisterColumnSymbols&&) = delete;

private:
  Table::PartitionPtr const& partitionPtr;
};

template <typename... ArgumentTypes>
class Select : public boss::engines::bulk::Operator<Select, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr, ComplexExpression>>;
  using Operator<Select, ArgumentTypes...>::Operator;
  bool checkArguments(Table::PartitionPtr const& /*partition*/,
                      ComplexExpression const& predicate) {
    return predicate.getHead().getName() == "Where" || predicate.getHead().getName() == "Function";
  }
  void operator()(Table::PartitionPtr const& partition, ComplexExpression const& predicate) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Select ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
#ifdef DEBUG_OUTPUT_OPS
      ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      this->pushUp(partition);
#ifdef DEBUG_OUTPUT_OPS
      --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      return;
    }
    ITTSection section{ITTSelectTask()};
    // prepare predicate for evaluation
    auto predicateWithArguments = [&]() -> Expression {
      if(predicate.getHead().getName() == "Where") {
        return predicate.cloneArgument(0);
      }
      auto oldPredicateNumArguments = predicate.getArguments().size();
      ExpressionArguments predicateArguments;
      predicateArguments.reserve(oldPredicateNumArguments + 1);
      for(int i = 0; i < oldPredicateNumArguments; ++i) {
        predicateArguments.emplace_back(predicate.cloneArgument(i));
      }
      predicateArguments.emplace_back("List"_(partition));
      return ComplexExpression(predicate.getHead(), std::move(predicateArguments));
    }();
    // evaluate predicate
    auto toKeep = [&]() {
      auto temporarlyRegisterSymbols =
          RegisterColumnSymbols(partition); // register symbol for each column
      return this->evaluateInternal(std::move(predicateWithArguments));
    }();
    // apply the predicate to filter the rows
    select(partition, std::move(toKeep), section);
  }

  void close() override {
    ITTSection section{ITTSelectTask()};
    auto outputPtr = outputRelation.finaliseAndGetPartitions();
    auto& output = *outputPtr;
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    std::for_each(
        std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()),
        [this](auto&& partition) { this->pushUp(std::forward<decltype(partition)>(partition)); });
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
  }

private:
  Table outputRelation;

  void select(Table::PartitionPtr const& partition, Expression&& toKeep, ITTSection& section) {
    std::visit(
        boss::utilities::overload(
            [&, this](bool constant) {
              if(!constant) {
                // empty output
                ComplexExpressionArrayBuilder builder(partition, true);
                auto output = (Table::PartitionPtr)builder;
                section.pause();
                this->pushUp(std::move(output));
                section.resume();
                return;
              }
              // all rows in the output
              auto outputPartitionIndex = outputRelation.addPartition(partition);
              auto outputPartition = outputRelation.finaliseAndGetPartition(outputPartitionIndex);
              section.pause();
              this->pushUp(std::move(outputPartition));
              section.resume();
            },
            [&, this](ValueArrayPtr<bool> const& bools) {
              auto numRows = bools->true_count();
              if(numRows == 0) {
                return;
              }
              auto outputPartitionIndex =
                  outputRelation.findMatchingPartitionIndex(partition, numRows);
              if(outputPartitionIndex < outputRelation.numPartitions()) {
                // add to the existing partition
                auto builder = outputRelation.getPartitionBuilder(outputPartitionIndex);
                builder->appendRowsWithCondition(*partition, *bools);
                outputPartitionIndex =
                    outputRelation.setPartition(outputPartitionIndex, std::move(builder));
              } else {
                // create a new partition
                int64_t expectedFullSizePartition = partition->length();
                auto builder = std::make_shared<ComplexExpressionArrayBuilder>(
                    partition, true, expectedFullSizePartition);
                builder->appendRowsWithCondition(*partition, *bools);
                outputPartitionIndex = outputRelation.addPartition(std::move(builder));
              }
              // push if it is unlikely to fit the next batch's rows
              // (assuming all batches have roughly the same cardinality)
              if(!outputRelation.hasEnoughSpaceInPartition(outputPartitionIndex, numRows)) {
                auto outputPartition = outputRelation.finaliseAndGetPartition(outputPartitionIndex);
                section.pause();
                this->pushUp(std::move(outputPartition));
                section.resume();
                // remove the partition we pushed
                // we know that it is the latest partition (newly converted from builder)
                // so it is cheap to remove
                outputRelation.removePartition(outputRelation.numPartitions() - 1);
              }
            },
            [&, this](ValueArrayPtr<int64_t> const& pos) {
              auto numRows = pos->length();
              if(numRows == 0) {
                return;
              }
              auto outputPartitionIndex =
                  outputRelation.findMatchingPartitionIndex(partition, numRows);
              if(outputPartitionIndex < outputRelation.numPartitions()) {
                // add to the existing partition
                auto builder = outputRelation.getPartitionBuilder(outputPartitionIndex);
                builder->appendRowsInIndexedOrder(*partition, *pos, false /*preserve order*/);
                outputPartitionIndex =
                    outputRelation.setPartition(outputPartitionIndex, std::move(builder));
              } else {
                // create a new partition
                int64_t expectedFullSizePartition = partition->length();
                auto builder = std::make_shared<ComplexExpressionArrayBuilder>(
                    partition, true, expectedFullSizePartition);
                builder->appendRowsInIndexedOrder(*partition, *pos, false /*preserve order*/);
                outputPartitionIndex = outputRelation.addPartition(std::move(builder));
              }
              // push if it is unlikely to fit the next batch's rows
              // (assuming all batches have roughly the same cardinality)
              if(!outputRelation.hasEnoughSpaceInPartition(outputPartitionIndex, numRows)) {
                auto outputPartition = outputRelation.finaliseAndGetPartition(outputPartitionIndex);
                section.pause();
                this->pushUp(std::move(outputPartition));
                section.resume();
                // remove the partition we pushed
                // we know that it is the latest partition (newly converted from builder)
                // so it is cheap to remove
                outputRelation.removePartition(outputRelation.numPartitions() - 1);
              }
            },
            [&, this](std::shared_ptr<ComplexExpressionArray> const& /*complexArray*/) {
              throw std::logic_error("unsupported unevaluated key for selection");
            },
            [](ComplexExpression const& expr) {
              auto oss = std::ostringstream();
              oss << expr;
              throw std::logic_error(
                  "'" + oss.str() +
                  "' complex expression not supported as a predicate for selection");
            },
            [](auto const& /*other*/) {
              throw std::logic_error("type not supported as a predicate for selection");
            }),
        std::move(toKeep));
  }
};
namespace {
boss::engines::bulk::Engine::Register<Select> const r100("Select"); // NOLINT
}

template <typename... ArgumentTypes>
class Project : public boss::engines::bulk::Operator<Project, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr, ComplexExpression>>;
  using Operator<Project, ArgumentTypes...>::Operator;
  bool checkArguments(Table::PartitionPtr const& /*partition*/,
                      ComplexExpression const& projector) {
    return projector.getHead().getName() == "Function" || projector.getHead().getName() == "As";
  }
  void operator()(Table::PartitionPtr const& partition, ComplexExpression const& projector) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Project ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
#ifdef DEBUG_OUTPUT_OPS
      ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      this->pushUp(partition);
#ifdef DEBUG_OUTPUT_OPS
      --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      return;
    }
    ITTSection section{ITTProjectTask()};
    // evaluate projector function
    auto projectedColumns = [&]() -> ExpressionArguments {
      auto temporarlyRegisterSymbols =
          RegisterColumnSymbols(partition); // register symbol for each column
      if(projector.getHead().getName() == "Function") {
        auto oldProjectorNumArguments = projector.getArguments().size();
        ExpressionArguments projectorArgs;
        projectorArgs.reserve(oldProjectorNumArguments + 1);
        for(int i = 0; i < oldProjectorNumArguments; ++i) {
          projectorArgs.emplace_back(projector.cloneArgument(i));
        }
        projectorArgs.emplace_back("List"_(partition));
        auto projectorWithArgs = ComplexExpression(projector.getHead(), std::move(projectorArgs));
        return get<ComplexExpression>(this->evaluateInternal(std::move(projectorWithArgs)))
            .getArguments();
      }
      auto unevaluatedArgs = projector.getArguments();
      ExpressionArguments evaluatedArgs;
      evaluatedArgs.reserve(unevaluatedArgs.size());
      bool isArgEvaluated = true; // evaluate only every other arg (keep the "as" symbol as it is)
      std::transform(std::begin(unevaluatedArgs), std::end(unevaluatedArgs),
                     std::back_inserter(evaluatedArgs),
                     [this, &isArgEvaluated](auto const& e) -> Expression {
                       isArgEvaluated = !isArgEvaluated;
                       if(isArgEvaluated) {
                         return this->evaluateInternal(e.clone());
                       }
                       return e.clone();
                     });
      return std::move(evaluatedArgs);
    }();
    // separate the column names (symbols) from the column values
    std::vector<Symbol> columnNames;
    arrow::ArrayVector columnArrays;
    auto projectedColumnsSize = projectedColumns.size();
    columnNames.reserve(projectedColumnsSize / 2);
    columnArrays.reserve(projectedColumnsSize);
    for(auto&& arg : std::move(projectedColumns)) {
      visit(
          boss::utilities::overload(
              [&columnArrays](auto&& valueArrayPtr) {
                using KeyArrayType = std::decay_t<decltype(valueArrayPtr)>;
                if constexpr(std::is_convertible_v<KeyArrayType, std::shared_ptr<arrow::Array>>) {
                  columnArrays.emplace_back(std::forward<decltype(valueArrayPtr)>(valueArrayPtr));
                } else {
                  auto oss = std::ostringstream();
                  oss << valueArrayPtr; // (not an actual value array ptr...)
                  throw std::logic_error("'" + oss.str() + "' not supported as a projected column");
                }
              },
              [&columnNames, &columnArrays](Symbol&& symbol) {
                if(columnArrays.size() < columnNames.size()) {
                  auto oss = std::ostringstream();
                  oss << symbol;
                  throw std::logic_error("'" + oss.str() + "' not supported as a projected column");
                }
                columnNames.emplace_back(std::move(symbol));
              }),
          std::move(arg));
    }
    // build a relation with all the projected columns
    columnNames.resize(columnArrays.size(), Symbol(""));
    // check if the shape changed and if so, try to merge with other partitions
    uint64_t missingHash = ComplexExpressionArray::computeMissingHash(columnArrays);
    if(missingHash != partition->getMissingHash()) {
      auto numRows = columnArrays[0] ? columnArrays[0]->length() : 0;
      auto partitionIndex =
          outputRelation.findMatchingPartitionIndex("List"_, columnArrays, numRows);
      if(partitionIndex < outputRelation.numPartitions()) {
        // add to the existing partition
        auto builder = outputRelation.getPartitionBuilder(partitionIndex);
        builder->appendRows(columnArrays, partition->getGlobalIndexes());
        partitionIndex = outputRelation.setPartition(partitionIndex, std::move(builder));
      } else {
        // add as a new partition (convert back to builder)
        auto builder =
            std::make_shared<ComplexExpressionArrayBuilder>("List"_, columnNames, columnArrays);
        builder->setGlobalIndexes(partition->getGlobalIndexes());
        partitionIndex = outputRelation.addPartition(std::move(builder));
      }
      // push if it is unlikely to fit the next batch's rows
      // (assuming all batches have roughly the same cardinality)
      if(!outputRelation.hasEnoughSpaceInPartition(partitionIndex, numRows)) {
        auto mergedPartition = outputRelation.finaliseAndGetPartition(partitionIndex);
        section.pause();
#ifdef DEBUG_OUTPUT_OPS
        ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
        this->pushUp(std::move(mergedPartition));
#ifdef DEBUG_OUTPUT_OPS
        --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
        section.resume();
        // remove the partition we pushed
        // we know that it is the latest partition (newly converted from builder)
        // so it is cheap to remove
        outputRelation.removePartition(outputRelation.numPartitions() - 1);
      }
      return;
    }
    // otherwise just push the single partition
    // (finalise it first in case global indexes are needed)
    auto output = std::make_shared<ComplexExpressionArray>("List"_, columnNames, columnArrays);
    output->setGlobalIndexes(partition->getGlobalIndexes());
    auto partitionIndex = outputRelation.addPartition(std::move(output));
    auto finalisedOutput = outputRelation.finaliseAndGetPartition(partitionIndex);
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    this->pushUp(std::move(finalisedOutput));
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
    // remove the partition we pushed
    // we know that it is the latest partition (newly added array)
    // so it is cheap to remove
    outputRelation.removePartition(partitionIndex);
  }

  void close() override {
    ITTSection section{ITTProjectTask()};
    auto outputPtr = outputRelation.finaliseAndGetPartitions();
    auto& output = *outputPtr;
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    std::for_each(
        std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()),
        [this](auto&& partition) { this->pushUp(std::forward<decltype(partition)>(partition)); });
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
  }

private:
  Table outputRelation;
};
namespace {
boss::engines::bulk::Engine::Register<Project> const r101("Project"); // NOLINT
}

template <bool ReserveFullBatchSize, typename... TypedContainers> class BagOfContainers {
public:
  template <typename T> auto& getTyped() {
    auto& container = std::get<std::unique_ptr<T>>(containers);
    if(!container) {
      container = std::make_unique<T>();
      if constexpr(ReserveFullBatchSize) {
        container->reserve(bulk::Properties::getMicroBatchesMaxSize());
      }
    }
    return *container;
  }
  bool empty() const {
    return std::apply(
        [](auto const&... container) { return ((!container || container->empty()) && ...); },
        containers);
  }
  size_t size() const {
    return std::apply(
        [](auto const&... container) { return ((container ? container->size() : 0) + ...); },
        containers);
  }
  template <typename Func> void visit(Func&& func) const {
    auto callWithContainerIfExists = [&func](auto&& container) {
      if(container) {
        func(*container);
      }
    };
    std::apply([&callWithContainerIfExists](
                   auto const&... container) { (callWithContainerIfExists(container), ...); },
               containers);
  }
  template <typename Func> void visit(Func&& func) {
    auto callWithContainerIfExists = [&func](auto&& container) {
      if(container) {
        func(*container);
      }
    };
    std::apply([&callWithContainerIfExists](
                   auto&&... container) { (callWithContainerIfExists(container), ...); },
               containers);
  }
  template <typename Func> void reversedVisit(Func&& func) const {
    auto callWithContainerIfExists = [&func](auto&& container) {
      if(container) {
        func(*container);
      }
    };
    std::apply(
        [&callWithContainerIfExists](auto&&... container) {
          int dummy = 0;
          ((callWithContainerIfExists(container), dummy) = ...);
        },
        containers);
  }
  template <typename Func> void reversedVisit(Func&& func) {
    auto callWithContainerIfExists = [&func](auto&& container) {
      if(container) {
        func(*container);
      }
    };
    std::apply(
        [&callWithContainerIfExists](auto&&... container) {
          int dummy = 0;
          ((callWithContainerIfExists(container), dummy) = ...);
        },
        containers);
  }

  void clear() {
    auto clearContainerIfExists = [](auto&& container) {
      if(container) {
        container->clear();
      }
    };
    std::apply([&clearContainerIfExists](
                   auto&&... container) { (clearContainerIfExists(container), ...); },
               containers);
  }

protected:
  auto& getContainers() { return containers; }

private:
  std::tuple<std::unique_ptr<TypedContainers>...> containers;
};

template <typename KeyType, typename ValueType>
class TypedHashMap
    : public Map<
          std::invoke_result_t<decltype(&ValueArray<KeyType>::Value), ValueArray<KeyType>, int64_t>,
          ValueType> {
public:
  using KeyArrayElementType = KeyType;
};

/*template <typename ValueType>
class TypedHashMap<ComplexExpression, ValueType>
  : public Map<ComplexExpression, ValueType, CompareExpression<true, true, true>> {
public:
  using KeyArrayElementType = ComplexExpression;
};*/

template <typename ValueType>
class TypedHashMap<MultipleKeys, ValueType> : public Map<MultipleKeys, ValueType> {
public:
  using KeyArrayElementType = MultipleKeys;

  std::vector<MultipleKeys>
  prepareMultiplekeysListForHashing(ComplexExpression const& keyArraysExpr) {
    auto multipleKeyList = std::vector<MultipleKeys>();
    auto length = visit(
        [](auto& valueArrayPtr) -> int64_t {
          using KeyArrayType = std::decay_t<decltype(valueArrayPtr)>;
          if constexpr(std::is_convertible_v<KeyArrayType, std::shared_ptr<arrow::Array>>) {
            return valueArrayPtr->length();
          } else {
            auto oss = std::ostringstream();
            oss << valueArrayPtr; // (not an actual value array ptr...)
            throw std::logic_error("'" + oss.str() +
                                   "' not supported as a key for multi-key grouping");
          }
        },
        keyArraysExpr.getArguments().front());
    for(int64_t rowIndex = 0; rowIndex < length; ++rowIndex) {
      multipleKeyList.emplace_back(convertToMultipleKeys(keyArraysExpr, rowIndex));
    }
    return multipleKeyList;
  }

  void clear() {
    Map<MultipleKeys, ValueType>::clear();
    dictionary.clear();
  }

private:
  Set<std::string> dictionary;

  auto convertToMultipleKeys(ComplexExpression const& keyArraysExpr, int64_t keyArrayIndex) {
    auto const& keysArguments = keyArraysExpr.getArguments();
    auto numKeys = keysArguments.size();
    MultipleKeys multipleKeys;
    multipleKeys.resize(numKeys);
    for(size_t keyComponent = 0; keyComponent < numKeys; ++keyComponent) {
      visit(
          [&keyArrayIndex, &keyComponent, &multipleKeys, this](auto& keyArrayPtr) {
            using KeyArrayPtrType = std::decay_t<decltype(keyArrayPtr)>;
            if constexpr(std::is_convertible_v<KeyArrayPtrType,
                                               std::shared_ptr<arrow::FlatArray>>) {
              auto const& keyArray = *keyArrayPtr;
              using KeyArrayType = std::decay_t<decltype(keyArray)>;
              using KeyValueType = typename KeyArrayType::ElementType;
              if constexpr(std::is_same_v<KeyValueType, int32_t> ||
                           std::is_same_v<KeyValueType, int64_t> ||
                           std::is_same_v<KeyValueType, float_t> ||
                           std::is_same_v<KeyValueType, double_t>) {
                multipleKeys.set(keyComponent, keyArray[keyArrayIndex]);
              } else if constexpr(std::is_same_v<KeyValueType, std::string>) {
                auto const& str = keyArray[keyArrayIndex];
                // use only the 8 first characters for now as a workaround
                // multipleKeys.set(keyComponent, str.data(), str.size());
                // new method: temporarly store in a dictionary
                // the pointer is a unique address to any identical string
                auto* strPointer = &(*dictionary.emplace(str).first);
                multipleKeys.set(keyComponent, strPointer);
              } else {
                throw std::logic_error("key type not supported for multi-key grouping");
              }
            } else {
              auto oss = std::ostringstream();
              oss << keyArrayPtr; // (not an actual array ptr...)
              throw std::logic_error("'" + oss.str() +
                                     "' not supported as a key for multi-key grouping");
            }
          },
          keysArguments.at(keyComponent));
    }
    return multipleKeys;
  }
};

template <bool ReserveFullBatchSize, typename... ValueTypes>
class ExpressionHashMap
    : public BagOfContainers<
          ReserveFullBatchSize, TypedHashMap<bool, ValueTypes>...,
          TypedHashMap<int32_t, ValueTypes>..., TypedHashMap<int64_t, ValueTypes>...,
          TypedHashMap<float_t, ValueTypes>..., TypedHashMap<double_t, ValueTypes>...,
          TypedHashMap<std::string, ValueTypes>...,
          TypedHashMap<Symbol, ValueTypes>..., /*TypedHashMap<ComplexExpression, ValueTypes>...,*/
          TypedHashMap<MultipleKeys, ValueTypes>...> {
public:
  using DefaultValueType = std::tuple_element_t<0, std::tuple<ValueTypes...>>;
  template <typename T, typename U = DefaultValueType> auto& getTyped() {
    auto& container = std::get<std::unique_ptr<TypedHashMap<T, U>>>(getHashMaps());
    if(!container) {
      container = std::make_unique<TypedHashMap<T, U>>();
      if constexpr(ReserveFullBatchSize) {
        container->reserve(bulk::Properties::getMicroBatchesMaxSize());
      }
    }
    return *container;
  }

protected:
  auto& getHashMaps() { return this->getContainers(); }
};

template <typename... ArgumentTypes>
class Group : public boss::engines::bulk::Operator<Group, ArgumentTypes...> {
public:
  // static constexpr int MAX_COLUMN_SYMBOLS = 3;
  // using ColumnSymbolTypesT = RepeatedArgumentTypeOfAnySize_t<1, MAX_COLUMN_SYMBOLS, Symbol>;
  static constexpr int MAX_AGGREGATES = 8;
  using AggregateTypesT = RepeatedArgumentTypeOfAnySize_t<1, MAX_AGGREGATES, ComplexExpression>;

  using ArgumentTypesT = variant_merge_t<
      //ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr, ComplexExpression>>,
      //                      ColumnSymbolTypesT>,
      //ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr>>, ColumnSymbolTypesT>,
      //ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr, ComplexExpression>>,
      //                      AggregateTypesT>,
      ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr>>, AggregateTypesT>,
      variant<tuple<Table::PartitionPtr, Symbol>>,
      variant<tuple<Table::PartitionPtr, ComplexExpression, Symbol>>
      //ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr, Symbol>>, AggregateTypesT>,
      //ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr, ComplexExpression, Symbol>>,
      //                      AggregateTypesT>
      /*ArgumentTypeCombine_t<
          ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr, ComplexExpression>>,
                                ColumnSymbolTypesT>,
          AggregateTypesT>,
      ArgumentTypeCombine_t<
          ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr>>, ColumnSymbolTypesT>,
          AggregateTypesT>*/>;
  using Operator<Group, ArgumentTypes...>::Operator;
  template <typename... Aggregate>

  void operator()(Table::PartitionPtr const& partition, Symbol const& groupFunction,
                  Aggregate const&... aggregate) {
    auto addColumnNameToSchema = [this](auto const& expr) {
      outputRelation.getSchema().emplace_back(
          std::make_shared<arrow::Field>(getColumnSymbol(expr).getName(), nullptr));
    };
    auto initialiseSchemaIfNeeded = [&, this]() {
      if(outputRelation.getSchema().empty()) {
        outputRelation.getSchema().reserve(1 + sizeof...(aggregate));
        addColumnNameToSchema(groupFunction);
        (addColumnNameToSchema(aggregate), ...);
      }
    };
    // groupFunction is actually an aggregate
    initialiseSchemaIfNeeded();
    groupOp<ComplexExpression>(partition, convertAggregateExprToFunction(groupFunction),
                               convertAggregateExprToFunction(aggregate)...);
  }

  template <typename... Aggregate>
  void operator()(Table::PartitionPtr const& partition, ComplexExpression const& groupFunction,
                  Aggregate&&... aggregate) {
    auto addColumnNameToSchema = [this](auto const& expr) {
      outputRelation.getSchema().emplace_back(
          std::make_shared<arrow::Field>(getColumnSymbol(expr).getName(), nullptr));
    };
    auto addGroupFunctionToSchema = [&, this](auto const& expr) {
      if(expr.getHead().getName() == "Function") {
        addColumnNameToSchema(expr);
      } else {
        for(auto const& arg : expr.getArguments()) {
          visit(boss::utilities::overload(
                    [&](ComplexExpression const& e) { addColumnNameToSchema(e); },
                    [&](Symbol const& s) { addColumnNameToSchema(s); },
                    [&](auto const& /*unused*/) {}),
                arg);
        }
      }
    };
    auto initialiseSchemaIfNeeded = [&, this]() {
      if(outputRelation.getSchema().empty()) {
        outputRelation.getSchema().reserve(1 + sizeof...(aggregate));
        addGroupFunctionToSchema(groupFunction);
        (addColumnNameToSchema(aggregate), ...);
      }
    };
    auto handleAsOperatorAndReturnExpr = [&, this](ComplexExpression const& asExpr) {
      ExpressionArguments allArgs = asExpr.getArguments();
      auto allArgsSize = allArgs.size();
      ExpressionArguments symbolArgs;
      ExpressionArguments aggrArgs;
      symbolArgs.reserve(allArgsSize / 2);
      aggrArgs.reserve(allArgsSize / 2);
      bool isSymbolArg = false;
      std::partition_copy(
          std::make_move_iterator(std::begin(allArgs)), std::make_move_iterator(std::end(allArgs)),
          std::back_inserter(symbolArgs), std::back_inserter(aggrArgs),
          [&isSymbolArg](auto&& /*ignored*/) { return (isSymbolArg = !isSymbolArg); });
      // initialise the schema if needed
      if(outputRelation.getSchema().empty()) {
        bool useGroupFunction = &asExpr != &groupFunction;
        outputRelation.getSchema().reserve((useGroupFunction ? 0 : 1) + sizeof...(aggregate));
        if(useGroupFunction) {
          addGroupFunctionToSchema(groupFunction);
        }
        for(auto&& arg : symbolArgs) {
          outputRelation.getSchema().emplace_back(std::make_shared<arrow::Field>(
              std::move(std::get<Symbol>(std::move(arg)).getName()), nullptr));
        }
      }
      // return only the list of aggregate operators
      return ComplexExpression("As"_, std::move(aggrArgs));
    };
    auto const& functionName = groupFunction.getHead().getName();
    if(functionName != "By" && functionName != "Function") {
      // groupFunction is actually an aggregate
      if constexpr(sizeof...(aggregate) == 0) {
        if(functionName == "As") {
          // handle "As" operator (should be the only aggregate expression)
          auto aggrExpr = handleAsOperatorAndReturnExpr(groupFunction);
          groupOp<ComplexExpression>(partition, aggrExpr);
          return;
        }
      }
      initialiseSchemaIfNeeded();
      groupOp<ComplexExpression>(partition, convertAggregateExprToFunction(groupFunction),
                                 convertAggregateExprToFunction(aggregate)...);
      return;
    }
    if constexpr(sizeof...(aggregate) == 1 &&
                 (... && std::is_same_v<std::decay_t<Aggregate>, ComplexExpression>)) {
      if((... && (aggregate.getHead().getName() == "As"))) {
        // handle "As" operator (should be the only aggregate expression)
        auto aggrExpr = handleAsOperatorAndReturnExpr(aggregate...);
        groupOp(partition, groupFunction, aggrExpr);
        return;
      }
    }
    initialiseSchemaIfNeeded();
    groupOp(partition, groupFunction, convertAggregateExprToFunction(aggregate)...);
  }

  template <typename... Aggregates>
  void groupOp(Table::PartitionPtr const& partition, ComplexExpression const& groupFunction,
               Aggregates const&... aggregateFunctions) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Group ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
      return;
    }
    ITTSection section{ITTGroupTask()};
    useGroupFunction = true;
    // register symbol for each column (needed for both the group function and the aggregates)
    auto temporarlyRegisterSymbols = RegisterColumnSymbols(partition);
    // evaluate the keys using the group function
    auto evaluateKeys = [&]() -> Expression {
      if(groupFunction.getHead().getName() == "By") {
        auto const& unevaluatedArgs = groupFunction.getArguments();
        auto argSize = unevaluatedArgs.size();
        ExpressionArguments evaluatedArgs;
        evaluatedArgs.reserve(argSize);
        for(auto const& arg : unevaluatedArgs) {
          visit([this, &evaluatedArgs](
                    auto const& val) { evaluatedArgs.emplace_back(this->evaluateInternal(val)); },
                arg);
        }
        if(argSize > 1) {
          return ComplexExpression(groupFunction.getHead(), std::move(evaluatedArgs));
        }
        return std::move(evaluatedArgs[0]);
      }
      if(groupFunction.getHead().getName() != "Function") {
        return this->evaluateInternal(groupFunction);
      }
      auto oldGroupFunctionNumArguments = groupFunction.getArguments().size();
      ExpressionArguments groupFunctionArgs;
      groupFunctionArgs.reserve(oldGroupFunctionNumArguments + 1);
      for(int i = 0; i < oldGroupFunctionNumArguments; ++i) {
        groupFunctionArgs.emplace_back(groupFunction.cloneArgument(i));
      }
      groupFunctionArgs.emplace_back("List"_(partition));
      return this->evaluateInternal(
          ComplexExpression(groupFunction.getHead(), std::move(groupFunctionArgs)));
    };
    auto keys = evaluateKeys();
    // build the row indexes based on the keys
    group(partition, std::move(keys), aggregateFunctions...);
  }

  template <typename FirstAggregate, typename... Aggregates>
  void groupOp(Table::PartitionPtr const& partition, FirstAggregate const& firstAggregateFunctions,
               Aggregates const&... aggregateFunctions) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Group ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
      return;
    }
    ITTSection section{ITTGroupTask()};
    // special case when there is no group function, we need only a constant as a key
    group(partition, int32_t(0), firstAggregateFunctions, aggregateFunctions...);
  }

  void close() override {
    ITTSection section{ITTGroupTask()};
    auto outputPtr = buildFinalOutput();
    auto& output = *outputPtr;
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    std::for_each(
        std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()),
        [this](auto&& partition) { this->pushUp(std::forward<decltype(partition)>(partition)); });
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
    // outputPartitionAndIndexFromKey.clear();
    listOfGenericAggregatesPerKey.clear();
    cacheOfPartitionsAndKeys.clear();
    useGroupFunction = false;
    noAggregate = false;
  }

private:
  template <typename KeyArrayType> class CacheOfPartitionsAndKeys {
  public:
    template <typename T>
    void insert(Table::PartitionPtr const& partition, ValueArrayPtr<T>&& keysPtr) {
      inputPartitionsAndKeys.emplace_back(partition, keysPtr);
    }
    template <typename T> void insert(Table::PartitionPtr const& partition, T&& key) {
      inputPartitionsAndKeys.emplace_back(partition, std::forward<decltype(key)>(key));
    }
    void clear() { inputPartitionsAndKeys.clear(); }
    auto& getPartition(size_t index) { return inputPartitionsAndKeys[index].first; }
    auto const& getPartition(size_t index) const { return inputPartitionsAndKeys[index].first; }
    auto& getKey(size_t index) { return inputPartitionsAndKeys[index].second; }
    auto const& getKey(size_t index) const { return inputPartitionsAndKeys[index].second; }

  private:
    std::vector<std::pair<Table::PartitionPtr, KeyArrayType>> inputPartitionsAndKeys;
  };

  Table outputRelation;
  // ExpressionHashMap<std::pair<size_t, size_t>> outputPartitionAndIndexFromKey;

  using GenericAggregateHashmap = ExpressionHashMap<false, int64_t, double_t>;
  std::vector<GenericAggregateHashmap> listOfGenericAggregatesPerKey;

  BagOfContainers<false, CacheOfPartitionsAndKeys<bool>, CacheOfPartitionsAndKeys<int32_t>,
                  CacheOfPartitionsAndKeys<int64_t>, CacheOfPartitionsAndKeys<float_t>,
                  CacheOfPartitionsAndKeys<double_t>, CacheOfPartitionsAndKeys<std::string>,
                  CacheOfPartitionsAndKeys<Symbol>, CacheOfPartitionsAndKeys<ComplexExpression>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<bool>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<int32_t>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<int64_t>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<float_t>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<double_t>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<std::string>>,
                  CacheOfPartitionsAndKeys<ValueArrayPtr<Symbol>>>
      cacheOfPartitionsAndKeys;

  bool useGroupFunction = false;
  bool noAggregate = false;

  static ComplexExpression const&
  convertAggregateExprToFunction(ComplexExpression const& aggrExpr) {
    return aggrExpr;
  }
  static ComplexExpression convertAggregateExprToFunction(Symbol const& aggrSymbol) {
    auto parameters = "List"_("tuple"_);
    ExpressionArguments bodyArgs;
    bodyArgs.emplace_back("tuple"_);
    auto body = ComplexExpression(aggrSymbol, std::move(bodyArgs));
    ExpressionArguments aggregateArgs;
    aggregateArgs.reserve(2);
    aggregateArgs.emplace_back(std::move(parameters));
    aggregateArgs.emplace_back(std::move(body));
    return ComplexExpression("Function"_, std::move(aggregateArgs));
  }
  template <typename AggregateType>
  static ComplexExpression convertAggregateExprToFunction(AggregateType const& aggrExpr) {
    return "Function"_(aggrExpr);
  }

  static Symbol const& getColumnSymbol(Symbol const& symbol) { return symbol; }
  static Symbol getColumnSymbol(ComplexExpression const& aggregateExpr) {
    auto const& aggregateArgs = aggregateExpr.getArguments();
    if(aggregateArgs.empty()) {
      return aggregateExpr.getHead();
    }
    if(aggregateExpr.getHead().getName() == "Function" && aggregateArgs.size() > 1) {
      auto const& functionParam = visit(
          boss::utilities::overload([&aggregateExpr](auto const& /*other*/) { return Symbol(""); },
                                    [](ComplexExpression const& paramList) {
                                      auto const& paramListArgs = paramList.getArguments();
                                      return paramListArgs.empty()
                                                 ? Symbol("")
                                                 : get<Symbol>(paramListArgs.front());
                                    },
                                    [](Symbol const& param) { return param; }),
          aggregateArgs.front());
      auto const& functionBody = aggregateArgs.at(1);
      return visit(boss::utilities::overload(
                       [](auto const& other) { return getColumnSymbol("Function"_(other)); },
                       [&functionParam](ComplexExpression const& expr) {
                         auto symbol = getColumnSymbol(expr);
                         return symbol != functionParam ? symbol : expr.getHead();
                       },
                       [](Symbol const& symbol) { return symbol; }),
                   functionBody);
    }
    // recursively combine the head and arguments as the column name
    return (Symbol)std::accumulate(
        aggregateArgs.begin(), aggregateArgs.end(), aggregateExpr.getHead().getName(),
        [](auto const& symbolName, auto const& arg) {
          return symbolName + "_" +
                 visit(boss::utilities::overload(
                           [](auto const& value) {
                             using ValueType = std::decay_t<decltype(value)>;
                             if constexpr(std::is_convertible_v<ValueType,
                                                                std::shared_ptr<arrow::Array>>) {
                               return std::string("Array");
                             } else {
                               return std::to_string(value);
                             }
                           },
                           [](Table::PartitionVectorPtr const& /*partitions*/) {
                             return std::string("Relation");
                           },
                           [](ComplexExpression const& expr) {
                             return getColumnSymbol(expr).getName();
                           },
                           [](Symbol const& symbol) { return symbol.getName(); },
                           [](std::string const& str) { return str; }),
                       arg);
        });
  }

  template <typename... Aggregates>
  void group(Table::PartitionPtr const& partition, Expression&& keys,
             Aggregates const&... aggregateFunctions) {
    std::visit(boss::utilities::overload(
                   [this, &partition, &aggregateFunctions...](auto&& arrayOrConstant) {
                     group(partition, std::forward<decltype(arrayOrConstant)>(arrayOrConstant),
                           aggregateFunctions...);
                   },
                   [this, &partition, &aggregateFunctions...](
                       std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
                     throw std::logic_error("unsupported unevaluated key for grouping");
                   },
                   [this, &partition, &aggregateFunctions...](ComplexExpression&& expr) {
                     if(expr.getArguments().size() > MAX_MULTIPLE_KEY_SIZE) {
                       throw std::logic_error(
                           std::to_string(expr.getArguments().size()) +
                           " keys provided for multi-key grouping but the maximum supported is " +
                           std::to_string(MAX_MULTIPLE_KEY_SIZE));
                       return;
                     }
                     group(partition, std::move(expr), aggregateFunctions...);
                   },
                   [](Table::PartitionVectorPtr&& /*other*/) {
                     throw std::logic_error("type not supported as a key for grouping");
                   }),
               std::move(keys));
  }

  template <typename T, typename... Aggregates>
  void group(Table::PartitionPtr const& partition, T&& key,
             Aggregates const&... aggregateFunctions) {
    auto const& keys = Utilities::RangeGenerator<T>(partition->length(), key);
    if constexpr(sizeof...(aggregateFunctions) == 0) {
      noAggregate = true;
      listOfGenericAggregatesPerKey.resize(1);
      auto& aggregatePerKey = listOfGenericAggregatesPerKey.front();
      using KeyType = typename Utilities::RangeGenerator<T>::ElementType;
      auto& hashmap = aggregatePerKey.template getTyped<KeyType, int64_t>();
      if constexpr(std::is_same_v<T, Symbol>) {
        hashmap.try_emplace(key.getName(), 0);
      } else {
        hashmap.try_emplace(key, 0);
      }
    } else {
      if((... && (aggregateFunctions.getHead().getName() == "As"))) {
        listOfGenericAggregatesPerKey.resize((... + aggregateFunctions.getArguments().size()));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (
            [&](auto const& args) {
              for(auto const& aggr : args) {
                handleAsSpecialization(partition, keys, *(genericAggregatePerKeyIt++),
                                       get<ComplexExpression>(aggr));
              }
            }(aggregateFunctions.getArguments()),
            ...);
      } else {
        listOfGenericAggregatesPerKey.resize(sizeof...(aggregateFunctions));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (..., handleAsSpecialization(partition, keys, *(genericAggregatePerKeyIt++),
                                     aggregateFunctions));
      }
    }
    cacheOfPartitionsAndKeys.template getTyped<CacheOfPartitionsAndKeys<T>>().insert(
        partition, std::forward<T>(key));
  }

  template <typename T, typename... Aggregates>
  void group(Table::PartitionPtr const& partition, ValueArrayPtr<T>&& keysPtr,
             Aggregates const&... aggregateFunctions) {
    auto const& keys = *keysPtr;
    if constexpr(sizeof...(aggregateFunctions) == 0) {
      noAggregate = true;
      listOfGenericAggregatesPerKey.resize(1);
      auto& aggregatePerKey = listOfGenericAggregatesPerKey.front();
      using KeyType = typename ValueArray<T>::ElementType;
      auto& hashmap = aggregatePerKey.template getTyped<KeyType, int64_t>();
      auto length = keys.length();
      for(int64_t index = 0; index < length; ++index) {
        auto const& key = keys[index];
        hashmap.try_emplace(key, 0);
      }
    } else {
      if((... && (aggregateFunctions.getHead().getName() == "As"))) {
        listOfGenericAggregatesPerKey.resize((... + aggregateFunctions.getArguments().size()));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (
            [&](auto const& args) {
              for(auto const& aggr : args) {
                handleAsSpecialization(partition, keys, *(genericAggregatePerKeyIt++),
                                       get<ComplexExpression>(aggr));
              }
            }(aggregateFunctions.getArguments()),
            ...);
      } else {
        listOfGenericAggregatesPerKey.resize(sizeof...(aggregateFunctions));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (..., handleAsSpecialization(partition, keys, *(genericAggregatePerKeyIt++),
                                     aggregateFunctions));
      }
    }
    cacheOfPartitionsAndKeys.template getTyped<CacheOfPartitionsAndKeys<ValueArrayPtr<T>>>().insert(
        partition, std::move(keysPtr));
  }

  template <typename... Aggregates>
  void group(Table::PartitionPtr const& partition, ComplexExpression&& multipleKeys,
             Aggregates const&... aggregateFunctions) {
    if constexpr(sizeof...(aggregateFunctions) == 0) {
      noAggregate = true;
      listOfGenericAggregatesPerKey.resize(1);
      auto& aggregatePerKey = listOfGenericAggregatesPerKey.front();
      auto& hashmap = aggregatePerKey.template getTyped<MultipleKeys, int64_t>();
      auto multipleKeysList = hashmap.prepareMultiplekeysListForHashing(multipleKeys);
      for(auto const& multipleKey : multipleKeysList) {
        hashmap.try_emplace(multipleKey, 0);
      }
    } else {
      if((... && (aggregateFunctions.getHead().getName() == "As"))) {
        listOfGenericAggregatesPerKey.resize((... + aggregateFunctions.getArguments().size()));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (
            [&](auto const& args) {
              for(auto const& aggr : args) {
                handleAsSpecialization(partition, multipleKeys, *(genericAggregatePerKeyIt++),
                                       get<ComplexExpression>(aggr));
              }
            }(aggregateFunctions.getArguments()),
            ...);
      } else {
        listOfGenericAggregatesPerKey.resize(sizeof...(aggregateFunctions));
        auto genericAggregatePerKeyIt = listOfGenericAggregatesPerKey.begin();
        (..., handleAsSpecialization(partition, multipleKeys, *(genericAggregatePerKeyIt++),
                                     aggregateFunctions));
      }
    }
    cacheOfPartitionsAndKeys.template getTyped<CacheOfPartitionsAndKeys<ComplexExpression>>()
        .insert(partition, std::move(multipleKeys));
  }

  template <typename KeyArrayType>
  bool handleAsSpecialization(Table::PartitionPtr const& partition, KeyArrayType const& keys,
                              GenericAggregateHashmap& aggregatePerKey,
                              ComplexExpression const& aggregateFunction) {
    auto const& aggregateFunctionName = aggregateFunction.getHead().getName();
    auto const& aggregateFunctionArgs = aggregateFunction.getArguments();
    if(aggregateFunctionName == "Function" && aggregateFunctionArgs.size() >= 2) {
      return handleAsSpecialization(partition, keys, aggregatePerKey,
                                    get<ComplexExpression>(aggregateFunctionArgs[1]),
                                    aggregateFunction.cloneArgument(0));
    }
    return handleAsSpecialization(partition, keys, aggregatePerKey, aggregateFunction, "List"_());
  }

  template <typename KeyArrayType>
  bool handleAsSpecialization(Table::PartitionPtr const& partition, KeyArrayType const& keys,
                              GenericAggregateHashmap& aggregatePerKey,
                              ComplexExpression const& aggregateOperator,
                              Expression&& /*functionParameters*/) {
    if(aggregateOperator.getHead().getName() == "Count") {
      if(countAggregate(aggregatePerKey, keys)) {
        return true;
      }
    }
    if(aggregateOperator.getHead().getName() == "Sum") {
      // register symbol for each column
      auto temporarlyRegisterSymbols = std::unique_ptr<RegisterColumnSymbols>();
      if(!useGroupFunction) { // only if we did not already do it for the group function
        temporarlyRegisterSymbols.reset(new RegisterColumnSymbols(partition));
      }
      // evaluate the argument that was passed to the aggregate function
      // and pass the resulting array to the hardcoded aggregation
      auto arrayArg = this->evaluateInternal(aggregateOperator.cloneArgument(0));
      if(visit(
             [&keys, &aggregatePerKey](auto const& columnPtr) {
               return sumAggregate(aggregatePerKey, keys, columnPtr);
             },
             arrayArg)) {
        return true;
      }
    }
    auto oss = std::ostringstream();
    oss << aggregateOperator.getHead().getName();
    throw std::logic_error("aggregate operator '" + oss.str() + "' not supported");
    return false;
  }

  template <typename KeyArrayType>
  static bool countAggregate(GenericAggregateHashmap& aggregatePerKey, KeyArrayType const& keys) {
    using KeyType = typename KeyArrayType::ElementType;
    auto& hashmap = aggregatePerKey.getTyped<KeyType, int64_t>();
    auto length = keys.length();
    for(int64_t index = 0; index < length; ++index) {
      auto const& key = keys[index];
#if defined(USE_TSL_HASHMAP)
      ++hashmap.try_emplace(key, 0).first.value();
#else
      ++hashmap.try_emplace(key, 0).first->second;
#endif
    }
    return true;
  }

  static bool countAggregate(GenericAggregateHashmap& aggregatePerKey,
                             ComplexExpression const& multipleKeysExpr) {
    auto& hashmap = aggregatePerKey.getTyped<MultipleKeys, int64_t>();
    auto multipleKeysList = hashmap.prepareMultiplekeysListForHashing(multipleKeysExpr);
    for(auto&& multipleKey : std::move(multipleKeysList)) {
#if defined(USE_TSL_HASHMAP)
      ++hashmap.try_emplace(std::move(multipleKey), 0).first.value();
#else
      ++hashmap.try_emplace(std::move(multipleKey), 0).first->second;
#endif
    }
    return true;
  }

  template <typename KeyArrayType, typename ValueType>
  static bool sumAggregate(GenericAggregateHashmap& aggregatePerKey, KeyArrayType const& keys,
                           ValueArrayPtr<ValueType> const& columnPtr) {
    if constexpr(std::is_same_v<ValueType, int32_t> || std::is_same_v<ValueType, int64_t> ||
                 std::is_same_v<ValueType, float_t> || std::is_same_v<ValueType, double_t>) {
      using KeyType = typename KeyArrayType::ElementType;
      auto const& column = *columnPtr;
      auto doAggregate = [&keys, &column](auto& hashmap) {
        using HashMapKeyType = typename std::decay_t<decltype(hashmap)>::key_type;
        using HashMapValueType = typename std::decay_t<decltype(hashmap)>::value_type::second_type;
        auto length = keys.length();
        for(int64_t index = 0; index < length; ++index) {
          auto const& key = keys[index];
          auto it = hashmap.try_emplace(static_cast<HashMapKeyType>(key), 0).first;
#if defined(USE_TSL_HASHMAP)
          it.value() += static_cast<HashMapValueType>(column[index]);
#else
          it->second += static_cast<HashMapValueType>(column[index]);
#endif
        }
      };
      // if the aggregation already started, just stick on the previous type
      // (so we don't end up with one aggregation per type if the type is heterogeneous
      bool done = false;
      aggregatePerKey.visit([&doAggregate, &done](auto& typedHashmap) {
        if constexpr(std::is_convertible_v<
                         KeyType, typename std::decay_t<decltype(typedHashmap)>::key_type>) {
          if(!typedHashmap.empty()) {
            doAggregate(typedHashmap);
            done = true;
          }
        }
      });
      if(!done) {
        using AggregateType =
            std::conditional_t<std::is_floating_point_v<ValueType>, double_t, int64_t>;
        doAggregate(aggregatePerKey.getTyped<KeyType, AggregateType>());
      }
      return true;
    }
    return false;
  }

  template <typename ValueType>
  static bool sumAggregate(GenericAggregateHashmap& aggregatePerKey,
                           ComplexExpression const& multipleKeysExpr,
                           ValueArrayPtr<ValueType> const& columnPtr) {
    if constexpr(std::is_same_v<ValueType, int32_t> || std::is_same_v<ValueType, int64_t> ||
                 std::is_same_v<ValueType, float_t> || std::is_same_v<ValueType, double_t>) {
      auto const& column = *columnPtr;
      auto doAggregate = [&multipleKeysExpr, &column](auto& hashmap) {
        auto multipleKeysList = hashmap.prepareMultiplekeysListForHashing(multipleKeysExpr);
        using HashMapValueType = typename std::decay_t<decltype(hashmap)>::value_type::second_type;
        auto length = column.length();
        for(int64_t rowIndex = 0; rowIndex < length; ++rowIndex) {
          auto it = hashmap.try_emplace(std::move(multipleKeysList[rowIndex]), 0).first;
#if defined(USE_TSL_HASHMAP)
          it.value() += static_cast<HashMapValueType>(column[rowIndex]);
#else
          it->second += static_cast<HashMapValueType>((ValueType)column[rowIndex]);
#endif
        }
      };
      // if the aggregation already started, just stick on the previous type
      // (so we don't end up with one aggregation per type if the type is heterogeneous
      bool done = false;
      aggregatePerKey.visit([&doAggregate, &done](auto& typedHashmap) {
        if constexpr(std::is_same_v<typename std::decay_t<decltype(typedHashmap)>::key_type,
                                    MultipleKeys>) {
          if(!typedHashmap.empty()) {
            doAggregate(typedHashmap);
            done = true;
          }
        }
      });
      if(!done) {
        using AggregateType =
            std::conditional_t<std::is_floating_point_v<ValueType>, double_t, int64_t>;
        doAggregate(aggregatePerKey.getTyped<MultipleKeys, AggregateType>());
      }
      return true;
    }
    return false;
  }

  template <typename KeyArrayType, typename T>
  static bool sumAggregate(GenericAggregateHashmap& /*aggregatePerKey*/,
                           KeyArrayType const& /*keys*/, T const& /*column*/) {
    return false;
  }

  Table::PartitionVectorPtr buildFinalOutput() {
    std::vector<arrow::ArrayVector> columnArraysPerPartition;
    if(!listOfGenericAggregatesPerKey.empty()) {
      auto totalNumRows = listOfGenericAggregatesPerKey[0].size();
      auto numColumns = outputRelation.getSchema().size();
      auto numPartitions = 1 + (totalNumRows / bulk::Properties::getMicroBatchesMaxSize());
      columnArraysPerPartition.resize(numPartitions);
      for(auto& columnArrays : columnArraysPerPartition) {
        columnArrays.reserve(numColumns);
      }
    }
    // output the grouping keys first
    if(useGroupFunction && !listOfGenericAggregatesPerKey.empty()) {
      auto const& anyAggregatePerKey = listOfGenericAggregatesPerKey[0];
      anyAggregatePerKey.visit([&columnArraysPerPartition, this](auto const& typedHashmap) {
        if(typedHashmap.empty()) {
          return;
        }
        auto typedHashmapIt = typedHashmap.begin();
        auto numValuesLeft = typedHashmap.size();
        for(auto columnArraysIt = columnArraysPerPartition.begin();
            columnArraysIt < columnArraysPerPartition.end(); ++columnArraysIt) {
          auto numValues = numValuesLeft > bulk::Properties::getMicroBatchesMaxSize()
                               ? bulk::Properties::getMicroBatchesMaxSize()
                               : numValuesLeft;
          numValuesLeft -= numValues;
          auto& columnArrays = *columnArraysIt;
          using HashMapType = std::decay_t<decltype(typedHashmap)>;
          using KeyType = typename HashMapType::KeyArrayElementType;
          if constexpr(std::is_same_v<KeyType, MultipleKeys>) {
            // retrieve the types from the cached key,
            // convert it back and build new value arrays
            auto const& partitionsAndKeys =
                cacheOfPartitionsAndKeys
                    .template getTyped<CacheOfPartitionsAndKeys<ComplexExpression>>();
            auto const& multipleKeyReference = partitionsAndKeys.getKey(
                0); // assuming all partitions have the same type for the multiple keys
            auto const& multipleKeyReferenceArgs = multipleKeyReference.getArguments();
            auto multipleKeyReferenceSize = multipleKeyReferenceArgs.size();
            auto startMapIt = typedHashmapIt;
            for(size_t keyComponent = 0; keyComponent < multipleKeyReferenceSize; ++keyComponent) {
              auto const& keyReference = multipleKeyReferenceArgs.at(keyComponent);
              visit(
                  [&columnArrays, &numValues, &typedHashmapIt,
                   &keyComponent](auto const& keyArrayPtr) {
                    using KeyArrayPtrType = std::decay_t<decltype(keyArrayPtr)>;
                    if constexpr(std::is_convertible_v<KeyArrayPtrType,
                                                       std::shared_ptr<arrow::FlatArray>>) {
                      auto const& keyArray = *keyArrayPtr;
                      using KeyArrayType = std::decay_t<decltype(keyArray)>;
                      using KeyValueType = typename KeyArrayType::ElementType;
                      if constexpr(std::is_same_v<KeyValueType, int32_t> ||
                                   std::is_same_v<KeyValueType, int64_t> ||
                                   std::is_same_v<KeyValueType, float_t> ||
                                   std::is_same_v<KeyValueType, double_t>) {
                        auto keyColumnBuilder = ValueBuilder<KeyValueType>(numValues);
                        for(int64_t index = 0; index < numValues; ++typedHashmapIt) {
                          auto const& [multipleKey, value] = *typedHashmapIt;
                          keyColumnBuilder[index++] =
                              multipleKey.template get<KeyValueType>(keyComponent);
                        }
                        columnArrays.emplace_back((KeyArrayPtrType)keyColumnBuilder);
                      } else if constexpr(std::is_same_v<KeyValueType, std::string>) {
                        auto keyColumnBuilder = ValueBuilder<KeyValueType>(0);
                        for(int64_t index = 0; index < numValues; ++index, ++typedHashmapIt) {
                          auto const& [multipleKey, value] = *typedHashmapIt;
                          // auto rawData =
                          //     multipleKey.get<uint8_t const(&)[sizeof(int64_t)]>(keyComponent);
                          // int64_t length = 0;
                          // for(; length < sizeof(int64_t) && rawData[length] != 0; ++length) {
                          // }
                          // keyColumnBuilder.Append(rawData, length);
                          auto& str = *multipleKey.template get<std::string*>(keyComponent);
                          keyColumnBuilder.Append(str);
                        }
                        columnArrays.emplace_back((KeyArrayPtrType)keyColumnBuilder);
                      }
                    }
                  },
                  keyReference);
              typedHashmapIt = startMapIt;
            }
          } else if constexpr(std::is_same_v<KeyType, std::string> ||
                              std::is_same_v<KeyType, Symbol>) {
            auto keyColumnBuilder = ValueBuilder<KeyType>(0);
            for(int64_t index = 0; index < numValues; ++index, ++typedHashmapIt) {
              auto const& [key, value] = *typedHashmapIt;
              keyColumnBuilder.Append(key);
            }
            columnArrays.emplace_back((ValueArrayPtr<KeyType>)keyColumnBuilder);
          } else {
            auto keyColumnBuilder = ValueBuilder<KeyType>(numValues);
            if constexpr(std::is_same_v<KeyType, bool>) {
              keyColumnBuilder.compute([&typedHashmapIt]() { return (typedHashmapIt++)->first; });
            } else {
              for(int64_t index = 0; index < numValues; ++typedHashmapIt) {
                auto const& [key, value] = *typedHashmapIt;
                keyColumnBuilder[index++] = key;
              }
            }
            columnArrays.emplace_back((ValueArrayPtr<KeyType>)keyColumnBuilder);
          }
        }
      });
    }
    // then output the aggregates
    if(!noAggregate) {
      for(auto const& aggregatePerKey : listOfGenericAggregatesPerKey) {
        aggregatePerKey.visit([&columnArraysPerPartition](auto const& typedHashmap) {
          if(typedHashmap.empty()) {
            return;
          }
          auto typedHashmapIt = typedHashmap.begin();
          auto numValuesLeft = typedHashmap.size();
          for(auto columnArraysIt = columnArraysPerPartition.begin();
              columnArraysIt < columnArraysPerPartition.end(); ++columnArraysIt) {
            auto numValues = numValuesLeft > bulk::Properties::getMicroBatchesMaxSize()
                                 ? bulk::Properties::getMicroBatchesMaxSize()
                                 : numValuesLeft;
            numValuesLeft -= numValues;
            auto& columnArrays = *columnArraysIt;
            using ValueType = std::decay_t<decltype(typedHashmap.begin()->second)>;
            auto columnBuilder = ValueBuilder<ValueType>(numValues);
            for(int64_t index = 0; index < numValues; ++typedHashmapIt) {
              auto const& [key, value] = *typedHashmapIt;
              columnBuilder[index++] = value;
            }
            columnArrays.emplace_back((ValueArrayPtr<ValueType>)columnBuilder);
          }
        });
      }
    }
    for(auto&& columnArrays : columnArraysPerPartition) {
      if(outputRelation.getSchema().size() != columnArrays.size()) {
        throw std::logic_error(
            "mismatch between schema size (" + std::to_string(outputRelation.getSchema().size()) +
            ") and the number of column arrays (" + std::to_string(columnArrays.size()) + ")");
      }
      outputRelation.addPartition(std::make_shared<ComplexExpressionArray>(
          "List"_, outputRelation.getSchema(), std::forward<decltype(columnArrays)>(columnArrays)));
    }
    return outputRelation.finaliseAndGetPartitions();
  }
};
namespace {
boss::engines::bulk::Engine::Register<Group> const r102("Group"); // NOLINT
}

template <typename KeyArrayType> class CacheOfSortedPartitions {
private:
  static constexpr size_t NUM_INDICES_TO_STORE_BEFORE_SORTING = 100'000;
  static constexpr size_t DESC_BITS_VECTOR_INITIAL_SIZE = 8;

  template <typename K> using PartitionsAndKeys = std::vector<std::pair<Table::PartitionPtr, K>>;
  PartitionsAndKeys<KeyArrayType> inputPartitionsAndKeys;
  std::vector<std::pair<size_t, int64_t>> partitionIndicesAndOffsets;
  std::vector<std::vector<Expression>> additionalSortKeysAndForEachPartition;

  std::vector<bool> descBits;

  size_t maxSize;
  size_t sortedSize;

public:
  CacheOfSortedPartitions()
      : maxSize(0), sortedSize(0), descBits(DESC_BITS_VECTOR_INITIAL_SIZE, false) {}

  /* to be used by a Top operator*/
  void setMaxSize(size_t size) { maxSize = size; }

  void setDesc(size_t index, bool desc = true) {
    descBits.resize(index);
    descBits[index] = desc;
  }

  void insert(Table::PartitionPtr const& partition, KeyArrayType&& key) {
    auto prevSize = partitionIndicesAndOffsets.size();
    auto partitionSize = partition->length();
    partitionIndicesAndOffsets.reserve(prevSize + partitionSize);
    for(int i = 0; i < partitionSize; ++i) {
      partitionIndicesAndOffsets.emplace_back(inputPartitionsAndKeys.size(), i);
    }
    inputPartitionsAndKeys.emplace_back(partition, std::move(key));
    if(partitionIndicesAndOffsets.size() - sortedSize > NUM_INDICES_TO_STORE_BEFORE_SORTING) {
      if(descBits[0]) {
        sortPartitionsWith<true>(inputPartitionsAndKeys, sortedSize);
      } else {
        sortPartitionsWith<false>(inputPartitionsAndKeys, sortedSize);
      }
      sortedSize = partitionIndicesAndOffsets.size();
    }
  }

  template <typename AdditionalKeysIterator, typename AdditionalKeysIteratorEnd>
  void insert(Table::PartitionPtr const& partition, KeyArrayType&& key,
              AdditionalKeysIterator&& additionalKeysIt,
              AdditionalKeysIteratorEnd&& additionalKeysItEnd) {
    insert(partition, std::move(key));
    // with additional keys, just store them and they will be used to sort only the final output
    additionalSortKeysAndForEachPartition.resize(
        std::distance(additionalKeysIt, additionalKeysItEnd));
    auto sortKeyPerPartitionIt = additionalSortKeysAndForEachPartition.begin();
    for(; additionalKeysIt != additionalKeysItEnd; ++additionalKeysIt) {
      auto&& expr = *additionalKeysIt;
      sortKeyPerPartitionIt->emplace_back(std::move(expr));
      ++sortKeyPerPartitionIt;
    }
  }

  auto buildOutput(Table& outputRelation) {
    if(partitionIndicesAndOffsets.empty()) {
      return;
    }
    // handle remaining unsorted data
    if(partitionIndicesAndOffsets.size() > sortedSize) {
      if(descBits[0]) {
        sortPartitionsWith<true>(inputPartitionsAndKeys, sortedSize);
      } else {
        sortPartitionsWith<false>(inputPartitionsAndKeys, sortedSize);
      }
      sortedSize = partitionIndicesAndOffsets.size();
    }
    // in case of multi-key sorting, sort with all the remaining keys now
    // the additional keys are reverse-iterated to apply the correct sorting priority
    auto keyIndex = additionalSortKeysAndForEachPartition.size();
    maxSize = 0; // not anymore applying Top algorithm
    for(auto additionalSortKeysPerPartitionIt = additionalSortKeysAndForEachPartition.rbegin();
        additionalSortKeysPerPartitionIt != additionalSortKeysAndForEachPartition.rend();
        ++additionalSortKeysPerPartitionIt, --keyIndex) {
      auto& keyExprPerPartition = *additionalSortKeysPerPartitionIt;
      auto subkeyPerPartitionIt = std::make_move_iterator(keyExprPerPartition.begin());
      std::visit(
          boss::utilities::overload(
              [&keyIndex, &subkeyPerPartitionIt, this](auto const& firstPartitionSubkey) {
                using SubkeyType = std::decay_t<decltype(firstPartitionSubkey)>;
                // create a container for the typed keys
                PartitionsAndKeys<SubkeyType> partitionAndSubkeys;
                partitionAndSubkeys.reserve(inputPartitionsAndKeys.size());
                // then insert the subkey for the other partitions
                // (assuming they all have the same type)
                std::transform(
                    inputPartitionsAndKeys.begin(), inputPartitionsAndKeys.end(),
                    std::back_inserter(partitionAndSubkeys), [&](auto const& partitionAndMainKey) {
                      auto&& subkeyExpr = *(subkeyPerPartitionIt++);
                      auto subkey = get<SubkeyType>(std::move(subkeyExpr));
                      return std::make_pair(partitionAndMainKey.first, std::move(subkey));
                    });
                // we sort one by one each subgroup which has identical main key
                // it is necessary for applying the right sorting priority for the keys
                // also, it is also more efficient to sort smaller groups
                size_t startIndex = 0;
                auto [startPartitionIndex, startOffset] = partitionIndicesAndOffsets[startIndex];
                auto length = partitionIndicesAndOffsets.size();
                for(size_t index = 0; index < length; ++index) {
                  auto const& [partitionIndex, offset] = partitionIndicesAndOffsets[index];
                  if constexpr(std::is_convertible_v<KeyArrayType, std::shared_ptr<arrow::Array>>) {
                    auto const& mainKeyArray = *inputPartitionsAndKeys[partitionIndex].second;
                    auto const& startMainKeyArray =
                        *inputPartitionsAndKeys[startPartitionIndex].second;
                    if(mainKeyArray[offset] == startMainKeyArray[startOffset]) {
                      continue;
                    }
                  } else {
                    // special case where a whole partition has the same key
                    if(partitionIndex == startPartitionIndex) {
                      continue;
                    }
                  }
                  // this subset of the partition can be sorted again according to that subkey
                  if(startIndex < index - 1) { // sort only if at least 2 values
                    if(descBits[keyIndex]) {
                      sortPartitionsWith<true>(partitionAndSubkeys, startIndex, index, false);
                    } else {
                      sortPartitionsWith<false>(partitionAndSubkeys, startIndex, index, false);
                    }
                  }
                  startIndex = index;
                  startPartitionIndex = partitionIndicesAndOffsets[startIndex].first;
                  startOffset = partitionIndicesAndOffsets[startIndex].second;
                }
                if(startIndex < length - 1) { // sort remaining ones only if at least 2 values
                  if(descBits[keyIndex]) {
                    sortPartitionsWith<true>(partitionAndSubkeys, startIndex, length, false);
                  } else {
                    sortPartitionsWith<false>(partitionAndSubkeys, startIndex, length, false);
                  }
                }
              },
              [](ComplexExpression&& expr) {
                auto oss = std::ostringstream();
                oss << expr;
                throw std::logic_error(
                    "'" + oss.str() + "' complex expression not supported as a subkey for sorting");
              },
              [](Table::PartitionVectorPtr const& /*other*/) {
                throw std::logic_error("type not supported as a subkey for sorting");
              }),
          *subkeyPerPartitionIt);
    }
    // generate the output (vector of indices to partitions)
    auto outputPartitionIndex = outputRelation.numPartitions();
    auto insert = [this, &outputRelation, &outputPartitionIndex](int64_t inputPartitionIndex,
                                                                 auto const& tupleIndices) {
      auto length = tupleIndices.size();
      auto const& inputPartitionPtr = inputPartitionsAndKeys[inputPartitionIndex].first;
      if(outputPartitionIndex < outputRelation.numPartitions() &&
         outputRelation.hasEnoughSpaceInPartition(outputPartitionIndex, length)) {
        // add to existing partition
        auto builder = outputRelation.getPartitionBuilder(outputPartitionIndex);
        builder->appendRowsInIndexedOrder(*inputPartitionPtr, tupleIndices, true /*re-init order*/,
                                          outputRelation.getMaxGlobalIndex());
        outputPartitionIndex =
            outputRelation.setPartition(outputPartitionIndex, std::move(builder));
      } else {
        // create new partition
        int64_t expectedFullSizePartition = inputPartitionPtr->length();
        auto builder = std::make_shared<ComplexExpressionArrayBuilder>(inputPartitionPtr, true,
                                                                       expectedFullSizePartition);
        builder->appendRowsInIndexedOrder(*inputPartitionPtr, tupleIndices, true /*re-init order*/,
                                          outputRelation.getMaxGlobalIndex());
        outputPartitionIndex = outputRelation.addPartition(std::move(builder));
      }
      outputRelation.getMaxGlobalIndex() += length;
    };
    auto partitionIndexAndOffsetIt = partitionIndicesAndOffsets.begin();
    auto [currentPartitionIndex, firstOffset] = *(partitionIndexAndOffsetIt++);
    auto tupleIndices = std::vector<int64_t>{firstOffset};
    for(; partitionIndexAndOffsetIt != partitionIndicesAndOffsets.end();
        ++partitionIndexAndOffsetIt) {
      auto const& [partitionIndex, offset] = *partitionIndexAndOffsetIt;
      if(partitionIndex == currentPartitionIndex) {
        tupleIndices.emplace_back(offset);
        continue;
      }
      insert(currentPartitionIndex, tupleIndices);
      currentPartitionIndex = partitionIndex;
      tupleIndices.clear();
      tupleIndices.emplace_back(offset);
    }
    if(!tupleIndices.empty()) {
      insert(currentPartitionIndex, tupleIndices);
    }
  }

  void clear() {
    inputPartitionsAndKeys.clear();
    partitionIndicesAndOffsets.clear();
    additionalSortKeysAndForEachPartition.clear();
    maxSize = 0;
    sortedSize = 0;
  }

private:
  template <bool DESC, typename T>
  void sortPartitionsWith(PartitionsAndKeys<T> const& partitionsAndKeys, size_t startOffset = 0,
                          bool merge = true) {
    sortPartitionsWith<DESC>(partitionsAndKeys, startOffset, partitionIndicesAndOffsets.size(),
                             merge);
  }

  template <bool DESC, typename T>
  void sortPartitionsWith(PartitionsAndKeys<ValueArrayPtr<T>> const& partitionsAndKeys,
                          size_t startOffset, size_t endOffset, bool merge = true) {
    auto comparator = [&partitionsAndKeys](std::pair<size_t, size_t> const& lhs,
                                           std::pair<size_t, size_t> const& rhs) {
      auto const& [lhsPartitionIndex, lhsOffset] = lhs;
      auto const& [rhsPartitionIndex, rhsOffset] = rhs;
      auto const& lhsKeyArray = *partitionsAndKeys[lhsPartitionIndex].second;
      auto const& rhsKeyArray = *partitionsAndKeys[rhsPartitionIndex].second;
      if constexpr(DESC) {
        return lhsKeyArray[lhsOffset] > rhsKeyArray[rhsOffset];
      } else {
        return lhsKeyArray[lhsOffset] < rhsKeyArray[rhsOffset];
      }
    };
    auto startIt = partitionIndicesAndOffsets.begin() + startOffset;
    auto endIt = partitionIndicesAndOffsets.begin() + endOffset;
    // general case, sort the indices according to the keys
    if(maxSize == 0) {
      // sort only the new values
      std::stable_sort(startIt, endIt, comparator);
      if(merge) {
        // merge with the beginning part of the container, which was assumed to be sorted
        std::inplace_merge(partitionIndicesAndOffsets.begin(), startIt, endIt, comparator);
      }
    } else {
      // same but keep only the top values
      auto numToKeep = endOffset < startOffset + maxSize ? endOffset - startOffset : maxSize;
      std::partial_sort(startIt, startIt + numToKeep, endIt, comparator);
      if(merge) {
        std::inplace_merge(partitionIndicesAndOffsets.begin(), startIt, startIt + numToKeep,
                           comparator);
      }
      if(partitionIndicesAndOffsets.size() > maxSize) {
        partitionIndicesAndOffsets.resize(maxSize);
      }
    }
  }

  template <bool DESC, typename T>
  void sortPartitionsWith(PartitionsAndKeys<T> const& partitionsAndKeys, size_t startOffset,
                          size_t endOffset, bool merge = true) {
    if(!merge) {
      return; // nothing to do here if we don't need to merge
    }
    auto comparator = [&partitionsAndKeys](std::pair<size_t, size_t> const& lhs,
                                           std::pair<size_t, size_t> const& rhs) {
      auto const& lhsPartitionIndex = lhs.first;
      auto const& rhsPartitionIndex = rhs.first;
      auto const& lhsKey = partitionsAndKeys[lhsPartitionIndex].second;
      auto const& rhsKey = partitionsAndKeys[rhsPartitionIndex].second;
      if constexpr(std::is_same_v<T, Symbol>) {
        if constexpr(DESC) {
          return lhsKey.getName() > rhsKey.getName();
        } else {
          return lhsKey.getName() < rhsKey.getName();
        }
      } else {
        return lhsKey < rhsKey;
      }
    };
    auto startIt = partitionIndicesAndOffsets.begin() + startOffset;
    auto endIt = partitionIndicesAndOffsets.begin() + endOffset;
    // special case with only a single key per partition, no need to sort the new elements
    // only merge with the beginning part of the container to sort between the partitions
    if(maxSize == 0) {
      std::inplace_merge(partitionIndicesAndOffsets.begin(), startIt, endIt, comparator);
    } else {
      // same but keep only the top values
      auto numToKeep = endOffset < startOffset + maxSize ? endOffset - startOffset : maxSize;
      std::inplace_merge(partitionIndicesAndOffsets.begin(), startIt, startIt + numToKeep,
                         comparator);
      if(partitionIndicesAndOffsets.size() > maxSize) {
        partitionIndicesAndOffsets.resize(maxSize);
      }
    }
  }
};

template <typename... ArgumentTypes>
class Sort : public boss::engines::bulk::Operator<Sort, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr, ComplexExpression>>;
  using Operator<Sort, ArgumentTypes...>::Operator;

  void operator()(Table::PartitionPtr const& partition, ComplexExpression const& sortFunction) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Sort ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
      return;
    }
    ITTSection section{ITTSortTask()};
    // evaluate the keys
    auto evaluateKeys = [&]() -> Expression {
      if(sortFunction.getHead().getName() == "By") {
        auto const& unevaluatedArgs = sortFunction.getArguments();
        auto argSize = unevaluatedArgs.size();
        ExpressionArguments evaluatedArgs;
        evaluatedArgs.reserve(argSize);
        size_t keyIndex = 0;
        for(auto const& arg : unevaluatedArgs) {
          visit(
              [this, &evaluatedArgs](auto const& val) {
                if constexpr(std::is_same_v<decltype(val), ComplexExpression const&>) {
                  if(val.getHead().getName() == "Minus") {
                    // auto& sortedPartitions =
                    //     cacheOfSortedPartitions.template
                    //     getTyped<CacheOfSortedPartitions<ComplexExpression>>();
                    // return;
                  }
                }
                evaluatedArgs.emplace_back(this->evaluateInternal(val));
              },
              arg);
        }
        if(argSize > 1) {
          return ComplexExpression(sortFunction.getHead(), std::move(evaluatedArgs));
        }
        return std::move(evaluatedArgs[0]);
      }
      if(sortFunction.getHead().getName() != "Function") {
        return this->evaluateInternal(sortFunction);
      }
      auto oldSortFunctionNumArguments = sortFunction.getArguments().size();
      ExpressionArguments sortFunctionArgs;
      sortFunctionArgs.reserve(oldSortFunctionNumArguments + 1);
      for(int i = 0; i < oldSortFunctionNumArguments; ++i) {
        sortFunctionArgs.emplace_back(sortFunction.cloneArgument(i));
      }
      sortFunctionArgs.emplace_back("List"_(partition));
      return this->evaluateInternal(
          ComplexExpression(sortFunction.getHead(), std::move(sortFunctionArgs)));
    };
    auto keys = [&]() {
      auto temporarlyRegisterSymbols =
          RegisterColumnSymbols(partition); // register symbol for each column
      return evaluateKeys();
    }();
    // build the row indexes based on the keys
    sortOp(partition, std::move(keys));
  }

  void sortOp(Table::PartitionPtr const& partition, Expression&& keys) {
    std::visit(
        boss::utilities::overload(
            [this, &partition](auto&& arrayOrConstant) {
              sort(partition, std::forward<decltype(arrayOrConstant)>(arrayOrConstant));
            },
            [this, &partition](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
              throw std::logic_error("unsupported unevaluated key for sorting");
            },
            [this, &partition](ComplexExpression&& expr) { sort(partition, std::move(expr)); },
            [](Table::PartitionVectorPtr&& /*other*/) {
              throw std::logic_error("type not supported as a key for sorting");
            }),
        std::move(keys));
  }

  void close() override {
    ITTSection section{ITTSortTask()};
    auto outputPtr = buildFinalOutput();
    auto& output = *outputPtr;
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    std::for_each(
        std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()),
        [this](auto&& partition) { this->pushUp(std::forward<decltype(partition)>(partition)); });
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
    cacheOfSortedPartitions.clear();
  }

private:
  BagOfContainers<false, CacheOfSortedPartitions<bool>, CacheOfSortedPartitions<int32_t>,
                  CacheOfSortedPartitions<int64_t>, CacheOfSortedPartitions<float_t>,
                  CacheOfSortedPartitions<double_t>, CacheOfSortedPartitions<std::string>,
                  CacheOfSortedPartitions<Symbol> /*, CacheOfSortedPartitions<ComplexExpression>*/,
                  CacheOfSortedPartitions<ValueArrayPtr<bool>>,
                  CacheOfSortedPartitions<ValueArrayPtr<int32_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<int64_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<float_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<double_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<std::string>>,
                  CacheOfSortedPartitions<ValueArrayPtr<Symbol>>>
      cacheOfSortedPartitions;

  template <typename T, typename... OtherArgs>
  void sort(Table::PartitionPtr const& partition, T&& key, OtherArgs&&... otherArgs) {
    auto& sortedPartitions =
        cacheOfSortedPartitions.template getTyped<CacheOfSortedPartitions<T>>();
    sortedPartitions.insert(partition, std::forward<decltype(key)>(key),
                            std::forward<decltype(otherArgs)>(otherArgs)...);
  }

  void sort(Table::PartitionPtr const& partition, ComplexExpression&& multipleKeysExpr) {
    // sort only the first key
    // for the other keys, pass them as additional keys
    ExpressionArguments multipleKeysExprArgs = std::move(multipleKeysExpr).getArguments();
    auto it = std::make_move_iterator(multipleKeysExprArgs.begin());
    auto const itEnd = std::make_move_iterator(multipleKeysExprArgs.end());
    visit(
        [this, &partition, &it, &itEnd](auto&& arg) {
          sort(partition, (Expression&&)std::forward<decltype(arg)>(arg), it + 1, itEnd);
        },
        std::move(*it));
  }

  /* used only for sorting the first subkey */
  template <typename... OtherArgs>
  void sort(Table::PartitionPtr const& partition, Expression&& keyExpr, OtherArgs&&... otherArgs) {
    std::visit(boss::utilities::overload(
                   [this, &partition, &otherArgs...](auto&& arrayOrConstant) {
                     sort(partition, std::forward<decltype(arrayOrConstant)>(arrayOrConstant),
                          std::forward<decltype(otherArgs)>(otherArgs)...);
                   },
                   [this, &partition,
                    &otherArgs...](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
                     throw std::logic_error("unsupported unevaluated key for multi-key sorting");
                   },
                   [](ComplexExpression&& expr) {
                     auto oss = std::ostringstream();
                     oss << expr;
                     throw std::logic_error(
                         "'" + oss.str() +
                         "' complex expression not supported as a key for multi-key sorting");
                   },
                   [](Table::PartitionVectorPtr&& /*other*/) {
                     throw std::logic_error("type not supported as a key for multi-key sorting");
                   }),
               std::move(keyExpr));
  }

  auto buildFinalOutput() {
    Table outputRelation;
    cacheOfSortedPartitions.visit([&outputRelation](auto& sortedPartitions) {
      sortedPartitions.buildOutput(outputRelation);
    });
    return outputRelation.finaliseAndGetPartitions();
  }
};
namespace {
boss::engines::bulk::Engine::Register<Sort> const r103("Sort");    // NOLINT
boss::engines::bulk::Engine::Register<Sort> const r103b("SortBy"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class Top : public boss::engines::bulk::Operator<Top, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr, ComplexExpression, int64_t>>;
  using Operator<Top, ArgumentTypes...>::Operator;

  void operator()(Table::PartitionPtr const& partition, ComplexExpression const& sortFunction,
                  int64_t maxRows) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "compute Top ";
      outputDebugPartition(partition);
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS
    if(!partition || partition->length() == 0) {
      return;
    }
    ITTSection section{ITTTopTask()};
    // evaluate the keys
    auto evaluateKeys = [&]() -> Expression {
      if(sortFunction.getHead().getName() == "By") {
        auto const& unevaluatedArgs = sortFunction.getArguments();
        auto argSize = unevaluatedArgs.size();
        ExpressionArguments evaluatedArgs;
        evaluatedArgs.reserve(argSize);
        for(auto const& arg : unevaluatedArgs) {
          visit([this, &evaluatedArgs](
                    auto const& val) { evaluatedArgs.emplace_back(this->evaluateInternal(val)); },
                arg);
        }
        if(argSize > 1) {
          return ComplexExpression(sortFunction.getHead(), std::move(evaluatedArgs));
        }
        return std::move(evaluatedArgs[0]);
      }
      if(sortFunction.getHead().getName() != "Function") {
        return this->evaluateInternal(sortFunction);
      }
      auto oldSortFunctionNumArguments = sortFunction.getArguments().size();
      ExpressionArguments sortFunctionArgs;
      sortFunctionArgs.reserve(oldSortFunctionNumArguments + 1);
      for(int i = 0; i < oldSortFunctionNumArguments; ++i) {
        sortFunctionArgs.emplace_back(sortFunction.cloneArgument(i));
      }
      sortFunctionArgs.emplace_back("List"_(partition));
      return this->evaluateInternal(
          ComplexExpression(sortFunction.getHead(), std::move(sortFunctionArgs)));
    };
    auto keys = [&]() {
      auto temporarlyRegisterSymbols =
          RegisterColumnSymbols(partition); // register symbol for each column
      return evaluateKeys();
    }();
    // build the row indexes based on the keys
    topOp(partition, std::move(keys), maxRows);
  }

  void topOp(Table::PartitionPtr const& partition, Expression&& keys, int64_t maxRows) {
    std::visit(boss::utilities::overload(
                   [this, &partition, &maxRows](auto&& arrayOrConstant) {
                     sort(partition, std::forward<decltype(arrayOrConstant)>(arrayOrConstant),
                          maxRows);
                   },
                   [this, &partition,
                    &maxRows](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
                     throw std::logic_error("unsupported unevaluated key for sorting (top)");
                   },
                   [this, &partition, &maxRows](ComplexExpression&& expr) {
                     sort(partition, std::move(expr), maxRows);
                   },
                   [](Table::PartitionVectorPtr&& /*other*/) {
                     throw std::logic_error("type not supported as a key for sorting");
                   }),
               std::move(keys));
  }

  void close() override {
    ITTSection section{ITTTopTask()};
    auto outputPtr = buildFinalOutput();
    auto& output = *outputPtr;
    section.pause();
#ifdef DEBUG_OUTPUT_OPS
    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    std::for_each(
        std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()),
        [this](auto&& partition) { this->pushUp(std::forward<decltype(partition)>(partition)); });
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    section.resume();
    cacheOfSortedPartitions.clear();
  }

private:
  BagOfContainers<false, CacheOfSortedPartitions<bool>, CacheOfSortedPartitions<int32_t>,
                  CacheOfSortedPartitions<int64_t>, CacheOfSortedPartitions<float_t>,
                  CacheOfSortedPartitions<double_t>, CacheOfSortedPartitions<std::string>,
                  CacheOfSortedPartitions<Symbol> /*, CacheOfSortedPartitions<ComplexExpression>*/,
                  CacheOfSortedPartitions<ValueArrayPtr<bool>>,
                  CacheOfSortedPartitions<ValueArrayPtr<int32_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<int64_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<float_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<double_t>>,
                  CacheOfSortedPartitions<ValueArrayPtr<std::string>>,
                  CacheOfSortedPartitions<ValueArrayPtr<Symbol>>>
      cacheOfSortedPartitions;

  template <typename T, typename... OtherArgs>
  void sort(Table::PartitionPtr const& partition, T&& key, int64_t maxRows,
            OtherArgs&&... otherArgs) {
    auto& sortedPartitions =
        cacheOfSortedPartitions.template getTyped<CacheOfSortedPartitions<T>>();
    sortedPartitions.setMaxSize(maxRows);
    sortedPartitions.insert(partition, std::forward<decltype(key)>(key),
                            std::forward<decltype(otherArgs)>(otherArgs)...);
  }

  void sort(Table::PartitionPtr const& partition, ComplexExpression&& multipleKeysExpr,
            int64_t maxRows) {
    // sort only the first key
    // for the other keys, pass them as additional keys
    ExpressionArguments multipleKeysExprArgs = std::move(multipleKeysExpr).getArguments();
    auto it = std::make_move_iterator(multipleKeysExprArgs.begin());
    auto itEnd = std::make_move_iterator(multipleKeysExprArgs.end());
    visit(
        [this, &partition, &maxRows, &it, &itEnd](auto&& arg) {
          sort(partition, (Expression&&)std::forward<decltype(arg)>(arg), maxRows, it + 1, itEnd);
        },
        std::move(*it));
  }

  /* used only for sorting the first subkey */
  template <typename... OtherArgs>
  void sort(Table::PartitionPtr const& partition, Expression&& keyExpr, int64_t maxRows,
            OtherArgs&&... otherArgs) {
    std::visit(boss::utilities::overload(
                   [this, &partition, &maxRows, &otherArgs...](auto&& arrayOrConstant) {
                     sort(partition, std::forward<decltype(arrayOrConstant)>(arrayOrConstant),
                          maxRows, std::forward<decltype(otherArgs)>(otherArgs)...);
                   },
                   [this, &partition, &maxRows,
                    &otherArgs...](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
                     throw std::logic_error("unsupported unevaluated key for multi-key sorting");
                   },
                   [](ComplexExpression&& expr) {
                     auto oss = std::ostringstream();
                     oss << expr;
                     throw std::logic_error(
                         "'" + oss.str() +
                         "' complex expression not supported as a key for multi-key sorting");
                   },
                   [](Table::PartitionVectorPtr&& /*other*/) {
                     throw std::logic_error("type not supported as a key for multi-key sorting");
                   }),
               std::move(keyExpr));
  }

  auto buildFinalOutput() {
    Table outputRelation;
    cacheOfSortedPartitions.visit([&outputRelation](auto& sortedPartitions) {
      sortedPartitions.buildOutput(outputRelation);
    });
    return outputRelation.finaliseAndGetPartitions();
  }
};
namespace {
boss::engines::bulk::Engine::Register<Top> const r104("Top"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class Join : public boss::engines::bulk::Operator<Join, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<Table::PartitionPtr, Table::PartitionPtr, ComplexExpression>,
              tuple<Table::PartitionPtr, Table::PartitionPtr, ComplexExpression, int64_t>>;
  using Operator<Join, ArgumentTypes...>::Operator;
  bool checkArguments(Table::PartitionPtr const& /*leftSidePartition*/,
                      Table::PartitionPtr const& /*rightSidePartition*/,
                      ComplexExpression const& predicate, int64_t /*leftSideCardinality*/ = 0) {
    return predicate.getHead().getName() == "Where" || predicate.getHead().getName() == "Function";
  }
  void operator()(Table::PartitionPtr const& leftSidePartition,
                  Table::PartitionPtr const& rightSidePartition, ComplexExpression const& predicate,
                  int64_t leftSideCardinality = 0) {
    if(predicate.getHead().getName() == "Where") {
      operator()(leftSidePartition, rightSidePartition,
                 get<ComplexExpression>(predicate.cloneArgument(0)), "List"_(),
                 leftSideCardinality);
      return;
    }
    // "Function"_ with only body
    if(predicate.getArguments().size() < 2) {
      operator()(leftSidePartition, rightSidePartition,
                 get<ComplexExpression>(predicate.cloneArgument(0)), "List"_(),
                 leftSideCardinality);
      return;
    }
    // "Function"_ with argument list + body
    operator()(leftSidePartition, rightSidePartition,
               get<ComplexExpression>(predicate.cloneArgument(1)), predicate.cloneArgument(0),
               leftSideCardinality);
  }
  void operator()(Table::PartitionPtr const& leftSidePartition,
                  Table::PartitionPtr const& rightSidePartition, ComplexExpression&& predicate,
                  Expression&& predicateParameters, int64_t leftSideCardinality = 0) {
#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "left: ";
      outputDebugPartition(leftSidePartition);
      std::cerr << " right: ";
      outputDebugPartition(rightSidePartition);
      std::cerr << " card: " << leftSideCardinality;
      std::cerr << std::endl;
    }
#endif // DEBUG_OUTPUT_OPS

    if((!leftSidePartition /*|| leftSidePartition->length() == 0*/) &&
       (!rightSidePartition /*|| rightSidePartition->length() == 0*/)) {
#ifdef DEBUG_OUTPUT_OPS
      ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      this->pushUp(Table::PartitionPtr());
#ifdef DEBUG_OUTPUT_OPS
      --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      return;
    }

    ITTSection section{ITTJoinTask()};
    ExpressionArguments predicateArgs = predicate.getArguments();
    auto predicateArgsSize = predicateArgs.size();
    if(predicate.getHead().getName() == "And") {
      // re-shape the expression to fit the expectation for a multi-key join
      ExpressionArguments leftArgs;
      ExpressionArguments rightArgs;
      leftArgs.reserve(predicateArgsSize);
      rightArgs.reserve(predicateArgsSize);
      bool canHandle = true;
      for(auto&& predicateArg : predicateArgs) {
        if(holds_alternative<ComplexExpression>(predicateArg)) {
          auto&& expr = get<ComplexExpression>(std::move(predicateArg));
          auto exprArgs = expr.getArguments();
          if(expr.getHead().getName() == "Equal" && exprArgs.size() == 2) {
            leftArgs.emplace_back(std::move(exprArgs.at(0)));
            rightArgs.emplace_back(std::move(exprArgs.at(1)));
            continue;
          }
        }
        canHandle = false;
        break;
      }
      if(canHandle) {
        Expression leftExpr = ComplexExpression("List"_, std::move(leftArgs));
        Expression rightExpr = ComplexExpression("List"_, std::move(rightArgs));
        join(leftSidePartition, rightSidePartition, std::move(leftExpr), std::move(rightExpr),
             std::move(predicateParameters), leftSideCardinality);
        return;
      }
    } else if(predicate.getHead().getName() == "Equal") {
      if(predicateArgsSize == 2) {
        auto leftExpr = std::move(predicateArgs[0]);
        auto rightExpr = std::move(predicateArgs[1]);
        join(leftSidePartition, rightSidePartition, std::move(leftExpr), std::move(rightExpr),
             std::move(predicateParameters), leftSideCardinality);
        return;
      }
    }
    auto oss = std::ostringstream();
    oss << predicate.getHead().getName();
    throw std::logic_error("join predicate operator '" + oss.str() + "' not supported");
  }

  void close() override {
    ITTSection section{ITTJoinTask()};
    hashedPartitions.clear();
    hashmaps.clear();
    outputRelation.clear();
  }

private:
  // assuming *mostly* unique values on left side joins
  // no indirection for the first element, execution overhead only if > 1 element
  // (still there is always sizeof(std::vector<T>) overhead for the storage)
  template <typename T> class IntrusiveVector {
    T value;
    std::vector<T> otherValues;

  public:
    using value_type = T;
    explicit IntrusiveVector(T&& val) : value(std::move(val)) {}
    explicit IntrusiveVector(T const& val) : value(val) {}
    template <typename... Args>
    explicit IntrusiveVector(Args&&... args) : value(std::forward<decltype(args)>(args)...) {}
    void push_back(T&& val) { otherValues.push_back(std::move(val)); }
    void push_back(T const& val) { otherValues.push_back(val); }
    template <typename... Args> void emplace_back(Args&&... args) {
      otherValues.emplace_back(std::forward<decltype(args)>(args)...);
    }
    size_t size() const { return otherValues.size() + 1; }
    T& at(size_t index) { return index == 0 ? value : otherValues.at(index - 1); }
    T const& at(size_t index) const { return index == 0 ? value : otherValues.at(index - 1); }

    template <typename ConstOrNonConstContainerT>
    class Iterator : public std::iterator<std::forward_iterator_tag, T> {
    private:
      ConstOrNonConstContainerT& container;
      size_t index;

    public:
      Iterator(ConstOrNonConstContainerT& container, size_t index)
          : container(container), index(index) {}
      Iterator& operator++() {
        ++index;
        return *this;
      }
      Iterator operator++(int) {
        auto before = *this;
        ++*this;
        return before;
      }
      auto& operator*() { return container.at(index); }
      bool operator==(Iterator const& other) const { return index == other.index; }
      bool operator!=(Iterator const& other) const { return index != other.index; }
      bool operator<(Iterator const& other) const { return index < other.index; }
      bool operator>(Iterator const& other) const { return index > other.index; }
      Iterator& operator=(Iterator const& other) = default;
      Iterator& operator=(Iterator&& other) noexcept = default;
      Iterator(Iterator const& other) = default;
      Iterator(Iterator&& other) noexcept = default;
      ~Iterator() = default;
    };

    auto begin() const { return Iterator<IntrusiveVector const>(*this, 0); }
    auto end() const { return Iterator<IntrusiveVector const>(*this, size()); }

    auto begin() { return Iterator<IntrusiveVector>(*this, 0); }
    auto end() { return Iterator<IntrusiveVector>(*this, size()); }
  };

  std::vector<Table::PartitionPtr> hashedPartitions;
  ExpressionHashMap<false, IntrusiveVector<std::pair<size_t, int64_t>>> hashmaps;

  std::vector<std::vector<int64_t>> leftIndicesPerPartition;
  std::vector<std::vector<int64_t>> rightIndicesPerPartition;

  Table outputRelation;

  void join(Table::PartitionPtr const& leftSidePartition,
            Table::PartitionPtr const& rightSidePartition, Expression&& predicateLeftSide,
            Expression&& predicateRightSide, Expression&& predicateParameters,
            int64_t leftSideCardinality) {
    // equi-join
    if(leftSidePartition && leftSidePartition->length() > 0) {
      // evaluate the keys
      ExpressionArguments leftArgs;
      leftArgs.reserve(3);
      leftArgs.emplace_back(std::move(predicateParameters));
      leftArgs.emplace_back(std::move(predicateLeftSide));
      leftArgs.emplace_back("List"_(leftSidePartition));
      auto keys = [&]() {
        auto temporarlyRegisterSymbols =
            RegisterColumnSymbols(leftSidePartition); // register symbol for each column
        return this->evaluateInternal(ComplexExpression("Function"_, std::move(leftArgs)));
      }();
      // add to hashmap
      hash(leftSidePartition, std::move(keys), leftSideCardinality);
    } else if(rightSidePartition && rightSidePartition->length() > 0) {
      // evaluate the keys
      ExpressionArguments rightArgs;
      rightArgs.reserve(3);
      rightArgs.emplace_back(std::move(predicateParameters));
      rightArgs.emplace_back(std::move(predicateRightSide));
      rightArgs.emplace_back("List"_(rightSidePartition));
      auto keys = [&]() {
        auto temporarlyRegisterSymbols =
            RegisterColumnSymbols(rightSidePartition); // register symbol for each column
        return this->evaluateInternal(ComplexExpression("Function"_, std::move(rightArgs)));
      }();
      // probe from hashmap and push the output rows
      probe(rightSidePartition, std::move(keys));
    }
  }

  void hash(Table::PartitionPtr const& partitionPtr, Expression&& keys,
            int64_t leftSideCardinality) {
    std::visit(
        boss::utilities::overload(
            [this, &partitionPtr, &leftSideCardinality](auto&& arrayOrConstant) {
              hash(partitionPtr, std::forward<decltype(arrayOrConstant)>(arrayOrConstant),
                   leftSideCardinality);
            },
            [](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
              throw std::logic_error("unsupported unevaluated key for join-hashing");
            },
            [this, &partitionPtr, &leftSideCardinality](ComplexExpression&& expr) {
              if(expr.getHead().getName() != "List") {
                auto oss = std::ostringstream();
                oss << expr;
                throw std::logic_error(
                    "'" + oss.str() +
                    "' complex expression not supported as a key for join-hashing");
                return;
              }
              if(expr.getArguments().size() > MAX_MULTIPLE_KEY_SIZE) {
                throw std::logic_error(
                    std::to_string(expr.getArguments().size()) +
                    " keys provided for multi-key join-hashing but the maximum supported is " +
                    std::to_string(MAX_MULTIPLE_KEY_SIZE));
                return;
              }
              hash(partitionPtr, std::move(expr), leftSideCardinality);
            },
            [](Table::PartitionVectorPtr const& /*other*/) {
              throw std::logic_error("type not supported as a key for join-hashing");
            }),
        std::move(keys));
  }

  template <typename T>
  void hash(Table::PartitionPtr const& /*partitionPtr*/, T&& /*key*/,
            int64_t /*leftSideCardinality*/) {
    // not making sense to implement a constant key for the left relation of a join
  }
  template <typename T>
  void hash(Table::PartitionPtr const& partitionPtr, ValueArrayPtr<T>&& keysPtr,
            int64_t leftSideCardinality) {
    auto& hashmap = hashmaps.template getTyped<T>();
    if(leftSideCardinality > 0 && hashedPartitions.empty()) {
      hashmap.reserve(leftSideCardinality);
      auto numPartitions = 1 + leftSideCardinality / bulk::Properties::getMicroBatchesMaxSize();
      hashedPartitions.reserve(numPartitions);
    }
    auto partitionIndex = hashedPartitions.size();
    auto length = partitionPtr->length();
    hashedPartitions.push_back(partitionPtr);
    auto const& keys = *keysPtr;
    for(int64_t index = 0; index < length; ++index) {
      auto const& key = keys[index];
      auto [it, inserted] = hashmap.try_emplace(key, partitionIndex, index);
      if(!inserted) {
        // rare case (to avoid) where keys are not unique
        // insert to the intrusive linked list
        auto& value = it->second;
        value.emplace_back(partitionIndex, index);
      }
    }
  }
  void hash(Table::PartitionPtr const& partitionPtr, ComplexExpression&& multipleKeysExpr,
            int64_t leftSideCardinality) {
    auto& hashmap = hashmaps.template getTyped<MultipleKeys>();
    if(leftSideCardinality > 0 && hashedPartitions.empty()) {
      hashmap.reserve(leftSideCardinality);
      auto numPartitions = leftSideCardinality / bulk::Properties::getMicroBatchesMaxSize();
      hashedPartitions.reserve(numPartitions);
    }
    auto partitionIndex = hashedPartitions.size();
    hashedPartitions.push_back(partitionPtr);
    auto multipleKeysList = hashmap.prepareMultiplekeysListForHashing(multipleKeysExpr);
    for(int64_t index = 0; index < multipleKeysList.size(); ++index) {
      auto multipleKey = std::move(multipleKeysList[index]);
      auto [it, inserted] = hashmap.try_emplace(std::move(multipleKey), partitionIndex, index);
      if(!inserted) {
        // rare case (to avoid) where keys are not unique
        // insert to the intrusive linked list
        auto& value = it->second;
        value.emplace_back(partitionIndex, index);
      }
    }
  }

  void probe(Table::PartitionPtr const& partitionPtr, Expression&& keys) {
    std::visit(
        boss::utilities::overload(
            [this, &partitionPtr](auto&& arrayOrConstant) {
              probe(partitionPtr, std::forward<decltype(arrayOrConstant)>(arrayOrConstant));
            },
            [this, &partitionPtr](std::shared_ptr<ComplexExpressionArray>&& /*complexArray*/) {
              throw std::logic_error("unsupported unevaluated key for join-probing");
            },
            [this, &partitionPtr](ComplexExpression&& expr) {
              if(expr.getHead().getName() != "List") {
                auto oss = std::ostringstream();
                oss << expr;
                throw std::logic_error(
                    "'" + oss.str() +
                    "' complex expression not supported as a key for join-probing");
                return;
              }
              if(expr.getArguments().size() > MAX_MULTIPLE_KEY_SIZE) {
                throw std::logic_error(
                    std::to_string(expr.getArguments().size()) +
                    " keys provided for multi-key join-probing but the maximum supported is " +
                    std::to_string(MAX_MULTIPLE_KEY_SIZE));
                return;
              }
              probe(partitionPtr, std::move(expr));
            },
            [](Table::PartitionVectorPtr&& /*other*/) {
              throw std::logic_error("type not supported as a key for join-probing");
            }),
        std::move(keys));
  }
  template <typename T> void probe(Table::PartitionPtr const& /*partitionPtr*/, T&& /*key*/) {
    // not making sense to implement a constant key for the right relation of a join
  }
  template <typename T>
  void probe(Table::PartitionPtr const& partitionPtr, ValueArrayPtr<T>&& keysPtr) {
    probe<T>(partitionPtr, *keysPtr, keysPtr->length());
  }
  void probe(Table::PartitionPtr const& partitionPtr, ComplexExpression&& multipleKeysExpr) {
    auto& hashmap = hashmaps.template getTyped<MultipleKeys>();
    auto multipleKeysList = hashmap.prepareMultiplekeysListForHashing(multipleKeysExpr);
    auto size = multipleKeysList.size();
    probe<MultipleKeys>(partitionPtr, std::move(multipleKeysList), size);
  }

  template <typename KeyType, typename KeyArray>
  void probe(Table::PartitionPtr const& partitionPtr, KeyArray const& keys, int64_t numKeys) {
    auto const& partition = *partitionPtr;
    auto outputRows = [&](auto const& leftSideIndices, auto const& rightSideIndices,
                          auto leftSidePartitionIndex) {
      auto const& head = partition.getHead();
      auto const& leftSidePartitionPtr = hashedPartitions[leftSidePartitionIndex];
      auto const& leftSidePartition = *leftSidePartitionPtr;
      // prepare the indices
      auto numRows = leftSideIndices.size();
      // join columns (to initialise the empty arrays or find existing matching array types)
      auto const& leftColumns = leftSidePartition.fields();
      auto const& rightColumns = partition.fields();
      auto mergedColumns = arrow::ArrayVector();
      mergedColumns.reserve(leftColumns.size() + rightColumns.size());
      mergedColumns.insert(mergedColumns.end(), leftColumns.begin(), leftColumns.end());
      mergedColumns.insert(mergedColumns.end(), rightColumns.begin(), rightColumns.end());
      // join the rows
      auto outputPartitionIndex =
          outputRelation.findMatchingPartitionIndex(head, mergedColumns, numRows);
      if(outputPartitionIndex < outputRelation.numPartitions()) {
        // add to existing partition
        auto builder = outputRelation.getPartitionBuilder(outputPartitionIndex);
        builder->joinRowsInIndexedOrder(leftSidePartition, leftSideIndices, partition,
                                        rightSideIndices, outputRelation.getMaxGlobalIndex());
        outputRelation.setPartition(outputPartitionIndex, std::move(builder));
      } else {
        // create new partition
        int64_t expectedFullSizePartition =
            std::min(leftSidePartition.length(), partition.length());
        auto builder = std::make_shared<ComplexExpressionArrayBuilder>(
            leftSidePartitionPtr, partitionPtr, expectedFullSizePartition);
        builder->joinRowsInIndexedOrder(leftSidePartition, leftSideIndices, partition,
                                        rightSideIndices, outputRelation.getMaxGlobalIndex());
        outputRelation.addPartition(std::move(builder));
      }
      outputRelation.getMaxGlobalIndex() += numRows;
    };
    if(leftIndicesPerPartition.size() < hashedPartitions.size()) {
      leftIndicesPerPartition.resize(hashedPartitions.size());
      rightIndicesPerPartition.resize(hashedPartitions.size());
    }
    auto& hashmap = hashmaps.template getTyped<KeyType>();
    for(int64_t keyIndex = 0; keyIndex < numKeys; ++keyIndex) {
      auto const& key = keys[keyIndex];
      auto foundIt = hashmap.find(key);
      if(foundIt != hashmap.end()) {
        auto const& hashedPositions = foundIt->second;
        for(auto it = hashedPositions.begin(); it != hashedPositions.end(); ++it) {
          auto const& position = *it;
          auto const& partitionIndex = position.first;
          leftIndicesPerPartition[partitionIndex].emplace_back(position.second);
          rightIndicesPerPartition[partitionIndex].emplace_back(keyIndex);
        }
      }
    }
    for(int i = 0; i < hashedPartitions.size(); ++i) {
      if(leftIndicesPerPartition[i].empty()) {
        continue;
      }
      outputRows(leftIndicesPerPartition[i], rightIndicesPerPartition[i], i);
      leftIndicesPerPartition[i].clear();
      rightIndicesPerPartition[i].clear();
    }

    auto outputPtr = outputRelation.finaliseAndGetPartitions();
    auto& output = *outputPtr;

#ifdef DEBUG_OUTPUT_OPS
    if(Properties::debugOutputRelationalOps()) {
      for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
        std::cerr << "  ";
      }
      std::cerr << "merged: ";
      outputDebugPartitions(outputPtr);
      std::cerr << std::endl;
    }

    ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
    for(auto&& partition : output) {
      this->pushUp(std::move(partition));
    }
#ifdef DEBUG_OUTPUT_OPS
    --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS

    outputRelation.clear();
  }
};
namespace {
boss::engines::bulk::Engine::Register<Join> const r105("Join"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class Count : public boss::engines::bulk::Operator<Count, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<bool>>, tuple<ValueArrayPtr<int32_t>>,
                                 tuple<ValueArrayPtr<int64_t>>, tuple<ValueArrayPtr<float_t>>,
                                 tuple<ValueArrayPtr<double_t>>, tuple<ValueArrayPtr<string>>,
                                 tuple<ValueArrayPtr<Symbol>>, tuple<Table::PartitionPtr>>;
  using Operator<Count, ArgumentTypes...>::Operator;
  template <typename ArrayType> void operator()(ArrayType const& arrayPtr) {
    aggregate += arrayPtr->length();
  }
  void close() override {
    this->pushUp(aggregate);
    aggregate = 0;
  }

private:
  int64_t aggregate = 0;
};
namespace {
boss::engines::bulk::Engine::Register<Count> const r110("Count"); // NOLINT
}

template <typename DefaultType, typename... ArgumentTypes> struct FirstArgument {
  using Type = std::tuple_element_t<0, std::tuple<ArgumentTypes...>>;
};
template <typename DefaultType> struct FirstArgument<DefaultType> {
  using Type = DefaultType;
};

template <typename... ArgumentTypes>
class Sum : public boss::engines::bulk::Operator<Sum, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<int32_t>>, tuple<ValueArrayPtr<int64_t>>,
                                 tuple<ValueArrayPtr<float_t>>, tuple<ValueArrayPtr<double_t>>>;
  using Operator<Sum, ArgumentTypes...>::Operator;
  using ArrayType =
      typename FirstArgument<std::tuple_element_t<0, std::variant_alternative_t<0, ArgumentTypesT>>,
                             ArgumentTypes...>::Type;
  using ElementType = typename ArrayType::element_type::ElementType;
  void operator()(ArrayType const& valueArrayPtr) {
    auto const& valueArray = *valueArrayPtr;
    auto length = valueArray.length();
    for(int64_t index = 0; index < length; ++index) {
      aggregate += valueArray.Value(index);
    }
  }
  void close() override {
    this->pushUp(aggregate);
    aggregate = 0;
  }

private:
  ElementType aggregate = 0;
};
namespace {
boss::engines::bulk::Engine::Register<Sum> const r111("Sum"); // NOLINT
}

} // namespace boss::engines::bulk
