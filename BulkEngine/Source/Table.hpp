#pragma once
#include "ArrowExtensions/ComplexExpressionArray.hpp"
#include "ArrowExtensions/ComplexExpressionBuilder.hpp"
#include "BulkExpression.hpp"
#include "BulkProperties.hpp"
#include "SymbolRegistry.hpp"

#include <mutex>
#include <set>
#include <unordered_map>

namespace boss::engines::bulk {

class Table {
public:
  using Partition = ComplexExpressionArray;
  using PartitionPtr = std::shared_ptr<ComplexExpressionArray>;
  using PartitionVector = std::vector<PartitionPtr>;
  using PartitionVectorPtr = std::shared_ptr<PartitionVector>;

  using PartitionBuilder = ComplexExpressionArrayBuilder;
  using PartitionBuilderPtr = std::shared_ptr<ComplexExpressionArrayBuilder>;
  using PartitionBuilderVector = std::vector<PartitionBuilderPtr>;

  Table() : partitions(new PartitionVector()) {}
  explicit Table(arrow::FieldVector&& schema)
      : schema(std::move(schema)), partitions(new PartitionVector()) {}

  ~Table() {
    for(auto& partitionBuilder : partitionBuilders) {
      if(partitionBuilder) {
        partitionBuilder->setInDestruction(); // from now on, skip unnecessary caching
      }
    }
  }

  Table(Table&) = delete;
  Table& operator=(Table&) = delete;
  Table(Table&&) = delete;
  Table& operator=(Table&&) = delete;

  arrow::FieldVector const& getSchema() const { return schema; }
  arrow::FieldVector& getSchema() { return schema; }

  void finalise() {
    for(auto& partitionBuilder : partitionBuilders) {
      if(!partitionBuilder) {
        continue;
      }
      addPartition((Table::PartitionPtr)(*partitionBuilder));
    }
    partitionBuilders.clear();
    missingHashToBuilders.clear();
    if(bulk::Properties::getEnableOrderPreservationCache()) {
      buildPartitionIndexes();
    }
  }

  size_t finalise(size_t index) {
    if(index < partitionBuilders.size()) {
      auto partition = (Table::PartitionPtr)(*partitionBuilders[index]);
      removePartition(index);
      index = addPartition(std::move(partition));
    }
    if(bulk::Properties::getEnableOrderPreservationCache()) {
      buildPartitionIndexes();
    }
    return index;
  }

  void buildPartitionIndexes() {
    if(partitions->size() <= numFinalisedPartitions) {
      return;
    }
    std::shared_ptr<ValueBuilder<int64_t>> tablePartitionIndexesBuilder;
    std::shared_ptr<ValueBuilder<int64_t>> tablePartitionLocalIndexesBuilder;
    if(*tablePartitionIndexes && *tablePartitionLocalIndexes) {
      // convert the existing indexes back to builders
      tablePartitionIndexesBuilder =
          std::make_shared<ValueBuilder<int64_t>>(*tablePartitionIndexes);
      tablePartitionLocalIndexesBuilder =
          std::make_shared<ValueBuilder<int64_t>>(*tablePartitionLocalIndexes);
    } else {
      if(*tablePartitionIndexes || *tablePartitionLocalIndexes) {
        throw std::runtime_error("tablePartitionIndexes or tablePartitionLocalIndexes has "
                                 "not been initialised properly.");
      }
      // create new indexes
      tablePartitionIndexesBuilder = std::make_shared<ValueBuilder<int64_t>>(0);
      tablePartitionLocalIndexesBuilder = std::make_shared<ValueBuilder<int64_t>>(0);
    }
    auto* partitionIndexes = &((*tablePartitionIndexesBuilder)[0]);
    auto* partitionLocalIndexes = &((*tablePartitionLocalIndexesBuilder)[0]);
    auto indexesLength = tablePartitionIndexesBuilder->length();
    auto resizeIndexes = [&tablePartitionIndexesBuilder, &tablePartitionLocalIndexesBuilder,
                          &partitionIndexes, &partitionLocalIndexes, &indexesLength](auto newSize) {
      auto numIndexesToAdd = newSize - indexesLength;
      auto status = tablePartitionIndexesBuilder->AppendEmptyValues(numIndexesToAdd);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
      status = tablePartitionLocalIndexesBuilder->AppendEmptyValues(numIndexesToAdd);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
      partitionIndexes = &((*tablePartitionIndexesBuilder)[0]);
      partitionLocalIndexes = &((*tablePartitionLocalIndexesBuilder)[0]);
      for(int64_t i = indexesLength; i < newSize; ++i) {
        partitionIndexes[i] = -1; // -1 for filtered-out rows
      }
      indexesLength = newSize;
    };
    for(int64_t partitionIndex = numFinalisedPartitions; partitionIndex < partitions->size();
        ++partitionIndex) {
      auto const& globalIndexes = (*partitions)[partitionIndex]->getGlobalIndexes();
      if(!globalIndexes) {
        continue;
      }
      auto const* globalIndexesValues = globalIndexes->rawData();
      auto numGlobalIndexes = globalIndexes->length();
      auto realPartitionIndex = partitionIndex - numFinalisedPartitions + nextStartPartitionIndex;
      for(int64_t i = numGlobalIndexes - 1; i >= 0; --i) {
        auto const& globalIndex = globalIndexesValues[i];
        if(globalIndex >= indexesLength) {
          resizeIndexes(globalIndex + 1);
        }
        partitionIndexes[globalIndex] = realPartitionIndex;
        partitionLocalIndexes[globalIndex] = i;
      }
    }
    *tablePartitionIndexes = (ValueArrayPtr<int64_t>)(*tablePartitionIndexesBuilder);
    *tablePartitionLocalIndexes = (ValueArrayPtr<int64_t>)(*tablePartitionLocalIndexesBuilder);
    for(auto partitionIt = partitions->begin() + numFinalisedPartitions;
        partitionIt != partitions->end(); ++partitionIt) {
      auto& partition = *partitionIt;
      partition->setTablePartitionIndexes(tablePartitionIndexes);
      partition->setTablePartitionLocalIndexes(tablePartitionLocalIndexes);
    }
    nextStartPartitionIndex += partitions->size() - numFinalisedPartitions;
    numFinalisedPartitions = partitions->size();
  }

  PartitionVectorPtr finaliseAndGetPartitions(bool threadSafe = false) {
    auto lock = threadSafe ? std::unique_lock(m) : std::unique_lock<std::mutex>();
    finalise();
    if(partitions->empty()) {
      // return a dummy partition vector (to represent an empty table)
      return PartitionVectorPtr(new PartitionVector{getPartition(0)});
    }
    return partitions;
  }

  PartitionPtr finaliseAndGetPartition(size_t index) {
    if(bulk::Properties::getForceToPreserveInsertionOrder()) {
      return (*finaliseAndGetPartitions())[index];
    }
    auto newIndex = finalise(index);
    return getPartition(newIndex);
  }

  PartitionPtr getPartition(size_t index) const {
    if(partitionBuilders.empty() && partitions->empty()) {
      // return a dummy partition (to represent an empty table)
      auto builder = ComplexExpressionArrayBuilder("List"_, schema, 0);
      return (Table::PartitionPtr)builder;
    }
    if(index < partitionBuilders.size()) {
      throw std::runtime_error("cannot call getPartition() to retrieve a builder. "
                               "only finaliseAndGetPartitions() can do.");
    }
    return (*partitions)[index - partitionBuilders.size()]; // partitions come after builders
  }

  PartitionBuilderPtr getPartitionBuilder(size_t index) const {
    if(partitionBuilders.empty() && partitions->empty()) {
      // return a dummy partition (to represent an empty table)
      return std::make_shared<ComplexExpressionArrayBuilder>("List"_, schema, 0);
    }
    if(index < partitionBuilders.size()) {
      return partitionBuilders[index];
    }
    return std::make_shared<ComplexExpressionArrayBuilder>(
        (*partitions)[index - partitionBuilders.size()]); // partitions come after builders
  }

  size_t numPartitions() { return partitionBuilders.size() + partitions->size(); }

  size_t addPartition(PartitionPtr&& partition) {
    partitions->push_back(std::move(partition));
    return numPartitions() - 1;
  }

  size_t addPartition(PartitionPtr const& partition) {
    partitions->push_back(partition);
    return numPartitions() - 1;
  }

  size_t setPartition(size_t index, PartitionPtr&& partition) {
    if(index < partitionBuilders.size()) {
      // it was a builder converted to an array
      if(!bulk::Properties::getForceToPreserveInsertionOrder() || index == 0) {
        // remove it from the builder list and add an array instead
        // (assuming the partitionBuilders mostly staying small for the overhead to be negligible)
        removePartition(index);
        return addPartition(std::move(partition));
      }
      // if order matters, just convert it back to a builder
      return setPartition(index, std::make_shared<ComplexExpressionArrayBuilder>(partition));
    }
    auto partitionIndex =
        index - partitionBuilders.size(); // partitions come after partitionBuilders
    (*partitions)[partitionIndex] = std::move(partition);
    return index;
  }

  void removePartition(size_t index) {
    if(index < partitionBuilders.size()) {
      missingHashToBuilders[partitionBuilders[index]->getMissingHash()].erase(index);
      partitionBuilders[index] = nullptr; // just empty it so indexes are not invalidated
      // clean-up null builders if possible (perf optimization for when iterating on the builders)
      while(!partitionBuilders.empty() && !partitionBuilders.back()) {
        partitionBuilders.pop_back();
      }
      return;
    }
    auto partitionIndex = index - partitionBuilders.size(); // partitions come after builders
    partitions->erase(partitions->begin() + partitionIndex);
    if(partitionIndex < numFinalisedPartitions) {
      numFinalisedPartitions--;
    }
  }

  void clear() {
    partitionBuilders.clear();
    missingHashToBuilders.clear();
    partitions->clear();
    numFinalisedPartitions = 0;
  }

  size_t addPartition(PartitionBuilderPtr&& partitionBuilder) {
    if(partitionBuilder->length() >= bulk::Properties::getMicroBatchesMaxSize()) {
      // convert to an array since it is full
      return addPartition((PartitionPtr)*partitionBuilder);
    }
    auto index = partitionBuilders.size();
    missingHashToBuilders[partitionBuilder->getMissingHash()].insert(index);
    partitionBuilders.push_back(std::move(partitionBuilder));
    return index;
  }

  size_t setPartition(size_t index, PartitionBuilderPtr&& partitionBuilder) {
    if(!bulk::Properties::getForceToPreserveInsertionOrder() || index == 0) {
      if(index >= partitionBuilders.size()) {
        // it was an array converted to a builder: remove and add as a builder
        removePartition(index);
        return addPartition(std::move(partitionBuilder));
      }
      if(partitionBuilder->length() >= bulk::Properties::getMicroBatchesMaxSize()) {
        // builder is full: convert to an array (remove and add)
        removePartition(index);
        return addPartition((PartitionPtr)*partitionBuilder);
      }
    }
    if(index >= partitionBuilders.size()) {
      // it was an array converted to a builder,
      // convert it back to an array
      return setPartition(index, (PartitionPtr)*partitionBuilder);
    }
    if(partitionBuilders[index]->getMissingHash() != partitionBuilder->getMissingHash()) {
      missingHashToBuilders[partitionBuilders[index]->getMissingHash()].erase(index);
      missingHashToBuilders[partitionBuilder->getMissingHash()].insert(index);
    }
    partitionBuilders[index] = std::move(partitionBuilder);
    return index;
  }

  bool hasEnoughSpaceInPartition(size_t index, int64_t neededLength) const {
    if(index < partitionBuilders.size()) {
      return hasEnoughSpaceInPartition(partitionBuilders[index], neededLength);
    }
    return hasEnoughSpaceInPartition((*partitions)[index - partitionBuilders.size()], neededLength);
  }

  size_t findMatchingPartitionIndex(Symbol const& head, arrow::ArrayVector const& columns) const {
    auto neededLength = !columns.empty()
                            ? columns[0]->length() // assuming all columns having same size
                            : 0;
    return findMatchingPartitionIndex(head, columns, neededLength);
  }

  size_t findMatchingPartitionIndex(Symbol const& head, arrow::ArrayVector const& columns,
                                    int64_t neededLength) const {
    uint64_t missingHash = ComplexExpressionArray::computeMissingHash(columns);
    return findMatchingPartitionIndex<arrow::ArrayVector, Symbol>(columns, head, neededLength,
                                                                  missingHash);
  }

  size_t findMatchingPartitionIndex(PartitionPtr const& partition) const {
    return findMatchingPartitionIndex(partition, partition->length());
  }

  size_t findMatchingPartitionIndex(PartitionPtr const& partition, int64_t neededLength) const {
    auto const& fields = partition->fields();
    auto const& head = partition->getHead();
    uint64_t missingHash = partition->getMissingHash();
    return findMatchingPartitionIndex<std::vector<std::shared_ptr<arrow::Array>>, Symbol>(
        fields, head, neededLength, missingHash);
  }

  size_t findMatchingPartitionIndex(ComplexExpression const& expression) const {
    auto args = expression.getArguments();
    auto neededLength =
        args.empty()
            ? (int64_t)1
            : visit(
                  [](auto const& v) -> int64_t {
                    using ValueType = std::decay_t<decltype(v)>;
                    if constexpr(std::is_convertible_v<ValueType, std::shared_ptr<arrow::Array>>) {
                      return v->length();
                    } else {
                      return 1;
                    }
                  },
                  args.at(0));
    std::bitset<64> missingBitset;
    for(int idx = 0; idx < args.size(); ++idx) {
      visit(boss::utilities::overload(
                [](bool /*v*/) {}, [](int32_t /*v*/) {}, [](int64_t /*v*/) {}, [](float_t /*v*/) {},
                [](double_t /*v*/) {}, [](std::string const& /*v*/) {},
                [](std::shared_ptr<std::vector<std::shared_ptr<ComplexExpressionArray>>> const&
                   /*partitions*/) {},
                [&missingBitset, &idx](Symbol const& /*s*/) { missingBitset[idx] = true; },
                [&missingBitset, &idx](ComplexExpression const& e) { missingBitset[idx] = true; },
                [&missingBitset, &idx](auto const& arrayPtr) {
                  if(arrayPtr->type_id() == arrow::Type::EXTENSION) {
                    missingBitset[idx] = true;
                  }
                }),
            args[idx]);
    }
    uint64_t missingHash = missingBitset.to_ullong();
    return findMatchingPartitionIndex<ComplexExpression>(expression, neededLength, missingHash);
  }

  auto& getMaxGlobalIndex() { return maxGlobalIndex; }
  auto const& getMaxGlobalIndex() const { return maxGlobalIndex; }

private:
  std::mutex m;

  arrow::FieldVector schema;
  PartitionVectorPtr partitions;
  PartitionBuilderVector partitionBuilders;
  std::unordered_map<uint64_t, std::set<size_t>> missingHashToBuilders;

  int64_t maxGlobalIndex = 0;
  int64_t numFinalisedPartitions = 0;
  int64_t nextStartPartitionIndex = 0;
  std::shared_ptr<ValueArrayPtr<int64_t>> tablePartitionIndexes{new ValueArrayPtr<int64_t>()};
  std::shared_ptr<ValueArrayPtr<int64_t>> tablePartitionLocalIndexes{new ValueArrayPtr<int64_t>()};

  template <typename PartitionArrayOrBuilderPtr>
  bool hasEnoughSpaceInPartition(PartitionArrayOrBuilderPtr& partition,
                                 int64_t neededLength) const {
    return partition->length() + neededLength <= bulk::Properties::getMicroBatchesMaxSize();
  }

  template <typename... Args>
  size_t findMatchingPartitionIndex(Args const&... args, int64_t neededLength,
                                    uint64_t missingHash) const {
    if(bulk::Properties::getDisableExpressionPartitioning()) {
      // allow matching only the atomic types of the rows
      // so expressions don't get partitioned (but atomic types do)
      if(neededLength == 1) {
        // a bit hacky: we assume that rows with missing values are inserted individually
        // whereas purely atomic values are inserted in batches
        return size_t(-1);
      }
    }
    auto cachedBuildersIt = missingHashToBuilders.find(missingHash);
    if(cachedBuildersIt == missingHashToBuilders.end()) {
      return size_t(-1);
    }
    auto const& cachedBuilders = cachedBuildersIt->second;
    auto indexIt =
        cachedBuilders.rbegin(); // larger index, i.e. latest, is the most likely to have space
    auto indexItEnd = cachedBuilders.rend();
    if(bulk::Properties::getForceToPreserveInsertionOrder()) {
      auto index = *indexIt;
      if(partitionBuilders[*indexIt]->isMatchingType(args..., missingHash)) {
        if(hasEnoughSpaceInPartition(partitionBuilders[*indexIt], neededLength)) {
          return index;
        }
      }
    } else {
      for(; indexIt != indexItEnd; ++indexIt) {
        auto index = *indexIt;
        if(partitionBuilders[index]->isMatchingType(args..., missingHash)) {
          if(hasEnoughSpaceInPartition(partitionBuilders[index], neededLength)) {
            return index;
          }
        }
      }
    }
    return size_t(-1);
  }
};

using TableSymbolRegistry = SymbolRegistry<Table, DefaultSymbolRegistry>;

} // namespace boss::engines::bulk