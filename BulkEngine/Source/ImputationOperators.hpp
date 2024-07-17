#pragma once

#include "Bulk.hpp"
#include "BulkExpression.hpp"
#include "BulkProperties.hpp"
#include "BulkUtilities.hpp"

#include <cfloat>
#include <random>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/learner.h>

namespace boss::engines::bulk {

template <typename T> constexpr auto safe_xgboost(T call) {
  int err = (call);
  if(err != 0) {
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                             ": error in " + __func__ + ":" + XGBGetLastError()); // NOLINT
  }
}

template <typename T>
std::unique_ptr<std::vector<double_t>>
computeStats(ValueBuilder<T> const& outputBuilder, Table::PartitionVectorPtr const& cleanPartitions,
             int col) {
  std::unique_ptr<std::vector<double_t>> stats = std::make_unique<std::vector<double_t>>(4);
  double_t min = DBL_MAX;
  double_t max = DBL_MIN;
  double_t sum = 0;
  double_t cnt = 0;
  auto outputType = outputBuilder.type()->id();
  for(auto const& partition : *cleanPartitions) {
    if(!partition || partition->length() == 0) { // check for empty partitions
      continue;
    }
    auto const& arrayPtr = partition->field(col);
    if(arrayPtr->type_id() != outputType) {
      continue;
    }
    auto const& typedArrayPtr = (ValueArrayPtr<T>)arrayPtr;
    for(auto i = 0UL; i < typedArrayPtr->length(); i++) { // NOLINT
      min = std::min(min, (double_t)typedArrayPtr->Value(i));
      max = std::max(max, (double_t)typedArrayPtr->Value(i));
      sum += (double_t)typedArrayPtr->Value(i);
      cnt++;
    }
  }
  (*stats)[0] = min;
  (*stats)[1] = max;
  (*stats)[2] = sum;
  (*stats)[3] = cnt;
  return stats;
};

template <typename... ArgumentTypes>
class Interpolate : public boss::engines::bulk::Operator<Interpolate, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant_merge_t<variant<tuple<Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>,
                      variant<tuple<ValueArrayPtr<Symbol>, ValueArrayPtr<Symbol>,
                                    Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>>;
  using Operator<Interpolate, ArgumentTypes...>::Operator;

  bool operator()(ValueArrayPtr<Symbol> const& tableNames, ValueArrayPtr<Symbol> const& columnNames,
                  Table::PartitionVectorPtr const& /*cleanPartitions*/, // use global table instead
                  Table::PartitionPtr const& dirtyPartition, int64_t columnIndex) {
    if(!dirtyPartition || dirtyPartition->length() == 0) {
      this->pushUp(dirtyPartition);
      return true;
    }
    int64_t outputLength = tableNames->length();
    if(outputLength == 0) {
      Table::PartitionPtr partition;
      this->pushUp(partition);
      return true;
    }
    // assume that all rows use the table, partitions and column definition of the first entry
    // TODO: check that the above it true!
    Symbol tableName(tableNames->SymbolArray::Value(0));
    Symbol columnName(columnNames->SymbolArray::Value(0));
    auto* tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableName);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableName.getName() + "\"_");
    }
    auto& table = *tablePtr;
    auto const& schema = table.getSchema();
    int col = -1;
    for(int i = 0; i < schema.size(); ++i) {
      if(schema[i]->name() == columnName.getName()) {
        col = i;
        break;
      }
    }
    if(col < 0) {
      throw std::logic_error("Cannot find column \"" + columnName.getName() + "\"_");
    }
    return impute(table.finaliseAndGetPartitions(), dirtyPartition, col);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& dirtyPartition, int64_t columnIndex) {
    if(!dirtyPartition || dirtyPartition->length() == 0) {
      this->pushUp(dirtyPartition);
      return true;
    }
    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      for(auto i = 0U; i < dirtyPartition->length(); ++i) {
        auto const& singleRow =
            std::static_pointer_cast<ComplexExpressionArray>(dirtyPartition->Slice(i, 1));
        singleRow->setGlobalIndexes(
            (ValueArrayPtr<int64_t>)(dirtyPartition->getGlobalIndexes()->Slice(i, 1)));
        singleRow->setTablePartitionIndexes(*dirtyPartition);
        if(!impute(cleanPartitions, singleRow, columnIndex - 1)) {
          return false;
        }
      }
      return true;
    }
    return impute(cleanPartitions, dirtyPartition, columnIndex - 1);
  }

private:
  bool impute(Table::PartitionVectorPtr const& cleanPartitions,
              Table::PartitionPtr const& dirtyPartition, int64_t col) {
    auto type = arrow::Type::NA;
    for(auto& partition : *cleanPartitions) {
      if(!partition || partition->length() == 0) {
        continue;
      }
      type = partition->field(col)->type_id();
      if(type == arrow::Type::INT32) {
        this->pushUp(impute<int32_t>(cleanPartitions, dirtyPartition, col));
        return true;
      }
      if(type == arrow::Type::INT64) {
        this->pushUp(impute<int64_t>(cleanPartitions, dirtyPartition, col));
        return true;
      }
      if(type == arrow::Type::FLOAT) {
        this->pushUp(impute<float_t>(cleanPartitions, dirtyPartition, col));
        return true;
      }
      if(type == arrow::Type::DOUBLE) {
        this->pushUp(impute<double_t>(cleanPartitions, dirtyPartition, col));
        return true;
      }
    }
    if(type == arrow::Type::NA) {
      throw std::runtime_error("No clean partitions when calling Interpolate");
    }
    throw std::runtime_error("Unsupported data type when calling Interpolate");
  }

  template <typename T>
  auto impute(Table::PartitionVectorPtr const& cleanPartitions,
              Table::PartitionPtr const& dirtyPartition, int64_t col) {
    auto outputBuilder = ValueBuilder<T>(dirtyPartition->length());
    if(outputBuilder.length() == 0) { // check for empty partition
      return (ValueArrayPtr<T>)outputBuilder;
    }
    auto outputType = outputBuilder.type()->id();

    auto const& globalIndexesArray = *dirtyPartition->getGlobalIndexes();
    auto numGlobalIndexes = globalIndexesArray.length();
    auto const* globalIndexes = globalIndexesArray.rawData();

    auto const& tablePartitionIndexesArray = *dirtyPartition->getTablePartitionIndexes();
    auto numTablePartitions = tablePartitionIndexesArray.length();
    auto const* tablePartitionIndexes = tablePartitionIndexesArray.rawData();

    auto const& tablePartitionLocalIndexesArray = *dirtyPartition->getTablePartitionLocalIndexes();
    auto const* tablePartitionLocalIndexes = tablePartitionLocalIndexesArray.rawData();

    std::vector<ValueArrayPtr<T>> globalPartitions{};
    // iterate backwards to minimize globalPartitions.resize()
    // since the last partition is likely the highest index
    for(size_t i = cleanPartitions->size() - 1; i < cleanPartitions->size(); --i) {
      auto& partition = (*cleanPartitions)[i];
      if(!partition || partition->length() == 0) { // check for empty partitions
        continue;
      }
      auto const& column = partition->field(col);
      if(column->type_id() != outputType) { // and type mismatch
        continue;
      }
      auto const& cleanGlobalIndexesArray = *partition->getGlobalIndexes();
      auto const* cleanGlobalIndexes = cleanGlobalIndexesArray.rawData();
      auto cleanPartitionIndex = tablePartitionIndexes[cleanGlobalIndexes[0]];
      if(cleanPartitionIndex >= globalPartitions.size()) {
        globalPartitions.resize(cleanPartitionIndex + 1);
      }
      globalPartitions[cleanPartitionIndex] = (ValueArrayPtr<T>)column;
    }

    auto findPrevIndexAndValue =
        [&tablePartitionIndexes, &tablePartitionLocalIndexes,
         &globalPartitions](int64_t globalIndex) -> std::optional<std::pair<int64_t, T>> {
      for(globalIndex--; globalIndex >= 0; globalIndex--) {
        if(tablePartitionIndexes[globalIndex] < 0) {
          // the tuple has been filtered out
          continue;
        }
        auto const& prevPartitionIndex = tablePartitionIndexes[globalIndex];
        if(prevPartitionIndex >= globalPartitions.size() || !globalPartitions[prevPartitionIndex]) {
          // the tuple belongs to a dirty partition
          continue;
        }
        auto const& prevPartition = globalPartitions[prevPartitionIndex];
        auto const& prevLocalIndex = tablePartitionLocalIndexes[globalIndex];
        if(prevLocalIndex >= prevPartition->length()) {
          throw std::runtime_error("wrong prevLocalIndex: " + std::to_string(prevLocalIndex));
        }
        return std::make_pair(globalIndex, prevPartition->Value(prevLocalIndex));
      }
      return {};
    };

    auto findNextIndexAndValue =
        [&tablePartitionIndexes, &numTablePartitions, &tablePartitionLocalIndexes,
         &globalPartitions](int64_t globalIndex) -> std::optional<std::pair<int64_t, T>> {
      for(globalIndex++; globalIndex < numTablePartitions; globalIndex++) {
        if(tablePartitionIndexes[globalIndex] < 0) {
          // the tuple has been filtered out
          continue;
        }
        auto const& nextPartitionIndex = tablePartitionIndexes[globalIndex];
        if(nextPartitionIndex >= globalPartitions.size() || !globalPartitions[nextPartitionIndex]) {
          // the tuple belongs to a dirty partition
          continue;
        }
        auto const& nextPartition = globalPartitions[nextPartitionIndex];
        auto const& nextLocalIndex = tablePartitionLocalIndexes[globalIndex];
        if(nextLocalIndex >= nextPartition->length()) {
          throw std::runtime_error("wrong nextLocalIndex: " + std::to_string(nextLocalIndex));
        }
        return std::make_pair(globalIndex, nextPartition->Value(nextLocalIndex));
      }
      return {};
    };

    auto batchInterpolate = [&findPrevIndexAndValue, &findNextIndexAndValue, &outputBuilder,
                             &globalIndexes](auto outputIndexStart, auto outputIndexEnd) {
      auto globalIndexStart = globalIndexes[outputIndexStart];
      auto globalIndexEnd = globalIndexes[outputIndexEnd - 1];
      auto prevIndexAndValue = findPrevIndexAndValue(globalIndexStart);
      auto nextIndexAndValue = findNextIndexAndValue(globalIndexEnd);
      if(!prevIndexAndValue) {
        if(!nextIndexAndValue) {
          throw std::runtime_error("No clean values to evaluate Interpolate");
        }
        auto [nextIndex, nextValue] = *nextIndexAndValue;
        for(auto i = outputIndexStart; i < outputIndexEnd; ++i) {
          outputBuilder[i] = nextValue;
        }
        return;
      }
      if(!nextIndexAndValue) {
        auto [prevIndex, prevValue] = *prevIndexAndValue;
        for(auto i = outputIndexStart; i < outputIndexEnd; ++i) {
          outputBuilder[i] = prevValue;
        }
        return;
      }
      auto [prevIndex, prevValue] = *prevIndexAndValue;
      auto [nextIndex, nextValue] = *nextIndexAndValue;
      for(auto i = 0; i < outputIndexEnd - outputIndexStart; ++i) {
        outputBuilder[outputIndexStart + i] =
            prevValue + (T)(1 + i) * (nextValue - prevValue) / (T)(nextIndex - prevIndex);
      }
    };

    // interpolate per group of consecutive missing values
    auto startOutputIndex = 0;
    for(int i = 1; i < numGlobalIndexes; ++i) {
      if(globalIndexes[i] != globalIndexes[i - 1] + 1) {
        batchInterpolate(startOutputIndex, i);
        startOutputIndex = i;
      }
    }
    if(startOutputIndex < numGlobalIndexes - 1) {
      batchInterpolate(startOutputIndex, numGlobalIndexes);
    }

    return (ValueArrayPtr<T>)outputBuilder;
  }
};
namespace {
boss::engines::bulk::Engine::Register<Interpolate> const imp01("Interpolate"); // NOLINT
}

template <typename... ArgumentTypes>
class DecisionTree : public boss::engines::bulk::Operator<DecisionTree, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant_merge_t<variant<tuple<Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>,
                      variant<tuple<ValueArrayPtr<Symbol>, ValueArrayPtr<Symbol>,
                                    Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>>;
  using Operator<DecisionTree, ArgumentTypes...>::Operator;

  bool operator()(ValueArrayPtr<Symbol> const& tableNames,
                  ValueArrayPtr<Symbol> const& imputedColumnNames,
                  Table::PartitionVectorPtr const& /*cleanPartitions*/, // use global table instead
                  Table::PartitionPtr const& partition2, int64_t columnIndex) {
    if(!partition2 || partition2->length() == 0) {
      this->pushUp(partition2);
      return true;
    }
    int64_t outputLength = tableNames->length();
    if(outputLength == 0) {
      Table::PartitionPtr partition;
      this->pushUp(partition);
      return true;
    }
    Symbol tableName(tableNames->SymbolArray::Value(0));
    auto* tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableName);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableName.getName() + "\"_");
    }
    auto& table = *tablePtr;
    // only if not cached yet
    if(globalColumnIndex < 0) {
      // assume that all rows use the table and column definition of the first entry
      // TODO: check that the above it true!
      Symbol imputedColumnName(imputedColumnNames->SymbolArray::Value(0));
      auto const& globalSchema = table.getSchema();
      auto const& localSchema = partition2->getSchema();
      globalColumnIndices.reserve(localSchema.size());
      localColumnIndices.reserve(localSchema.size());
      for(int64_t i = 0; i < globalSchema.size(); ++i) {
        if(globalSchema[i]->name() == imputedColumnName.getName()) {
          globalColumnIndex = i + 1;
          continue;
        }
        for(int64_t j = 0; j < localSchema.size(); ++j) {
          if(j == columnIndex - 1) {
            continue;
          }
          if(globalSchema[i]->name() == localSchema[j]->name()) {
            globalColumnIndices.emplace_back(i);
            localColumnIndices.emplace_back(j);
            break;
          }
        }
      }
      if(globalColumnIndex < 0) {
        throw std::logic_error("Cannot find column \"" + imputedColumnName.getName() + "\"_");
      }
      if(localColumnIndices.empty()) {
        throw std::logic_error("Cannot find any input column to use");
      }
    }
    if(localColumnIndices.size() == partition2->getSchema().size() - 1) {
      // if all local columns are used, we can just pass empty vector
      localColumnIndices.clear();
    }
    if(globalColumnIndices.size() == table.getSchema().size() - 1) {
      // if all global columns are used, we can just pass empty vector
      globalColumnIndices.clear();
    }
    return operator()(table.finaliseAndGetPartitions(), partition2, columnIndex, localColumnIndices,
                      globalColumnIndex, globalColumnIndices);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& partition2, int64_t columnIndex) {
    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      for(auto i = 0U; i < partition2->length(); ++i) {
        auto const& singleRow =
            std::static_pointer_cast<ComplexExpressionArray>(partition2->Slice(i, 1));
        if(!operator()(cleanPartitions, singleRow, columnIndex, std::vector<int64_t>{}, columnIndex,
                       std::vector<int64_t>{})) {
          return false;
        }
      }
      return true;
    }
    return operator()(cleanPartitions, partition2, columnIndex, std::vector<int64_t>{}, columnIndex,
                      std::vector<int64_t>{});
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& partition2, int64_t sampleColumnIndex,
                  std::vector<int64_t> const& sampleColumnIndices, int64_t dataColumnIndex,
                  std::vector<int64_t> const& dataColumnIndices) {
    if(!cleanPartitions || cleanPartitions->empty() || !partition2 || partition2->length() == 0) {
      this->pushUp(partition2);
      return true;
    }

    auto dataCol = dataColumnIndex - 1;

    Table::PartitionPtr partition1;
    int64_t numRows = 0;
    for(auto const& p : *cleanPartitions) {
      if(p && p->length() > 0 && p->field(dataCol)->type()->id() != arrow::Type::EXTENSION) {
        if(partition1 == nullptr) {
          partition1 = p;
        }
        // break;
        numRows += p->length();
      }
    }
    if(!partition1) {
      throw std::runtime_error("No clean partitions when calling DecisionTree");
    }

    int64_t numColumns = partition1->num_fields();

    if(lastNumColumns != numColumns || (lastNumRows != numRows &&
       (lastNumRows < 0 || (std::abs(lastNumRows - numRows) >= lastNumRows / 10)))) { // NOLINT
      lastNumRows = numRows;
      lastNumColumns = numColumns;
      if(boosterHandle != nullptr) {
        safe_xgboost(XGBoosterFree(boosterHandle));
        boosterHandle = nullptr;
      }
      if(trainHandle != nullptr) {
        safe_xgboost(XGDMatrixFree(trainHandle));
        trainHandle = nullptr;
      }
    }

    if(trainHandle == nullptr) {
      // wrap data pointers and types
      if(prepareBatches(*cleanPartitions, dataCol, dataColumnIndices)) {
        // train the decision tree model
        trainDecisionTree();
      }
      if(callbackData.batchProxy != nullptr) {
        safe_xgboost(XGDMatrixFree(callbackData.batchProxy));
      }
      callbackData.batches.clear();
    }

    // the partition we impute
    auto sampleRows = partition2->length();
    auto sampleCol = sampleColumnIndex - 1;

    if(trainHandle == nullptr) {
      // something went wrong with the training
      // we fallback to mean calculation (like Weka's RepTree)
      switch(partition1->field(dataCol)->type()->id()) {
      case arrow::Type::INT32: {
        this->pushUp(imputeFallback<int32_t>(sampleRows, cleanPartitions, sampleCol));
      } break;
      case arrow::Type::INT64: {
        this->pushUp(imputeFallback<int64_t>(sampleRows, cleanPartitions, sampleCol));
      } break;
      case arrow::Type::FLOAT: {
        this->pushUp(imputeFallback<float_t>(sampleRows, cleanPartitions, sampleCol));
      } break;
      case arrow::Type::DOUBLE: {
        this->pushUp(imputeFallback<double_t>(sampleRows, cleanPartitions, sampleCol));
      } break;
      default: {
        throw std::runtime_error("error: Unsupported data type when calling DecisionTree");
      } break;
      }
      return true;
    }

    // wrap the data for which we want to perform predictions
    std::vector<void*> data;
    std::vector<const char*> types;
    if(sampleColumnIndices.empty()) {
      prepareData(data, types, sampleRows, *partition2, sampleCol);
    } else {
      prepareData(data, types, sampleRows, *partition2, sampleCol, sampleColumnIndices);
    }
    // return results based on the column data type
    switch(partition1->field(dataCol)->type()->id()) {
    case arrow::Type::INT32: {
      this->pushUp(predict<int32_t>(data, types, sampleRows));
    } break;
    case arrow::Type::INT64: {
      this->pushUp(predict<int64_t>(data, types, sampleRows));
    } break;
    case arrow::Type::FLOAT: {
      this->pushUp(predict<float_t>(data, types, sampleRows));
    } break;
    case arrow::Type::DOUBLE: {
      this->pushUp(predict<double_t>(data, types, sampleRows));
    } break;
    default: {
      throw std::runtime_error("error: Unsupported data type when calling DecisionTree");
    } break;
    }
    return true;
  }

private:
  // cached data for fallback to mean calculation
  std::unique_ptr<std::vector<double_t>> stats = nullptr;

  const size_t epochsSlow = 10; // NOLINT
  const size_t epochsFast = 2;  // NOLINT
  //std::unordered_map<int, std::unordered_map<std::string, uint64_t>> stringMap;
  //std::unordered_map<int, std::vector<std::vector<uint64_t>>> vectorMap;

  // cached for global table parameters
  int64_t globalColumnIndex = -1;
  std::vector<int64_t> globalColumnIndices;
  std::vector<int64_t> localColumnIndices;

  // cached from xgboost training
  inline static int64_t lastNumRows = -1;
  inline static int64_t lastNumColumns = -1;
  inline static DMatrixHandle trainHandle = nullptr;
  inline static BoosterHandle boosterHandle = nullptr;

  struct DataBatch {
    int64_t rows = 0;
    // data
    std::vector<void*> data;
    std::vector<const char*> types;
    // labels
    void* labels = nullptr;
    int labelType = 0;
  };

  struct CallbackData {
    std::vector<DataBatch> batches;
    typename std::vector<DataBatch>::iterator batchIterator;
    DMatrixHandle batchProxy = nullptr;
  } callbackData;

  static int DataBatchIteratorNext(DataIterHandle handle) {
    auto& data = *static_cast<CallbackData*>(handle);
    auto& batchIt = data.batchIterator;
    auto& batches = data.batches;
    if(batchIt == batches.end()) {
      batchIt = batches.begin();
      return 0; // end of batches
    }

    auto& batch = *batchIt;
    auto& batchProxy = data.batchProxy;

    // convert data
    safe_xgboost(XGProxyDMatrixSetDataFromDT(batchProxy, batch.data.data(), batch.types.data(),
                                             batch.rows, batch.types.size()));
    // convert labels
    safe_xgboost(
        XGDMatrixSetDenseInfo(batchProxy, "label", batch.labels, batch.rows, batch.labelType));

    ++batchIt;
    return 1; // continue to next batch
  }

  static void DataBatchIteratorReset(DataIterHandle handle) {
    auto& data = *static_cast<CallbackData*>(handle);
    auto& batchIt = data.batchIterator;
    auto& batches = data.batches;
    batchIt = batches.begin();
  }

  bool prepareBatches(Table::PartitionVector const& partitions, size_t labelCol,
                      std::vector<int64_t> const& inputColumnIndices) {
    auto numBatches = partitions.size();
    callbackData.batches.clear();
    callbackData.batches.reserve(numBatches);
    for(int i = 0; i < numBatches; ++i) {
      auto const& partitionPtr = partitions[i];
      if(!partitionPtr || partitionPtr->length() == 0) {
        continue;
      }
      auto const& partition = *partitionPtr;
      auto& batch = callbackData.batches.emplace_back();
      batch.rows = partition.length();
      if(prepareLabelData(batch.labels, batch.labelType, partition, labelCol)) {
        if(inputColumnIndices.empty()
               ? prepareData(batch.data, batch.types, batch.rows, partition, labelCol)
               : prepareData(batch.data, batch.types, batch.rows, partition, labelCol,
                             inputColumnIndices)) {
          if(!batch.data.empty()) {
            // no features
            continue;
          }
        }
      }
      // something failed
      callbackData.batches.pop_back();
    }

    if(callbackData.batches.empty()) {
      return false;
    }

    callbackData.batchIterator = callbackData.batches.begin();
    safe_xgboost(XGProxyDMatrixCreate(&callbackData.batchProxy));
    return true;
  }

  template <typename... OptionalIndices>
  bool prepareData(std::vector<void*>& data, std::vector<const char*>& types, size_t rows,
                   Table::Partition const& partition, size_t labelCol,
                   OptionalIndices const&... inputColumnIndices) {
    // initialise data and type vectors
    auto cols = partition.num_fields();
    data.reserve(cols - 1);
    types.reserve(cols - 1);

    for(auto c = 0L; c < cols; c++) { // NOLINT
      if constexpr(sizeof...(OptionalIndices) > 0) {
        // use the explicit vector of column indices
        if(((!inputColumnIndices.empty() &&
             (std::find(inputColumnIndices.begin(), inputColumnIndices.end(), c) ==
              inputColumnIndices.end())) &&
            ...)) {
          continue;
        }
      } else {
        // use all the columns, only skip the label column
        if(c == labelCol) {
          continue;
        }
      }
      auto const& colPtr = partition.field(c);
      auto const& col = *colPtr;
      void* colData = nullptr;
      char const* colType = nullptr;
      switch(col.type()->id()) {
      case arrow::Type::STRING: {
        // hash values
        /*auto& map = stringMap[c];
        auto& nums = vectorMap[c];
        nums.resize(rows);
        auto const& parColArray = ValueArray<std::string>(col);
        for(auto i = 0L; i < rows; i++) {
          auto const iter = map.find(parColArray.GetString(i));
          if(iter != map.end()) {
            nums[i] = iter->second;
          } else {
            map[parColArray.GetString(i)] = i;
            nums[i] = i;
          }
        }
        colData = reinterpret_cast<void*>(nums.data());
        colType = "int64";*/
      } break;
      case arrow::Type::INT32: {
        colData = (void*)(ValueArray<int32_t>(col).rawData());
        colType = "int32";
      } break;
      case arrow::Type::INT64: {
        colData = (void*)(ValueArray<int64_t>(col).rawData());
        colType = "int64";
      } break;
      case arrow::Type::FLOAT: {
        colData = (void*)(ValueArray<float_t>(col).rawData());
        colType = "float32";
      } break;
      case arrow::Type::DOUBLE: {
        colData = (void*)(ValueArray<double_t>(col).rawData());
        colType = "float64";
      } break;
      default: {
        // usually happens if this is not a clean partition
        return false;
      } break;
      }

      if(colData != nullptr) {
        data.emplace_back(colData);
        types.emplace_back(colType);
      }
    }
    return true;
  }

  bool prepareLabelData(void*& labels, int& labelType, Table::Partition const& partition,
                        size_t labelCol) {
    auto const& col = *partition.field(labelCol);
    switch(col.type()->id()) {
    case arrow::Type::INT32: {
      labels = (void*)(ValueArray<int32_t>(col).rawData());
      labelType = static_cast<int>(xgboost::DataType::kUInt32);
    } break;
    case arrow::Type::INT64: {
      labels = (void*)(ValueArray<int64_t>(col).rawData());
      labelType = static_cast<int>(xgboost::DataType::kUInt64);
    } break;
    case arrow::Type::FLOAT: {
      labels = (void*)(ValueArray<float_t>(col).rawData());
      labelType = static_cast<int>(xgboost::DataType::kFloat32);
    } break;
    case arrow::Type::DOUBLE: {
      labels = (void*)(ValueArray<double_t>(col).rawData());
      labelType = static_cast<int>(xgboost::DataType::kDouble);
    } break;
    default: {
      // usually happens if this is not a clean partition
      return false;
    } break;
    }
    return true;
  }

  void trainDecisionTree() {
    // 1) Create a DMatrix object which iterates on the batches
    safe_xgboost(XGDMatrixCreateFromCallback(
        &callbackData, callbackData.batchProxy, &DataBatchIteratorReset, &DataBatchIteratorNext,
        "{\"missing\": NaN, \"cache_prefix\": \"cache\"}", &trainHandle));
    // 2) Create the booster and load some parameters
    setUpBooster();

    // 3) Perform learning iterations
    for(int iter = 0; iter < epochsSlow; iter++) { // NOLINT
      // Update the model performance for each iteration
      safe_xgboost(XGBoosterUpdateOneIter(boosterHandle, iter, trainHandle));
    }
  }

  void setUpBooster() {
    safe_xgboost(XGBoosterCreate(&trainHandle, 1, &boosterHandle));
    safe_xgboost(
        XGBoosterSetParam(boosterHandle, "booster",
                          "gbtree")); // Can be gbtree, gblinear or dart; gbtree and dart use tree
                                      // based models while gblinear uses linear functions.
    safe_xgboost(XGBoosterSetParam(boosterHandle, "tree_method",
                                   "hist")); // default= auto, hist accelerates training
    safe_xgboost(XGBoosterSetParam(boosterHandle, "objective",
                                   "reg:squarederror")); // reg:squarederror, reg:squaredlogerror,
                                                         // reg:linear, reg:pseudohubererror
    safe_xgboost(
        XGBoosterSetParam(boosterHandle, "max_depth", "6")); // maximum depth of a tree (try 3?)
    safe_xgboost(XGBoosterSetParam(boosterHandle, "eta",
                                   "0.3")); // learning rate: step size shrinkage (try 1.0?)
    safe_xgboost(XGBoosterSetParam(
        boosterHandle, "gamma",
        "1.0")); // minimum loss reduction required to make a further partition -- the larger
                 // gamma is, the more conservative the algorithm will be.
    safe_xgboost(
        XGBoosterSetParam(boosterHandle, "min_child_weight",
                          "1")); // minimum sum of instance weight(hessian) needed in a child
    safe_xgboost(
        XGBoosterSetParam(boosterHandle, "subsample", "0.5")); // set to 0.5 for overfitting
    safe_xgboost(XGBoosterSetParam(boosterHandle, "colsample_bytree", "1"));
    safe_xgboost(XGBoosterSetParam(boosterHandle, "num_parallel_tree", "1"));
  }

  template <typename T>
  auto predict(std::vector<void*>& data, std::vector<const char*>& types, size_t rows) {
    auto outputBuilder = ValueBuilder<T>(rows);

    DMatrixHandle testHandle = nullptr;
    safe_xgboost(
        XGDMatrixCreateFromDT(data.data(), types.data(), rows, types.size(), &testHandle, 0));

    bst_ulong outLength = 0;
    float const* outFloatValues = nullptr;
    safe_xgboost(XGBoosterPredict(boosterHandle, testHandle, 0, 0, 0, &outLength, &outFloatValues));

    // copy result to output buffer
    for(auto i = 0UL; i < outLength; i++) {
      outputBuilder[i] = static_cast<T>(outFloatValues[i]);
    }

    // free xgboost internal structures
    safe_xgboost(XGDMatrixFree(testHandle));

    return (ValueArrayPtr<T>)outputBuilder;
  }

  template <typename T>
  auto imputeFallback(size_t outputSize, Table::PartitionVectorPtr const& cleanPartitions,
                      int col) {
    auto outputBuilder = ValueBuilder<T>(outputSize);
    if(outputBuilder.length() == 0) { // check for empty partition
      return (ValueArrayPtr<T>)outputBuilder;
    }

    // compute statistics if they don't exist
    // todo: detect when a change has happened to update the statistics
    if(stats == nullptr) {
      stats = computeStats(outputBuilder, cleanPartitions, col);
    }
    auto mean = (*stats)[2] / (*stats)[3];

    for(auto index = 0L; index < outputBuilder.length(); ++index) {
      outputBuilder[index] = (T)mean;
    }

    return (ValueArrayPtr<T>)outputBuilder;
  }

  void close() override {
    // free xgboost internal structures
    /*if(boosterHandle != nullptr) {
      safe_xgboost(XGBoosterFree(boosterHandle));
    }
    if(trainHandle != nullptr) {
      safe_xgboost(XGDMatrixFree(trainHandle));
    }
    if(callbackData.batchProxy != nullptr) {
      safe_xgboost(XGDMatrixFree(callbackData.batchProxy));
    }*/
  }
};
namespace {
boss::engines::bulk::Engine::Register<DecisionTree> const imp02("DecisionTree"); // NOLINT
}

template <typename... ArgumentTypes>
class HotDeck : public boss::engines::bulk::Operator<HotDeck, ArgumentTypes...> {
private:
  vector<int> prefixSums;

public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<Symbol>, ValueArrayPtr<Symbol>>,
                                 tuple<Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>;
  using Operator<HotDeck, ArgumentTypes...>::Operator;

  bool operator()(ValueArrayPtr<Symbol> const& tableNames,
                  ValueArrayPtr<Symbol> const& columnNames) {
    int64_t outputLength = tableNames->length();
    if(outputLength == 0) {
      Table::PartitionPtr partition;
      this->pushUp(partition);
      return true;
    }
    // assume that all rows use the table, partitions and column definition of the first entry
    // TODO: check that the above it true!
    Symbol tableName(tableNames->SymbolArray::Value(0));
    Symbol columnName(columnNames->SymbolArray::Value(0));
    auto* tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableName);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableName.getName() + "\"_");
    }
    auto& table = *tablePtr;
    auto const& schema = table.getSchema();
    int col = -1;
    for(int i = 0; i < schema.size(); ++i) {
      if(schema[i]->name() == columnName.getName()) {
        col = i;
        break;
      }
    }
    if(col < 0) {
      throw std::logic_error("Cannot find column \"" + columnName.getName() + "\"_");
    }
    return operator()(table.finaliseAndGetPartitions(), outputLength, col);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& partition2, int64_t columnIndex) {
    if(!partition2 || partition2->length() == 0) {
      this->pushUp(partition2);
      return true;
    }

    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      for(auto i = 0; i < partition2->length(); ++i) {
        if(!operator()(cleanPartitions, 1, columnIndex - 1)) {
          return false;
        }
      }
      return true;
    }
    return operator()(cleanPartitions, partition2->length(), columnIndex - 1);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions, int64_t outputLength,
                  int64_t col) {
    auto type = arrow::Type::NA;
    for(auto& partition : *cleanPartitions) {
      if(!partition || partition->length() == 0) {
        continue;
      }
      type = partition->field(col)->type_id();
      if(type == arrow::Type::INT32) {
        this->pushUp(impute<int32_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::INT64) {
        this->pushUp(impute<int64_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::FLOAT) {
        this->pushUp(impute<float_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::DOUBLE) {
        this->pushUp(impute<double_t>(outputLength, cleanPartitions, col));
        return true;
      }
    }
    if(type == arrow::Type::NA) {
      throw std::runtime_error("No clean partitions when calling HotDeck");
    }
    throw std::runtime_error("Unsupported data type when calling HotDeck");
  }

private:
  template <typename T>
  auto impute(int64_t size, Table::PartitionVectorPtr const& cleanPartitions, int col) {
    auto outputBuilder = ValueBuilder<T>(size);
    if(outputBuilder.length() == 0) { // check for empty partition
      return (ValueArrayPtr<T>)outputBuilder;
    }

    // missing values in a column are replaced with randomly
    // selected complete values from the same column of another partition

    // maybe initialize this once?
    // std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, cleanPartitions->size() - 1); // NOLINT
    /*for (auto &n : *cleanPartitions) { // todo: do we need uniform distribution? use pickIndex()
      prefixSums.push_back(n->length() + (prefixSums.empty() ? 0 : prefixSums.back()));
    }*/

    auto outputType = outputBuilder.type()->id();
    for(auto index = 0L; index < outputBuilder.length(); ++index) { // NOLINT
      std::shared_ptr<arrow::Array> arrayPtr;
      while(!arrayPtr || arrayPtr->length() == 0    // check for empty partitions
            || arrayPtr->type_id() != outputType) { // and type mismatch
        auto const& p = (*cleanPartitions)[distr(gen)];
        arrayPtr = p ? p->field(col) : std::shared_ptr<arrow::Array>();
      }
      std::uniform_int_distribution<> distr2(0, arrayPtr->length() - 1);
      outputBuilder[index] = ((ValueArrayPtr<T>)(arrayPtr))->Value(distr2(gen));
    }

    return (ValueArrayPtr<T>)outputBuilder;
  }

  int pickIndex() {
    // generate a random number in the range of [0, 1]
    float randNum = (float)rand() / RAND_MAX; // NOLINT
    float target = randNum * prefixSums.back();
    return upper_bound(begin(prefixSums), end(prefixSums), target) - begin(prefixSums);
  }
};
namespace {
boss::engines::bulk::Engine::Register<HotDeck> const imp03("HotDeck"); // NOLINT
}

template <typename... ArgumentTypes>
class ApproxMean : public boss::engines::bulk::Operator<ApproxMean, ArgumentTypes...> {
private:
  // cached data
  std::unique_ptr<std::vector<double_t>> stats = nullptr;

public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<Symbol>, ValueArrayPtr<Symbol>>,
                                 tuple<Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>;
  using Operator<ApproxMean, ArgumentTypes...>::Operator;

  bool operator()(ValueArrayPtr<Symbol> const& tableNames,
                  ValueArrayPtr<Symbol> const& columnNames) {
    int64_t outputLength = tableNames->length();
    if(outputLength == 0) {
      Table::PartitionPtr partition;
      this->pushUp(partition);
      return true;
    }
    // assume that all rows use the table, partitions and column definition of the first entry
    // TODO: check that the above it true!
    Symbol tableName(tableNames->SymbolArray::Value(0));
    Symbol columnName(columnNames->SymbolArray::Value(0));
    auto* tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableName);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableName.getName() + "\"_");
    }
    auto& table = *tablePtr;
    auto const& schema = table.getSchema();
    int col = -1;
    for(int i = 0; i < schema.size(); ++i) {
      if(schema[i]->name() == columnName.getName()) {
        col = i;
        break;
      }
    }
    if(col < 0) {
      throw std::logic_error("Cannot find column \"" + columnName.getName() + "\"_");
    }
    return operator()(table.finaliseAndGetPartitions(), outputLength, col);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& partition2, int64_t columnIndex) {
    if(!partition2 || partition2->length() == 0) {
      this->pushUp(partition2);
      return true;
    }

    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      for(auto i = 0; i < partition2->length(); ++i) {
        if(!operator()(cleanPartitions, 1, columnIndex - 1)) {
          return false;
        }
      }
      return true;
    }
    return operator()(cleanPartitions, partition2->length(), columnIndex - 1);
  }

  bool operator()(Table::PartitionVectorPtr const& cleanPartitions, int64_t outputLength,
                  int64_t col) {
    auto type = arrow::Type::NA;
    for(auto& partition : *cleanPartitions) {
      if(!partition || partition->length() == 0) {
        continue;
      }
      type = partition->field(col)->type_id();
      if(type == arrow::Type::INT32) {
        this->pushUp(impute<int32_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::INT64) {
        this->pushUp(impute<int64_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::FLOAT) {
        this->pushUp(impute<float_t>(outputLength, cleanPartitions, col));
        return true;
      }
      if(type == arrow::Type::DOUBLE) {
        this->pushUp(impute<double_t>(outputLength, cleanPartitions, col));
        return true;
      }
    }
    if(type == arrow::Type::NA) {
      throw std::runtime_error("No clean partitions when calling ApproxMean");
    }
    throw std::runtime_error("Unsupported data type when calling ApproxMean");
  }

private:
  template <typename T>
  auto impute(size_t outputSize, Table::PartitionVectorPtr const& cleanPartitions, int col) {
    auto outputBuilder = ValueBuilder<T>(outputSize);
    if(outputBuilder.length() == 0) { // check for empty partition
      return (ValueArrayPtr<T>)outputBuilder;
    }

    // compute statistics if they don't exist
    // todo: detect when a change has happened to update the statistics
    if(stats == nullptr) {
      stats = computeStats(outputBuilder, cleanPartitions, col);
    }
    auto mean = (*stats)[2] / (*stats)[3];

    for(auto index = 0L; index < outputBuilder.length(); ++index) {
      outputBuilder[index] = (T)mean;
    }

    return (ValueArrayPtr<T>)outputBuilder;
  }
};
namespace {
boss::engines::bulk::Engine::Register<ApproxMean> const imp04("ApproxMean"); // NOLINT
}

template <typename... ArgumentTypes>
class NoOp : public boss::engines::bulk::Operator<NoOp, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<int32_t>, tuple<int64_t>, tuple<float_t>, tuple<double_t>, tuple<std::string>,
              tuple<ValueArrayPtr<int32_t>>, tuple<ValueArrayPtr<int64_t>>,
              tuple<ValueArrayPtr<float_t>>, tuple<ValueArrayPtr<double_t>>,
              tuple<ValueArrayPtr<std::string>>,
              tuple<ComplexExpression, Table::PartitionVectorPtr, Table::PartitionPtr, int64_t>>;
  using Operator<NoOp, ArgumentTypes...>::Operator;
  template <typename T> void operator()(T val) { this->pushUp(val); }
  template <typename T> void operator()(ValueArrayPtr<T> const& arrayPtr) {
    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      using OutputT = decltype(arrayPtr->Value(0));
      auto outputBuilder = ValueBuilder<OutputT>(arrayPtr->length());
      for(int64_t index = 0; index < arrayPtr->length(); ++index) {
        outputBuilder[index] = arrayPtr->Value(index);
      }
      this->pushUp((ValueArrayPtr<OutputT>)outputBuilder);
    } else {
      this->pushUp(arrayPtr);
    }
  }
  void operator()(ValueArrayPtr<std::string> const& arrayPtr) {
    if(bulk::Properties::getDisableExpressionDecomposition()) {
      // call the imputation operator for each row one by one
      auto outputBuilder = ValueBuilder<std::string>(0);
      outputBuilder.Reserve(arrayPtr->length());
      outputBuilder.ReserveData(arrayPtr->total_values_length());
      for(int64_t index = 0; index < arrayPtr->length(); ++index) {
        outputBuilder.UnsafeAppend(arrayPtr->Value(index));
      }
      this->pushUp((ValueArrayPtr<std::string>)outputBuilder);
    } else {
      this->pushUp(arrayPtr);
    }
  }
  void operator()(ComplexExpression const& expr, Table::PartitionVectorPtr const& cleanPartitions,
                  Table::PartitionPtr const& dirtyPartition, int64_t columnIndex) {
    ExpressionArguments args = expr.getArguments();
    args.reserve(args.size() + 3);
    args.emplace_back(cleanPartitions);
    args.emplace_back(dirtyPartition);
    args.emplace_back(columnIndex);
    auto output = ComplexExpression(expr.getHead(), std::move(args));
    this->pushUp(this->evaluateInternal(std::move(output)));
  }
};
namespace {
boss::engines::bulk::Engine::Register<NoOp> const no1("NoOp1"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no2("NoOp2"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no3("NoOp3"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no4("NoOp4"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no5("NoOp5"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no6("NoOp6"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no7("NoOp7"); // NOLINT
boss::engines::bulk::Engine::Register<NoOp> const no8("NoOp8"); // NOLINT
} // namespace

} // namespace boss::engines::bulk
