#pragma once

#include "../Bulk.hpp"
#include "../BulkArrays.hpp"
#include "../BulkExpression.hpp"
#include "../BulkProperties.hpp"
#include "ComplexExpressionArray.hpp"

#include <arrow/builder.h>

#include <bitset>

namespace boss::engines::bulk {

/** ComplexExpressionArrayBuilder is the builder corresponding to ComplexExpressionArray.
 * Because it uses custom type ComplexExpressionArrayType, Arrow will automatically
 * pick it up to create ComplexExpressionArray when finishing the builder.*/
class ComplexExpressionArrayBuilder : public arrow::StructBuilder {
public:
  ComplexExpressionArrayBuilder(Symbol const& head, arrow::FieldVector const& schema,
                                arrow::ArrayVector const& fromFields, bool clear = false,
                                int64_t minSize = 0,
                                arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::StructBuilder(std::make_shared<arrow::StructType>(schema), pool,
                             createChildrenFromExistingColumns(fromFields, clear, minSize, pool)),
        head(head), cachedMissings(ComplexExpressionArray::computeMissingHash(fromFields)) {
    if(!clear && !fromFields.empty()) {
      // this is not done by the StructBuilder constructor
      capacity_ = length_ = std::max(minSize, fromFields[0]->length());
    } else if(minSize > 0) {
      auto status = Resize(minSize);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  ComplexExpressionArrayBuilder(Symbol const& head, arrow::ArrayVector const& fromFields,
                                bool clear = false, int64_t minSize = 0,
                                arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ComplexExpressionArrayBuilder(head, ComplexExpressionArray::makeFields(fromFields.size()),
                                      fromFields, clear, minSize, pool) {}

  ComplexExpressionArrayBuilder(Symbol const& head, arrow::FieldVector const& schema, int64_t size,
                                arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ComplexExpressionArrayBuilder(head, schema, arrow::ArrayVector(schema.size()), true, size,
                                      pool) {}

  ComplexExpressionArrayBuilder(Symbol const& head, std::vector<Symbol> const& columnNames,
                                int64_t size,
                                arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ComplexExpressionArrayBuilder(head, ComplexExpressionArray::makeFields(columnNames),
                                      arrow::ArrayVector(columnNames.size()), true, size, pool) {}

  ComplexExpressionArrayBuilder(Symbol const& head, std::vector<Symbol> const& columnNames,
                                arrow::ArrayVector const& fromFields)
      : ComplexExpressionArrayBuilder(head, ComplexExpressionArray::makeFields(columnNames),
                                      fromFields) {}

  ComplexExpressionArrayBuilder(Symbol const& head, size_t argCount, int64_t size,
                                arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ComplexExpressionArrayBuilder(head, ComplexExpressionArray::makeFields(argCount),
                                      arrow::ArrayVector(argCount), true, size, pool) {}

  explicit ComplexExpressionArrayBuilder(std::shared_ptr<ComplexExpressionArray> const& fromArray,
                                         bool clear = false, int64_t minSize = 0,
                                         arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::StructBuilder(
            static_cast<ComplexExpressionArray::ComplexExpressionArrayType const&>(
                *fromArray->type())
                .storage_type(),
            pool, createChildrenFromExistingColumns(fromArray->fields(), clear, minSize, pool)),
        head(fromArray->getHead()), cachedMissings(fromArray->getMissingHash()) {
    if(!clear) {
      // this is not done by the StructBuilder constructor
      capacity_ = length_ = std::max(minSize, fromArray->length());
      // convert globalIndexes back to builder as well (if used, i.e. preserving order)
      auto const& srcGlobalIndexes = fromArray->getGlobalIndexes();
      if(srcGlobalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(srcGlobalIndexes, minSize, pool);
      }
    } else if(minSize > 0) {
      auto status = Resize(minSize);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  // for initialising a join
  ComplexExpressionArrayBuilder(
      std::shared_ptr<ComplexExpressionArray> const& fromLeftArray,
      std::shared_ptr<ComplexExpressionArray> const& fromRightArray, bool clear = false,
      int64_t minSize = 0, arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::StructBuilder(
            std::make_shared<arrow::StructType>(
                mergeSchemas(fromLeftArray->getSchema(), fromRightArray->getSchema())),
            pool,
            createChildrenFromExistingColumns(
                fromRightArray->fields(), clear, minSize, pool,
                createChildrenFromExistingColumns(fromLeftArray->fields(), clear, minSize, pool))),
        head(fromLeftArray->getHead()),
        cachedMissings(fromLeftArray->getMissingHash() & fromRightArray->getMissingHash()) {
    if(!clear) {
      // this is not done by the StructBuilder constructor
      capacity_ = length_ = std::max(minSize, fromLeftArray->length() + fromRightArray->length());
    } else if(minSize > 0) {
      auto status = Resize(minSize);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  ~ComplexExpressionArrayBuilder() override = default;
  ComplexExpressionArrayBuilder(ComplexExpressionArrayBuilder const& other) = delete;
  ComplexExpressionArrayBuilder(ComplexExpressionArrayBuilder&& other) = delete;
  ComplexExpressionArrayBuilder& operator=(ComplexExpressionArrayBuilder const& other) = delete;
  ComplexExpressionArrayBuilder& operator=(ComplexExpressionArrayBuilder&& other) = delete;

  Symbol const& getHead() const { return head; }

  void setInDestruction() {
    type(); // create cache
    cachedType->setInDestruction();
  }

  std::shared_ptr<arrow::DataType> type() const override {
    if(!cachedType) {
      cachedType = std::make_shared<ComplexExpressionArray::ComplexExpressionArrayType>(
          head, arrow::StructBuilder::type()->fields());
    }
    return cachedType;
  }

  uint64_t getMissingHash() const { return cachedMissings.to_ullong(); }

  bool isMatchingType(ComplexExpression const& expr, uint64_t exprMissingHash) const {
    if(exprMissingHash != getMissingHash()) {
      return false;
    }
    return isMatchingType(expr);
  }
  bool isMatchingType(ComplexExpression const& expr) const {
    auto const& args = expr.getArguments();
    auto size = args.size();
    if(size != num_fields()) {
      return false;
    }
    if(getHead() != expr.getHead()) {
      return false;
    }
    for(int idx = 0; idx < size; ++idx) {
      if(!visit([this, &idx](auto&& arg) { return isMatchingType((Expression const&)arg, idx); },
                args.at(idx))) {
        return false;
      }
    }
    return true;
  }

  bool isMatchingType(std::vector<std::shared_ptr<arrow::Array>> const& argArrays,
                      Symbol const& head, uint64_t exprMissingHash) const {
    if(exprMissingHash != getMissingHash()) {
      return false;
    }
    return isMatchingType(argArrays, head);
  }
  bool isMatchingType(std::vector<std::shared_ptr<arrow::Array>> const& argArrays,
                      Symbol const& head) const {
    if(argArrays.size() != num_fields()) {
      return false;
    }
    if(getHead() != head) {
      return false;
    }
    for(int idx = 0; idx < argArrays.size(); ++idx) {
      auto const& argArray = *argArrays[idx];
      auto childTypeId = child_type_id(idx);
      if(childTypeId < arrow::Type::MAX_ID) {
        if(childTypeId != argArray.type_id()) {
          return false;
        }
        continue;
      }
      if(argArray.type_id() != arrow::Type::EXTENSION) {
        return false;
      }
      if(childTypeId == ArrowTypeExtension::SYMBOL) {
        auto const* argExtensionType =
            dynamic_cast<arrow::ExtensionType const*>(argArray.type().get());
        return argExtensionType && argExtensionType->extension_name()[0] == 's';
      }
      auto const* complexArray = dynamic_cast<ComplexExpressionArray const*>(&argArray);
      if(complexArray == nullptr) {
        return false;
      }
      auto const& complexBuilder =
          static_cast<ComplexExpressionArrayBuilder const&>(*children_[idx]);
      return complexBuilder.isMatchingType(complexArray->fields(), complexArray->getHead());
    }
    return true;
  }

  // conversion from builder to array
  // to avoid this boilerplate in every code (e.g. in the bulk operators)
  explicit operator std::shared_ptr<ComplexExpressionArray>() {
    for(auto& child : children_) {
      // make sure that all children exist even if the builder is empty
      // just set a default array type (any of them works)
      if(!child) {
        child = std::make_shared<ValueBuilder<int64_t>>(0);
      }
    }
    std::shared_ptr<arrow::Array> outputArray;
    auto status = Finish(&outputArray);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    auto cexprArrayPtr = std::static_pointer_cast<ComplexExpressionArray>(outputArray);
    if(globalIndexes) {
      cexprArrayPtr->setGlobalIndexes((ValueArrayPtr<int64_t>)(*globalIndexes));
    }
    cexprArrayPtr->setMissingHash(getMissingHash());
    return std::move(cexprArrayPtr);
  }

  void appendExpression(ComplexExpression const& expr, bool initOrder = false,
                        int64_t startGlobalIndex = 0) {
    // append each argument
    auto const& args = expr.getArguments();
    size_t additionalLength = args.empty() ? 1 : 0;
    auto size = args.size();
    for(int idx = 0; idx < children_.size() && idx < size; ++idx) {
      visit(
          [this, &idx, &additionalLength](auto const& value) {
            using ArgType = std::decay_t<decltype(value)>;
            if constexpr(std::is_same_v<ArgType, std::shared_ptr<std::vector<
                                                     std::shared_ptr<ComplexExpressionArray>>>>) {
              throw std::logic_error(
                  "type not supported for insertion into ComplexExpressionBuilder");
            } else {
              // this will load them as a whole array or a single argument
              // depending what is in the expression
              // (assuming it is consistent along all the arguments!)
              appendToChildBuilder(idx, value, additionalLength);
            }
          },
          args.at(idx));
    }

    // append to the args structure
    auto structStatus = Reserve(additionalLength);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(additionalLength, true);

    // initialise the global indexes (only for order preservation)
    if(initOrder && bulk::Properties::getEnableOrderPreservationCache()) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendConsecutiveValues(*globalIndexes, additionalLength, startGlobalIndex);
    }
  };

  void appendRows(arrow::ArrayVector const& srcColumns, ValueArrayPtr<int64_t> srcGlobalIndexes) {
    size_t numRowsToInsert = !srcColumns.empty() && srcColumns[0] ? srcColumns[0]->length() : 0;
    if(numRowsToInsert == 0) {
      return;
    }
    auto structStatus = Reserve(numRowsToInsert);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(numRowsToInsert, true);
    size_t additionalLength;
    for(size_t idx = 0; idx < srcColumns.size(); ++idx) {
      auto const& srcColumnPtr = srcColumns[idx];
      auto& destColumnPtr = children_[idx];
      switch(child_type_id(idx)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        appendToChildBuilder<bool>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        appendToChildBuilder<int32_t>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        appendToChildBuilder<int64_t>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        appendToChildBuilder<float_t>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        appendToChildBuilder<double_t>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        appendToChildBuilder<std::string>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        appendToChildBuilder<Symbol>(*destColumnPtr, srcColumn, additionalLength);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumn = static_cast<ComplexExpressionArray const&>(*srcColumnPtr);
        appendToChildBuilder(*destColumnPtr, srcTypedColumn, additionalLength);

      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    if(srcGlobalIndexes) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendToChildBuilder(*globalIndexes, *srcGlobalIndexes, additionalLength);
    }
  }

  bool appendRowsWithCondition(ComplexExpressionArray const& srcArray,
                               ValueArray<bool> const& conditionArray) {
    size_t numRowsToInsert = conditionArray.true_count();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    appendRowsWithCondition(srcArray, conditionArray, numRowsToInsert);
    return true;
  }

  void appendRowsWithCondition(ComplexExpressionArray const& srcArray,
                               ValueArray<bool> const& conditionArray, size_t numRowsToInsert) {
    auto structStatus = Reserve(numRowsToInsert);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(numRowsToInsert, true);
    // recursive call for every column
    for(size_t argIndex = 0; argIndex < srcArray.num_fields(); ++argIndex) {
      auto const& srcColumnPtr = srcArray.field(argIndex);
      auto& destColumnPtr = children_[argIndex];
      switch(child_type_id(argIndex)) {
      case arrow::Type::BOOL: {
        auto srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsWithCondition(*srcTypedColumnPtr, conditionArray, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    auto const& srcGlobalIndexes = srcArray.getGlobalIndexes();
    if(srcGlobalIndexes) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendRowsToChildWithCondition(*globalIndexes, *srcGlobalIndexes, conditionArray,
                                     numRowsToInsert);
    }
  }

  bool appendRowsInIndexedOrder(ComplexExpressionArray const& srcArray,
                                std::vector<int64_t> const& rowIndices, bool resetOrder,
                                int64_t startGlobalIndex = 0) {
    size_t numRowsToInsert = rowIndices.size();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    appendRowsInIndexedOrder(srcArray, rowIndices, numRowsToInsert, resetOrder, startGlobalIndex);
    return true;
  }

  bool appendRowsInIndexedOrder(ComplexExpressionArray const& srcArray,
                                ValueArray<int64_t> const& rowIndices, bool resetOrder,
                                int64_t startGlobalIndex = 0) {
    size_t numRowsToInsert = rowIndices.length();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    appendRowsInIndexedOrder(srcArray, rowIndices, numRowsToInsert, resetOrder, startGlobalIndex);
    return true;
  }

  template <typename IndexArrayType>
  void appendRowsInIndexedOrder(ComplexExpressionArray const& srcArray,
                                IndexArrayType const& rowIndices, size_t numRowsToInsert,
                                bool resetOrder, int64_t startGlobalIndex = 0) {
    auto structStatus = Reserve(numRowsToInsert);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(numRowsToInsert, true);
    // recursive call for every column
    for(size_t argIndex = 0; argIndex < srcArray.num_fields(); ++argIndex) {
      auto const& srcColumnPtr = srcArray.field(argIndex);
      auto& destColumnPtr = children_[argIndex];
      switch(child_type_id(argIndex)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rowIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsInIndexedOrder(*srcTypedColumnPtr, rowIndices, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    if(resetOrder && bulk::Properties::getEnableOrderPreservationCache()) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendConsecutiveValues(*globalIndexes, numRowsToInsert, startGlobalIndex);
    } else {
      auto const& srcGlobalIndexes = srcArray.getGlobalIndexes();
      if(srcGlobalIndexes) {
        if(!globalIndexes) {
          globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
        }
        appendRowsToChildInIndexedOrder(*globalIndexes, *srcArray.getGlobalIndexes(), rowIndices,
                                        numRowsToInsert);
      }
    }
  }

  bool joinRowsWithCondition(ComplexExpressionArray const& srcLeftSideArray,
                             ComplexExpressionArray const& srcRightSideArray,
                             ValueArray<bool> const& conditionArray, int64_t startGlobalIndex = 0) {
    size_t numRowsToInsert = conditionArray.true_count();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    joinRowsWithCondition(srcLeftSideArray, srcRightSideArray, conditionArray, numRowsToInsert,
                          startGlobalIndex);
    return true;
  }

  void joinRowsWithCondition(ComplexExpressionArray const& srcLeftSideArray,
                             ComplexExpressionArray const& srcRightSideArray,
                             ValueArray<bool> const& conditionArray, size_t numRowsToInsert,
                             int64_t startGlobalIndex = 0) {
    auto structStatus = Reserve(numRowsToInsert);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(numRowsToInsert, true);
    // recursive call for every left side column
    size_t fieldIndex = 0;
    for(; fieldIndex < srcLeftSideArray.num_fields(); ++fieldIndex) {
      auto const& srcColumnPtr = srcLeftSideArray.field(fieldIndex);
      auto& destColumnPtr = children_[fieldIndex];
      switch(child_type_id(fieldIndex)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsWithCondition(*srcTypedColumnPtr, conditionArray, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    // then same but for right side columns
    for(size_t argIndex = 0; argIndex < srcRightSideArray.num_fields(); ++argIndex, ++fieldIndex) {
      auto const& srcColumnPtr = srcRightSideArray.field(argIndex);
      auto& destColumnPtr = children_[fieldIndex];
      switch(child_type_id(fieldIndex)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildWithCondition(destColumn, srcColumn, conditionArray, numRowsToInsert);
      }
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsWithCondition(*srcTypedColumnPtr, conditionArray, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    if(bulk::Properties::getEnableOrderPreservationCacheForJoins()) {
      auto const& leftSideGlobalIndexes = srcLeftSideArray.getGlobalIndexes();
      auto const& rightSideGlobalIndexes = srcRightSideArray.getGlobalIndexes();
      if(leftSideGlobalIndexes && rightSideGlobalIndexes) {
        if(!globalIndexes) {
          globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
        }
        auto prevLength = globalIndexes->length();
        appendRowsToChildWithCondition(*globalIndexes, *leftSideGlobalIndexes, conditionArray,
                                       numRowsToInsert);
        auto leftSideMaxGlobalIndex = srcLeftSideArray.getTablePartitionIndexes()->length();
        auto const* rightSideIndexValues = rightSideGlobalIndexes->rawData();
        auto* globalIndexesValues = &((*globalIndexes)[prevLength]);
        for(size_t index = 0; index < numRowsToInsert; ++index) {
          globalIndexesValues[index] += rightSideIndexValues[index] * leftSideMaxGlobalIndex;
        }
      }
    } else if(bulk::Properties::getEnableOrderPreservationCache()) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendConsecutiveValues(*globalIndexes, numRowsToInsert, startGlobalIndex);
    }
  }

  bool joinRowsInIndexedOrder(ComplexExpressionArray const& srcLeftSideArray,
                              std::vector<int64_t> const& leftSideIndices,
                              ComplexExpressionArray const& srcRightSideArray,
                              std::vector<int64_t> const& rightSideIndices,
                              int64_t startGlobalIndex = 0) {
    size_t numRowsToInsert = leftSideIndices.size();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    joinRowsInIndexedOrder(srcLeftSideArray, leftSideIndices, srcRightSideArray, rightSideIndices,
                           numRowsToInsert, startGlobalIndex);
    return true;
  }

  bool joinRowsInIndexedOrder(ComplexExpressionArray const& srcLeftSideArray,
                              ValueArray<int64_t> const& leftSideIndices,
                              ComplexExpressionArray const& srcRightSideArray,
                              ValueArray<int64_t> const& rightSideIndices,
                              int64_t startGlobalIndex = 0) {
    size_t numRowsToInsert = leftSideIndices.length();
    if(numRowsToInsert == 0) {
      return false; // nothing to do
    }
    joinRowsInIndexedOrder(srcLeftSideArray, leftSideIndices, srcRightSideArray, rightSideIndices,
                           numRowsToInsert, startGlobalIndex);
    return true;
  }

  template <typename IndexArrayType>
  void joinRowsInIndexedOrder(ComplexExpressionArray const& srcLeftSideArray,
                              IndexArrayType const& leftSideIndices,
                              ComplexExpressionArray const& srcRightSideArray,
                              IndexArrayType const& rightSideIndices, size_t numRowsToInsert,
                              int64_t startGlobalIndex = 0) {
    auto structStatus = Reserve(numRowsToInsert);
    if(!structStatus.ok()) {
      throw std::runtime_error(structStatus.ToString());
    }
    UnsafeAppendToBitmap(numRowsToInsert, true);
    // recursive call for every left side column
    size_t fieldIndex = 0;
    for(; fieldIndex < srcLeftSideArray.num_fields(); ++fieldIndex) {
      auto const& srcColumnPtr = srcLeftSideArray.field(fieldIndex);
      auto& destColumnPtr = children_[fieldIndex];
      switch(child_type_id(fieldIndex)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, leftSideIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsInIndexedOrder(*srcTypedColumnPtr, leftSideIndices, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    // then same but for right side columns
    for(size_t argIndex = 0; argIndex < srcRightSideArray.num_fields(); ++argIndex, ++fieldIndex) {
      auto const& srcColumnPtr = srcRightSideArray.field(argIndex);
      auto& destColumnPtr = children_[fieldIndex];
      switch(child_type_id(fieldIndex)) {
      case arrow::Type::BOOL: {
        auto const& srcColumn = ValueArray<bool>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<bool>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT32: {
        auto const& srcColumn = ValueArray<int32_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int32_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::INT64: {
        auto const& srcColumn = ValueArray<int64_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<int64_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::FLOAT: {
        auto const& srcColumn = ValueArray<float_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<float_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::DOUBLE: {
        auto const& srcColumn = ValueArray<double_t>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<double_t>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case arrow::Type::STRING: {
        auto const& srcColumn = ValueArray<std::string>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<std::string>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::SYMBOL: {
        auto const& srcColumn = ValueArray<Symbol>(*srcColumnPtr);
        auto& destColumn = static_cast<ValueBuilder<Symbol>&>(*destColumnPtr);
        appendRowsToChildInIndexedOrder(destColumn, srcColumn, rightSideIndices, numRowsToInsert);
      } break;
      case ArrowTypeExtension::COMPLEX_EXPRESSION: {
        auto const& srcTypedColumnPtr =
            std::static_pointer_cast<ComplexExpressionArray>(srcColumnPtr);
        auto& destColumn = static_cast<ComplexExpressionArrayBuilder&>(*destColumnPtr);
        destColumn.appendRowsInIndexedOrder(*srcTypedColumnPtr, rightSideIndices, numRowsToInsert);
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    if(bulk::Properties::getEnableOrderPreservationCacheForJoins()) {
      auto const& leftSideGlobalIndexes = srcLeftSideArray.getGlobalIndexes();
      auto const& rightSideGlobalIndexes = srcRightSideArray.getGlobalIndexes();
      if(leftSideGlobalIndexes && rightSideGlobalIndexes) {
        if(!globalIndexes) {
          globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
        }
        auto prevLength = globalIndexes->length();
        appendRowsToChildInIndexedOrder(*globalIndexes, *leftSideGlobalIndexes, leftSideIndices,
                                        numRowsToInsert);
        auto leftSideMaxGlobalIndex = srcLeftSideArray.getTablePartitionIndexes()->length();
        auto const* rightSideIndexValues = rightSideGlobalIndexes->rawData();
        auto* globalIndexesValues = &((*globalIndexes)[prevLength]);
        for(size_t index = 0; index < numRowsToInsert; ++index) {
          globalIndexesValues[index] += rightSideIndexValues[index] * leftSideMaxGlobalIndex;
        }
      }
    } else if(bulk::Properties::getEnableOrderPreservationCache()) {
      if(!globalIndexes) {
        globalIndexes = std::make_shared<ValueBuilder<int64_t>>(capacity(), 0, pool_);
      }
      appendConsecutiveValues(*globalIndexes, numRowsToInsert, startGlobalIndex);
    }
  }

  void setGlobalIndexes(ValueArrayPtr<int64_t> indexes) {
    if(indexes) {
      globalIndexes = std::make_shared<ValueBuilder<int64_t>>(indexes);
    } else {
      globalIndexes = nullptr;
    }
  }

  enum ArrowTypeExtension {
    SYMBOL = arrow::Type::MAX_ID,
    COMPLEX_EXPRESSION,
  };

  int child_type_id(size_t idx) const {
    if(cachedChildrenTypeIds.empty()) {
      cachedChildrenTypeIds.reserve(children_.size());
      std::transform(children_.begin(), children_.end(), std::back_inserter(cachedChildrenTypeIds),
                     [](auto const& child) -> int {
                       if(!child) {
                         return -1;
                       }
                       auto const& type = child->type();
                       if(type->id() != arrow::Type::EXTENSION) {
                         return type->id();
                       }
                       auto const& extensionType = static_cast<arrow::ExtensionType const&>(*type);
                       if(extensionType.extension_name()[0] == 'c') {
                         return ArrowTypeExtension::COMPLEX_EXPRESSION;
                       }
                       return ArrowTypeExtension::SYMBOL;
                     });
    }
    return cachedChildrenTypeIds[idx];
  }

private:
  Symbol head;
  std::shared_ptr<ComplexExpressionArray::ComplexExpressionArrayType> mutable cachedType;
  std::vector<int> mutable cachedChildrenTypeIds;
  std::bitset<64> cachedMissings; // for fast pruning IsMatchingType

  // only used for order preservation
  std::shared_ptr<ValueBuilder<int64_t>> globalIndexes;

  bool isMatchingType(Expression const& expr, size_t childIndex) const {
    auto childTypeId = child_type_id(childIndex);
    return std::visit(
        boss::utilities::overload(
            [&childTypeId](bool /*v*/) { return childTypeId == arrow::Type::BOOL; },
            [&childTypeId](int32_t /*v*/) { return childTypeId == arrow::Type::INT32; },
            [&childTypeId](int64_t /*v*/) { return childTypeId == arrow::Type::INT64; },
            [&childTypeId](float_t /*v*/) { return childTypeId == arrow::Type::FLOAT; },
            [&childTypeId](double_t /*v*/) { return childTypeId == arrow::Type::DOUBLE; },
            [&childTypeId](std::string const& /*v*/) { return childTypeId == arrow::Type::STRING; },
            [&childTypeId](Symbol const& /*s*/) {
              return childTypeId == ArrowTypeExtension::SYMBOL;
            },
            [&childTypeId, &childIndex, this](ComplexExpression const& e) {
              if(childTypeId != ArrowTypeExtension::COMPLEX_EXPRESSION) {
                return false;
              }
              auto const& complexBuilder =
                  static_cast<ComplexExpressionArrayBuilder const&>(*children_[childIndex]);
              return complexBuilder.isMatchingType(e);
            },
            [](std::shared_ptr<std::vector<std::shared_ptr<ComplexExpressionArray>>> const&
               /*partitions*/) {
              throw std::logic_error("this type is not supported for insertion");
              return false;
            },
            [&childTypeId, &childIndex, this](auto const& arrayPtr) {
              if(childTypeId < arrow::Type::MAX_ID) {
                return childTypeId == arrayPtr->type_id();
              }
              if(arrayPtr->type_id() != arrow::Type::EXTENSION) {
                return false;
              }
              if(childTypeId == ArrowTypeExtension::SYMBOL) {
                auto const* argExtensionType =
                    dynamic_cast<arrow::ExtensionType const*>(arrayPtr->type().get());
                return argExtensionType && argExtensionType->extension_name()[0] == 's';
              }
              auto const* complexArray =
                  dynamic_cast<ComplexExpressionArray const*>(arrayPtr.get());
              if(complexArray == nullptr) {
                return false;
              }
              auto const& complexBuilder =
                  static_cast<ComplexExpressionArrayBuilder const&>(*children_[childIndex]);
              return complexBuilder.isMatchingType(complexArray->fields(), complexArray->getHead());
            }),
        expr);
  }

  // re-implement FinishInternal for performance: do not shrink_to_fit the buffers
  arrow::Status FinishInternal(std::shared_ptr<arrow::ArrayData>* out) override {
    std::shared_ptr<arrow::Buffer> null_bitmap;
    RETURN_NOT_OK(null_bitmap_builder_.Finish(&null_bitmap, false));

    std::vector<std::shared_ptr<arrow::ArrayData>> child_data(children_.size());
    for(size_t i = 0; i < children_.size(); ++i) {
      if(length_ == 0) {
        // Try to make sure the child buffers are initialized
        RETURN_NOT_OK(children_[i]->Resize(0));
      }
      RETURN_NOT_OK(children_[i]->FinishInternal(&child_data[i]));
    }

    *out = arrow::ArrayData::Make(type(), length_, {null_bitmap}, null_count_);
    (*out)->child_data = std::move(child_data);

    capacity_ = length_ = null_count_ = 0;
    return arrow::Status::OK();
  }

  static arrow::FieldVector mergeSchemas(arrow::FieldVector const& leftSchema,
                                         arrow::FieldVector const& rightSchema) {
    arrow::FieldVector schema;
    schema.reserve(leftSchema.size() + rightSchema.size());
    schema.insert(schema.end(), leftSchema.begin(), leftSchema.end());
    schema.insert(schema.end(), rightSchema.begin(), rightSchema.end());
    return schema;
  }

  static std::vector<std::shared_ptr<arrow::ArrayBuilder>> createChildrenFromExistingColumns(
      arrow::ArrayVector const& fromColumns, bool clear, int64_t minSize, arrow::MemoryPool* pool,
      std::vector<std::shared_ptr<arrow::ArrayBuilder>>&& children = {}) {
    children.reserve(fromColumns.size());
    std::transform(
        fromColumns.begin(), fromColumns.end(), std::back_inserter(children),
        [&clear, &minSize, &pool](auto const& columnPtr) -> std::shared_ptr<arrow::ArrayBuilder> {
          if(!columnPtr) {
            return nullptr;
          }
          switch(columnPtr->type_id()) {
          case arrow::Type::BOOL: {
            if(clear) {
              return std::make_shared<ValueBuilder<bool>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<bool>>(columnPtr, minSize, pool);
          }
          case arrow::Type::INT32: {
            if(clear) {
              return std::make_shared<ValueBuilder<int32_t>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<int32_t>>(columnPtr, minSize, pool);
          }
          case arrow::Type::INT64: {
            if(clear) {
              return std::make_shared<ValueBuilder<int64_t>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<int64_t>>(columnPtr, minSize, pool);
          }
          case arrow::Type::FLOAT: {
            if(clear) {
              return std::make_shared<ValueBuilder<float_t>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<float_t>>(columnPtr, minSize, pool);
          }
          case arrow::Type::DOUBLE: {
            if(clear) {
              return std::make_shared<ValueBuilder<double_t>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<double_t>>(columnPtr, minSize, pool);
          }
          case arrow::Type::STRING: {
            if(clear) {
              return std::make_shared<ValueBuilder<std::string>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<std::string>>(columnPtr, minSize, pool);
          }
          case arrow::Type::EXTENSION: {
            auto const& extensionType =
                static_cast<arrow::ExtensionType const&>(*columnPtr->type());
            if(extensionType.extension_name()[0] == 'c') {
              // complex array
              return std::make_shared<ComplexExpressionArrayBuilder>(
                  std::static_pointer_cast<ComplexExpressionArray>(columnPtr), clear, minSize,
                  pool);
            }
            // symbol array
            if(clear) {
              return std::make_shared<ValueBuilder<Symbol>>(minSize, 0, pool);
            }
            return std::make_shared<ValueBuilder<Symbol>>(columnPtr, minSize, pool);
            break;
          }
          default:
            throw std::logic_error("unsupported arrow array type");
            break;
          }
        });
    return std::move(children);
  }

  template <typename T>
  void appendToChildBuilder(size_t index, T const& value, size_t& outNumInsertedRows) {
    outNumInsertedRows = 1;
    auto& childBuilderPtr = children_[index];
    if(!childBuilderPtr) {
      childBuilderPtr = std::make_shared<ValueBuilder<T>>(capacity(), 0, pool_);
      cachedMissings[index] = false;
    }
    static_cast<ValueBuilder<T>&>(*childBuilderPtr).Append(value);
  }

  void appendToChildBuilder(size_t index, Symbol const& value, size_t& outNumInsertedRows) {
    outNumInsertedRows = 1;
    auto& childBuilderPtr = children_[index];
    if(!childBuilderPtr) {
      childBuilderPtr = std::make_shared<ValueBuilder<Symbol>>(capacity(), 0, pool_);
      cachedMissings[index] = true;
    }
    static_cast<ValueBuilder<Symbol>&>(*childBuilderPtr).Append(value.getName());
  }

  void appendToChildBuilder(size_t index, ComplexExpression const& expr,
                            size_t& outNumInsertedRows) {
    auto& childBuilderPtr = children_[index];
    if(!childBuilderPtr) {
      auto const& head = expr.getHead();
      auto argCount = expr.getArguments().size();
      childBuilderPtr =
          std::make_shared<ComplexExpressionArrayBuilder>(head, argCount, capacity(), pool_);
      cachedMissings[index] = true;
    }
    auto& childComplexBuilder = static_cast<ComplexExpressionArrayBuilder&>(*childBuilderPtr);
    auto prevCount = childComplexBuilder.length();
    childComplexBuilder.appendExpression(expr);
    outNumInsertedRows = childComplexBuilder.length() - prevCount;
  }

  void appendToChildBuilder(size_t index,
                            std::shared_ptr<ComplexExpressionArray> const& complexArrayPtr,
                            size_t& outNumInsertedRows) {
    outNumInsertedRows = complexArrayPtr->length();
    auto& childBuilderPtr = children_[index];
    if(!childBuilderPtr) {
      // to avoid copy, just convert the source array data back to a builder
      childBuilderPtr = std::make_shared<ComplexExpressionArrayBuilder>(complexArrayPtr, false,
                                                                        capacity(), pool_);
      cachedMissings[index] = true;
      return;
    }
    appendToChildBuilder(*childBuilderPtr, *complexArrayPtr, outNumInsertedRows);
  }

  void appendToChildBuilder(arrow::ArrayBuilder& childBuilder,
                            ComplexExpressionArray const& complexArray,
                            size_t& outNumInsertedRows) {
    auto const& head = complexArray.getHead();
    auto const& columns = complexArray.fields();
    ExpressionArguments args;
    args.reserve(columns.size());
    std::transform(columns.begin(), columns.end(), std::back_inserter(args),
                   [](auto const& column) { return genericArrowArrayToBulkExpression(column); });
    static_cast<ComplexExpressionArrayBuilder&>(childBuilder)
        .appendExpression(ComplexExpression(head, std::move(args)));
  }

  template <typename T>
  void appendToChildBuilder(size_t index, ValueArrayPtr<T> const& srcArrayPtr,
                            size_t& outNumInsertedRows) {
    auto const& srcArray = *srcArrayPtr;
    outNumInsertedRows = srcArray.length();
    auto& childBuilderPtr = children_[index];
    if(!childBuilderPtr) {
      // to avoid copy, just convert the source array data back to a builder
      childBuilderPtr = std::make_shared<ValueBuilder<T>>(srcArrayPtr, capacity(), pool_);
      cachedMissings[index] = false;
      return;
    }
    // otherwise, if data already exist, we need to resize and merge
    appendToChildBuilder(*childBuilderPtr, srcArray, outNumInsertedRows);
  }

  template <typename T>
  void appendToChildBuilder(arrow::ArrayBuilder& childBuilder, ValueArray<T> const& srcArray,
                            size_t& outNumInsertedRows) {
    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
      auto const& valueDataLength = srcArray.value_data()->size();
      auto& typedDestBuilder = static_cast<ValueBuilder<T>&>(childBuilder);
      typedDestBuilder.Reserve(outNumInsertedRows);
      typedDestBuilder.ReserveData(valueDataLength);
      for(size_t i = 0; i < outNumInsertedRows; ++i) {
        typedDestBuilder.UnsafeAppend(srcArray.GetView(i));
      }
    } else {
      auto const* srcValues = srcArray.rawData();
      auto srcLength = srcArray.rawLength();
      static_cast<ValueBuilder<T>&>(childBuilder).AppendValues(srcValues, srcLength);
    }
  }

  template <typename T>
  static void
  appendRowsToChildWithCondition(arrow::ArrayBuilder& childBuilder, ValueArray<T> const& srcArray,
                                 ValueArray<bool> const& conditionArray, size_t numRowsToInsert) {
    auto& destBuilder = static_cast<ValueBuilder<T>&>(childBuilder);
    auto prevLength = destBuilder.length();
    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
      destBuilder.Reserve(numRowsToInsert);
    } else {
      auto status = destBuilder.AppendEmptyValues(numRowsToInsert);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
    int64_t destIndex = 0;
    auto const* condition = conditionArray.values()->data();
    constexpr auto bitsPerPack = (int)sizeof(*condition) * CHAR_BIT;
    auto const packedLength = conditionArray.length() / bitsPerPack;
    int64_t packIndex = 0;
    for(; packIndex < packedLength; ++packIndex) {
      if(condition[packIndex] != 0U) {
        for(auto i = 0U; i < bitsPerPack; ++i) {
          auto rowIndex = packIndex * bitsPerPack;
          if((condition[packIndex] & (1U << i)) != 0U) {
            auto const& srcValue = srcArray.Value(rowIndex + i);
            if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
              destBuilder.ReserveData(srcValue.length());
              destBuilder.UnsafeAppend(srcValue);
            } else {
              destBuilder[prevLength + destIndex] = srcValue;
            }
            ++destIndex;
          }
        }
      }
    }
    auto rowIndex = packIndex * bitsPerPack;
    if(conditionArray.length() > rowIndex) {
      if(condition[packIndex] != 0U) {
        auto remaining = conditionArray.length() - rowIndex;
        for(auto i = 0U; i < remaining; ++i) {
          if((condition[packIndex] & (1U << i)) != 0U) {
            auto const& srcValue = srcArray.Value(rowIndex + i);
            if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
              destBuilder.ReserveData(srcValue.length());
              destBuilder.UnsafeAppend(srcValue);
            } else {
              destBuilder[prevLength + destIndex] = srcValue;
            }
            ++destIndex;
          }
        }
      }
    }
  }

  static void appendRowsToChildWithCondition(arrow::ArrayBuilder& childBuilder,
                                             ValueArray<bool> const& srcArray,
                                             ValueArray<bool> const& conditionArray,
                                             size_t numRowsToInsert) {
    auto& destBuilder = static_cast<ValueBuilder<bool>&>(childBuilder);
    auto prevLength = destBuilder.length();
    auto status = destBuilder.AppendEmptyValues(numRowsToInsert);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    destBuilder.appendWithCondition(srcArray, conditionArray, prevLength);
  }

  template <typename T, typename IndexArrayType>
  static std::enable_if_t<!(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>), void>
  appendRowsToChildInIndexedOrder(arrow::ArrayBuilder& childBuilder, ValueArray<T> const& srcArray,
                                  IndexArrayType const& rowIndices, size_t numRowsToInsert) {
    auto& destBuilder = static_cast<ValueBuilder<T>&>(childBuilder);
    auto prevLength = destBuilder.length();
    auto status = destBuilder.AppendEmptyValues(numRowsToInsert);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    for(size_t destIndex = 0; destIndex < numRowsToInsert; ++destIndex) {
      auto srcIndex = rowIndices[destIndex];
      auto const& srcValue = srcArray.Value(srcIndex);
      destBuilder[prevLength + destIndex] = srcValue;
    }
  }

  template <typename T, typename IndexArrayType>
  static std::enable_if_t<std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>, void>
  appendRowsToChildInIndexedOrder(arrow::ArrayBuilder& childBuilder, ValueArray<T> const& srcArray,
                                  IndexArrayType const& rowIndices, size_t numRowsToInsert) {
    auto& destBuilder = static_cast<ValueBuilder<T>&>(childBuilder);
    destBuilder.Reserve(numRowsToInsert);
    for(size_t destIndex = 0; destIndex < numRowsToInsert; ++destIndex) {
      auto srcIndex = rowIndices[destIndex];
      auto const& srcValue = srcArray.Value(srcIndex);
      destBuilder.ReserveData(srcValue.length());
      destBuilder.UnsafeAppend(srcValue);
    }
  }

  template <typename IndexArrayType>
  static void appendRowsToChildInIndexedOrder(arrow::ArrayBuilder& childBuilder,
                                              ValueArray<bool> const& srcArray,
                                              IndexArrayType const& rowIndices,
                                              size_t numRowsToInsert) {
    auto& destBuilder = static_cast<ValueBuilder<bool>&>(childBuilder);
    auto prevLength = destBuilder.length();
    auto status = destBuilder.AppendEmptyValues(numRowsToInsert);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    destBuilder.appendInIndexedOrder(srcArray, rowIndices, prevLength);
  }

  template <typename T>
  static std::enable_if_t<!(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>), void>
  appendConsecutiveValues(ValueBuilder<T>& destBuilder, size_t numRowsToInsert,
                          int64_t startIndex) {
    auto prevLength = destBuilder.length();
    auto status = destBuilder.AppendEmptyValues(numRowsToInsert);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    auto* rawValues = &destBuilder[prevLength];
    std::iota(rawValues, rawValues + numRowsToInsert, startIndex);
  }
};

} // namespace boss::engines::bulk