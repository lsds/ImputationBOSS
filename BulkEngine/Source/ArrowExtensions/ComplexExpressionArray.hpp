#pragma once
#include "../BulkExpression.hpp"
#include "../SymbolRegistry.hpp"
#include <arrow/array.h>
#include <arrow/extension_type.h>

#include <bitset>

namespace boss::engines::bulk {

/** We use ComplexExpressionArray to store complex expressions.
 * This is an extension of the StructArray.
 * The children array are the arguments.
 * We store the head symbol as a metadata to the custom type. */
class ComplexExpressionArray : public arrow::StructArray {
public:
  ComplexExpressionArray(std::shared_ptr<arrow::ArrayData> const& data, Symbol const& head,
                         bool doInit = true)
      : arrow::StructArray(data) {
    // make sure to set back the extension type after the end of call from base array class
    auto adjustedData = data->Copy();
    adjustedData->type = std::make_shared<ComplexExpressionArrayType>(head, data->type->fields());
    SetData(adjustedData);

    if(doInit) {
      // cache the column symbols in advance
      // (to speed up symbol registration for the relational ops)
      initSymbolRegister();
    }
  }

  ComplexExpressionArray(Symbol const& head, arrow::FieldVector const& schema,
                         arrow::ArrayVector const& fromFields)
      : arrow::StructArray(std::make_shared<arrow::StructType>(schema), fromFields[0]->length(),
                           fromFields, std::make_shared<arrow::Buffer>(nullptr, 0), 0, 0),
        missingHash(computeMissingHash(fromFields)) {
    // make sure to set back the extension type after the end of call from base array class
    auto adjustedData = data()->Copy();
    adjustedData->type = std::make_shared<ComplexExpressionArrayType>(head, data()->type->fields());
    SetData(adjustedData);

    // cache the column symbols in advance (to speed up symbol registration for the relational ops)
    initSymbolRegister();
  }

  ComplexExpressionArray(Symbol const& head, arrow::ArrayVector const& fromFields)
      : ComplexExpressionArray(head, makeFields(fromFields.size()), fromFields) {}

  ComplexExpressionArray(Symbol const& head, std::vector<Symbol> const& columnNames,
                         arrow::ArrayVector const& fromFields)
      : ComplexExpressionArray(head, makeFields(columnNames), fromFields) {}

  ~ComplexExpressionArray() {
    if(symbolRegistered) {
      throw std::logic_error(
          "destroying ComplexExpressionArray without calling unregisterSymbols()");
    }
  }

  arrow::FieldVector const& getSchema() const {
    auto const& extensionType = static_cast<arrow::ExtensionType const&>(*type());
    return extensionType.storage_type()->fields();
  }

  Symbol const& getHead() const {
    return static_cast<ComplexExpressionArray::ComplexExpressionArrayType const&>(*type())
        .getHead();
  }

  Expression column(size_t index) const { return genericArrowArrayToBulkExpression(field(index)); }

  Expression row(size_t index) const {
    auto rowAsListofList =
        (boss::Expression) static_cast<ComplexExpressionArray const&>(*Slice(index, 1));
    auto complexExpr = get<boss::ComplexExpression>(std::move(rowAsListofList));
    auto& bossExpr = get<boss::Expression>(complexExpr.getArguments().at(0));
    return std::move(bossExpr);
  }

  explicit operator boss::Expression() const {
    // convert each array of data to a complex expression
    std::vector<boss::ExpressionArguments> columnarArgs;
    columnarArgs.reserve(num_fields());
    for(auto const& arrowArrayPtr : fields()) {
      switch(arrowArrayPtr->type_id()) {
      case arrow::Type::BOOL: {
        auto arrayPtr = std::static_pointer_cast<arrow::BooleanArray>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<bool>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::INT32:
      case arrow::Type::DATE32: {
        auto arrayPtr = std::static_pointer_cast<arrow::Int32Array>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<int32_t>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::INT64:
      case arrow::Type::DATE64: {
        auto arrayPtr = std::static_pointer_cast<arrow::Int64Array>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<int64_t>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::FLOAT: {
        auto arrayPtr = std::static_pointer_cast<arrow::FloatArray>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<float_t>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::DOUBLE: {
        auto arrayPtr = std::static_pointer_cast<arrow::DoubleArray>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<double_t>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::STRING: {
        auto arrayPtr = std::static_pointer_cast<arrow::StringArray>(arrowArrayPtr);
        auto expr = (boss::Expression)(ValueArrayPtr<std::string>)arrayPtr;
        auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
        columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                  std::make_move_iterator(argsToInsert.end()));
      } break;
      case arrow::Type::EXTENSION: {
        auto const& extensionType =
            static_cast<arrow::ExtensionType const&>(*arrowArrayPtr->type());
        if(extensionType.extension_name()[0] == 'c') {
          auto arrayPtr = std::static_pointer_cast<ComplexExpressionArray>(arrowArrayPtr);
          auto expr = (boss::Expression)*arrayPtr;
          auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
          columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                    std::make_move_iterator(argsToInsert.end()));
        } else {
          auto arrayPtr = std::static_pointer_cast<arrow::StringArray>(arrowArrayPtr);
          auto expr = (boss::Expression)(ValueArrayPtr<Symbol>)arrayPtr;
          auto argsToInsert = get<boss::ComplexExpression>(expr).getArguments();
          columnarArgs.emplace_back(std::make_move_iterator(argsToInsert.begin()),
                                    std::make_move_iterator(argsToInsert.end()));
        }
      } break;
      default:
        throw std::logic_error("unsupported arrow array type");
        break;
      }
    }
    // transpose from decomposed to n-ary representation
    auto const& head = getHead();
    auto numRows = length();
    boss::ExpressionArguments rowArgs;
    rowArgs.reserve(numRows);
    for(int rowIndex = 0; rowIndex < numRows; ++rowIndex) {
      boss::ExpressionArguments row;
      row.reserve(columnarArgs.size());
      std::transform(std::make_move_iterator(columnarArgs.begin()),
                     std::make_move_iterator(columnarArgs.end()), std::back_inserter(row),
                     [&rowIndex](auto&& column) { return std::move(column[rowIndex]); });
      rowArgs.emplace_back(boss::ComplexExpression(head, std::move(row)));
    }
    return boss::ComplexExpression("List"_, std::move(rowArgs));
  }

  void initSymbolRegister() {
    symbolRegistry = &DefaultSymbolRegistry::instance();
    auto const& schema = getSchema();
    symbolsToRegister.clear();
    symbolsToRegister.reserve(schema.size());
    for(int i = 0; i < schema.size(); ++i) {
      auto const& symbol = Symbol(schema[i]->name());
      auto& whereToRegister = symbolRegistry->findOrCreateSymbolReference(symbol);
      auto columnExprPtr = std::make_unique<Expression>(column(i));
      symbolsToRegister.emplace_back(std::make_pair(&whereToRegister, std::move(columnExprPtr)));
    }
  }

  void registerSymbols() {
    if(symbolRegistered) {
      throw std::logic_error("calling registerSymbols() twice without calling unregisterSymbols()");
    }
    if(symbolRegistry == nullptr ||
       (symbolRegistry != &DefaultSymbolRegistry::globalInstance() &&
        symbolRegistry->ownerThreadId() != DefaultSymbolRegistry::instance().ownerThreadId())) {
      initSymbolRegister();
    }
    for(auto& [registerExpressionPtr, arrayPtrExpr] : symbolsToRegister) {
      std::swap(*registerExpressionPtr, arrayPtrExpr);
    }
    symbolRegistered = true;
  }

  void unregisterSymbols() {
    if(!symbolRegistered) {
      throw std::logic_error("calling unregisterSymbols() without matching registerSymbols()");
    }
    for(auto& [registerExpressionPtr, arrayPtrExpr] : symbolsToRegister) {
      std::swap(*registerExpressionPtr, arrayPtrExpr);
    }
    symbolRegistered = false;
  }

  static arrow::FieldVector makeFields(size_t argCount) {
    return arrow::FieldVector(argCount, std::make_shared<arrow::Field>("", nullptr));
  }

  static arrow::FieldVector makeFields(std::vector<Symbol> const& columnNames) {
    auto fields = arrow::FieldVector();
    fields.reserve(columnNames.size());
    std::transform(columnNames.begin(), columnNames.end(), std::back_inserter(fields),
                   [](auto const& symbol) {
                     return std::make_shared<arrow::Field>(symbol.getName(), nullptr);
                   });
    return fields;
  }

  auto const& getGlobalIndexes() const { return globalIndexes; }
  void setGlobalIndexes(ValueArrayPtr<int64_t> indexes) { globalIndexes = std::move(indexes); }

  auto const& getTablePartitionIndexes() const { return *tablePartitionIndexes; }
  auto const& getTablePartitionLocalIndexes() const { return *tablePartitionLocalIndexes; }
  void setTablePartitionIndexes(std::shared_ptr<ValueArrayPtr<int64_t>> indexes) {
    tablePartitionIndexes = std::move(indexes);
  }
  void setTablePartitionLocalIndexes(std::shared_ptr<ValueArrayPtr<int64_t>> indexes) {
    tablePartitionLocalIndexes = std::move(indexes);
  }

  void setTablePartitionIndexes(ComplexExpressionArray const& otherPartition) {
    setTablePartitionIndexes(otherPartition.tablePartitionIndexes);
    setTablePartitionLocalIndexes(otherPartition.tablePartitionLocalIndexes);
  }

  static uint64_t computeMissingHash(arrow::ArrayVector const& columns) {
    std::bitset<64> missingBitset;
    for(int idx = 0; idx < columns.size(); ++idx) {
      auto const& column = columns[idx];
      if(column && column->type_id() == arrow::Type::EXTENSION) {
        missingBitset[idx] = true;
      }
    }
    return missingBitset.to_ullong();
  }

  uint64_t getMissingHash() const { return missingHash; }
  void setMissingHash(uint64_t hash) { missingHash = hash; }

  /** Custom type to implement an Arrow array for complex expressions.
   * This is mostly boilerplate code to be compliant with Arrow.
   * It also stores an additional metadata for the head symbol. */
  class ComplexExpressionArrayType : public arrow::ExtensionType {
  public:
    explicit ComplexExpressionArrayType(Symbol const& head, arrow::FieldVector const& fields)
        : ExtensionType(arrow::struct_(fields)), head(head), doInit(true) {}

    Symbol const& getHead() const { return head; }

    void setInDestruction() { doInit = false; }

    /// Called by Arrow to create our custom ComplexExpressionArray from a
    /// ComplexExpressionArrayBuilder
    std::shared_ptr<arrow::Array> MakeArray(std::shared_ptr<arrow::ArrayData> data) const override {
      // temporarly change to the underline type for the construction
      // it will be reverted in the ComplexExpressionArray constructor
      auto adjustedData = data->Copy();
      adjustedData->type = arrow::struct_(storage_type()->fields());
      return std::make_shared<ComplexExpressionArray>(adjustedData, head, doInit);
    }

    ///////////////////////////////////////////////////////////////////////
    // code required by Arrow to implement an extension type
    std::string extension_name() const override { return "complex-expr-type"; }
    bool ExtensionEquals(ExtensionType const& other) const override {
      auto const& other_ext = static_cast<ExtensionType const&>(other);
      if(other_ext.extension_name() != this->extension_name()) {
        return false;
      }
      return this->getHead().getName() ==
             static_cast<ComplexExpressionArrayType const&>(other).getHead().getName();
    }
    arrow::Result<std::shared_ptr<DataType>>
    Deserialize(std::shared_ptr<DataType> storage_type,
                std::string const& serialized) const override {
      return std::make_shared<ComplexExpressionArrayType>(Symbol(serialized),
                                                          storage_type->fields());
    }
    std::string Serialize() const override { return head.getName(); }
    ///////////////////////////////////////////////////////////////////////

  private:
    Symbol head;
    bool doInit;
  };

private:
  // cache accesses for symbol registration
  using RegisterExpressionPtr = std::unique_ptr<Expression>*;
  std::vector<std::pair<RegisterExpressionPtr, std::unique_ptr<Expression>>> symbolsToRegister;
  bool symbolRegistered = false;
  DefaultSymbolRegistry* symbolRegistry = nullptr;

  // only used for order preservation
  ValueArrayPtr<int64_t> globalIndexes;
  std::shared_ptr<ValueArrayPtr<int64_t>> tablePartitionIndexes;
  std::shared_ptr<ValueArrayPtr<int64_t>> tablePartitionLocalIndexes;

  // for fast pruning IsMatchingType
  uint64_t missingHash;
};

} // namespace boss::engines::bulk