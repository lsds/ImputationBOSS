#include "BulkExpression.hpp"
#include "ArrowExtensions/ComplexExpressionArray.hpp"

#include <arrow/extension_type.h>
#include <arrow/type_fwd.h>

namespace boss::engines::bulk {

Expression genericArrowArrayToBulkExpression(std::shared_ptr<arrow::Array> const& arrowArrayPtr) {
  auto const& type = arrowArrayPtr->type();
  switch(type->id()) {
  case arrow::Type::BOOL:
    return (ValueArrayPtr<bool>)std::static_pointer_cast<arrow::BooleanArray>(arrowArrayPtr);
  case arrow::Type::DATE32:
    return (ValueArrayPtr<int32_t>)std::static_pointer_cast<arrow::Date32Array>(arrowArrayPtr);
  case arrow::Type::INT32:
    return (ValueArrayPtr<int32_t>)std::static_pointer_cast<arrow::Int32Array>(arrowArrayPtr);
  case arrow::Type::DATE64:
    return (ValueArrayPtr<int64_t>)std::static_pointer_cast<arrow::Date64Array>(arrowArrayPtr);
  case arrow::Type::INT64:
    return (ValueArrayPtr<int64_t>)std::static_pointer_cast<arrow::Int64Array>(arrowArrayPtr);
  case arrow::Type::FLOAT:
    return (ValueArrayPtr<float_t>)std::static_pointer_cast<arrow::FloatArray>(arrowArrayPtr);
  case arrow::Type::DOUBLE:
    return (ValueArrayPtr<double_t>)std::static_pointer_cast<arrow::DoubleArray>(arrowArrayPtr);
  case arrow::Type::STRING:
    return (ValueArrayPtr<std::string>)std::static_pointer_cast<arrow::StringArray>(arrowArrayPtr);
  case arrow::Type::EXTENSION: {
    auto const& extensionType = static_cast<arrow::ExtensionType const&>(*type);
    if(extensionType.extension_name()[0] == 'c') {
      return std::static_pointer_cast<ComplexExpressionArray>(arrowArrayPtr);
    }
    return (ValueArrayPtr<Symbol>)std::static_pointer_cast<arrow::StringArray>(arrowArrayPtr);
  }
  default:
    throw std::logic_error("unsupported arrow array type");
    return false;
  }
}

} // namespace boss::engines::bulk
