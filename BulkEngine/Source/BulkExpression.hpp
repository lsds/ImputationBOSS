#pragma once
#include "BulkArrays.hpp"
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>

namespace boss::engines::bulk {

class ComplexExpressionArray;

using BulkExpressionSystem = ExtensibleExpressionSystem<
    int32_t, float_t, ValueArrayPtr<bool>, ValueArrayPtr<int32_t>, ValueArrayPtr<int64_t>,
    ValueArrayPtr<float_t>, ValueArrayPtr<double_t>, ValueArrayPtr<std::string>,
    ValueArrayPtr<Symbol>, std::shared_ptr<ComplexExpressionArray>,
    std::shared_ptr<std::vector<std::shared_ptr<ComplexExpressionArray>>>>;

using AtomicExpression = BulkExpressionSystem::AtomicExpression;
using ComplexExpression = BulkExpressionSystem::ComplexExpression;
using Expression = BulkExpressionSystem::Expression;
using ExpressionArguments = BulkExpressionSystem::ExpressionArguments;

using ExpressionBuilder = boss::utilities::ExtensibleExpressionBuilder<BulkExpressionSystem>;
static ExpressionBuilder operator""_(const char* name, size_t /*unused*/) {
  return ExpressionBuilder(name);
};

Expression genericArrowArrayToBulkExpression(std::shared_ptr<arrow::Array> const& arrowArrayPtr);

} // namespace boss::engines::bulk
