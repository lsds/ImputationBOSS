#pragma once

#include "FastStringBuilder.hpp"
#include "SymbolArray.hpp"

namespace boss::engines::bulk {

/** SymbolArrayBuilder is the builder corresponding to SymbolArray.
 * Because it uses custom type SymbolType, Arrow will automatically
 * pick it up to create a SymbolArray when finishing the builder.*/
class SymbolBuilder : public FastStringBuilder {
public:
  explicit SymbolBuilder(const std::shared_ptr<arrow::DataType>& type,
                         arrow::MemoryPool* pool = arrow::default_memory_pool())
      : FastStringBuilder(type, pool) {
  }

  explicit SymbolBuilder(arrow::MemoryPool* pool = arrow::default_memory_pool())
      : FastStringBuilder(pool) {
  }

  std::shared_ptr<arrow::DataType> type() const override {
    return std::make_shared<SymbolArray::SymbolType>();
  }
};

} // namespace boss::engines::bulk