#pragma once

#include "ArrowExtensions/ComplexExpressionBuilder.hpp"
#include "Bulk.hpp"
#include "OperatorUtilities.hpp"
#include "SymbolRegistry.hpp"
#include "Table.hpp"

namespace boss::engines::bulk {

template <typename... ArgumentTypes>
class CreateTable : public boss::engines::bulk::Operator<CreateTable, ArgumentTypes...> {
public:
  static constexpr int MAX_NUM_COLUMNS = 50;
  using ArgumentTypesT = RepeatedArgumentTypeOfAnySize_t<2, MAX_NUM_COLUMNS + 1, Symbol>;
  using Operator<CreateTable, ArgumentTypes...>::Operator;
  template <typename... Columns> void operator()(Symbol const& table, Columns const&... columns) {
    auto schema = arrow::FieldVector{std::make_shared<arrow::Field>(columns.getName(), nullptr)...};
    TableSymbolRegistry::globalInstance().registerSymbol(table, std::move(schema));
  }
};
namespace {
boss::engines::bulk::Engine::Register<CreateTable> const r1000("CreateTable"); // NOLINT
}

template <typename... ArgumentTypes>
class InsertInto : public boss::engines::bulk::Operator<InsertInto, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      RepeatedArgumentTypeOfAnySize_t<2, CreateTable<>::MAX_NUM_COLUMNS + 1, Expression>;
  using Operator<InsertInto, ArgumentTypes...>::Operator;
  template <typename... Args>
  void operator()(Expression const& exprTable, Args const&... exprValues) {
    auto const& tableSymbol = get<Symbol>(exprTable);
    auto const& tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableSymbol);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableSymbol.getName() + "\"_");
    }
    auto& table = *tablePtr;
    auto evaluatedRow = "List"_(this->evaluateInternal(exprValues)...);
    // find matching partition (or create new partition)
    auto partitionIndex = table.findMatchingPartitionIndex(evaluatedRow);
    if(partitionIndex < table.numPartitions()) {
      // get the builder and insert the row
      auto builder = table.getPartitionBuilder(partitionIndex);
      auto prevSize = builder->length();
      builder->appendExpression(evaluatedRow, true /*init order*/, table.getMaxGlobalIndex());
      table.getMaxGlobalIndex() += builder->length() - prevSize;
      // re-set the builder again for internal updates
      table.setPartition(partitionIndex, std::move(builder));
    } else {
      // create builder and insert row
      auto builder = std::make_shared<ComplexExpressionArrayBuilder>(evaluatedRow.getHead(),
                                                                     table.getSchema(), 0);
      builder->appendExpression(evaluatedRow, true /*init order*/, table.getMaxGlobalIndex());
      table.getMaxGlobalIndex() += builder->length();
      table.addPartition(std::move(builder));
    }
  }
};
namespace {
boss::engines::bulk::Engine::Register<InsertInto> const r1001("InsertInto");     // NOLINT
boss::engines::bulk::Engine::Register<InsertInto> const r1001b("AttachColumns"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class GetColumns : public boss::engines::bulk::Operator<GetColumns, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr>>;
  using Operator<GetColumns, ArgumentTypes...>::Operator;
  void operator()(Table::PartitionPtr const& partitionPtr) {
    auto outputBuilder = ValueBuilder<Symbol>(0);
    auto const& schema = partitionPtr->getSchema();
    outputBuilder.Reserve(schema.size());
    auto dataSize =
        std::accumulate(schema.begin(), schema.end(), 0, [](auto total, auto const& field) {
          return total + field->name().length();
        });
    outputBuilder.ReserveData(dataSize);
    for(auto const& field : schema) {
      outputBuilder.UnsafeAppend(field->name());
    }
    this->pushUp((ValueArrayPtr<Symbol>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<GetColumns> const r1002("Columns"); // NOLINT
}

template <typename... ArgumentTypes>
class FinaliseTable : public boss::engines::bulk::Operator<FinaliseTable, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol>>;
  using Operator<FinaliseTable, ArgumentTypes...>::Operator;
  void operator()(Symbol const& tableSymbol) {
    auto const& tablePtr = TableSymbolRegistry::globalInstance().findSymbol(tableSymbol);
    if(tablePtr == nullptr) {
      throw std::logic_error("Cannot find table \"" + tableSymbol.getName() + "\"_");
    }
    tablePtr->finaliseAndGetPartitions();
  }
};
namespace {
boss::engines::bulk::Engine::Register<FinaliseTable> const r1003("FinaliseTable"); // NOLINT
}

template <typename... ArgumentTypes>
class DropTable : public boss::engines::bulk::Operator<DropTable, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol>>;
  using Operator<DropTable, ArgumentTypes...>::Operator;
  void operator()(Symbol const& table) { TableSymbolRegistry::globalInstance().clearSymbol(table); }
};
namespace {
boss::engines::bulk::Engine::Register<DropTable> const r1004("DropTable"); // NOLINT
}

} // namespace boss::engines::bulk
