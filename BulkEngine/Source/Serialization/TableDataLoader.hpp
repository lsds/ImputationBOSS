#pragma once

#include "../Bulk.hpp"
#include "../Table.hpp"

#include <Expression.hpp>
#include <ExpressionUtilities.hpp>

#include <functional>
#include <string>
#include <vector>

using boss::utilities::operator""_;

namespace boss::engines::bulk::serialization {

class TableDataLoader {
public:
  TableDataLoader()
      : batchSize(std::numeric_limits<int64_t>::max()), memoryMapped(true),
        forceNoOpForAtoms(false) {}

  void setBatchSize(size_t size) { batchSize = size; }
  void setMemoryMapped(bool mm) { memoryMapped = mm; }
  void setForceNoOpForAtoms(bool force) { forceNoOpForAtoms = force; }

  bool load(Symbol const& table, std::string const& filepath,
            Expression const& defaultMissing = "Missing"_, unsigned long long maxRows = -1,
            float missingChance = 0.0F,
            std::vector<bool> const& columnsToLoad = std::vector<bool>()) const {
    if(filepath.rfind(".tbl") != std::string::npos) {
      return loadFromTBL(table, filepath, defaultMissing, maxRows, missingChance, columnsToLoad);
    }

    if(filepath.rfind(".csv") != std::string::npos) {
      return loadFromCSV(table, filepath, defaultMissing, maxRows, missingChance, columnsToLoad);
    }

    throw std::runtime_error("unsupported file format for " + filepath);
  }

  bool loadFromTBL(Symbol const& table, std::string const& filepath,
                   Expression const& defaultMissing = "Missing"_, unsigned long long maxRows = -1,
                   float missingChance = 0.0F,
                   std::vector<bool> const& columnsToLoad = std::vector<bool>()) const {
    return load(TableInfo(table), filepath, '|', true, false, defaultMissing, maxRows,
                missingChance, columnsToLoad);
  }

  bool loadFromCSV(Symbol const& table, std::string const& filepath,
                   Expression const& defaultMissing = "Missing"_, unsigned long long maxRows = -1,
                   float missingChance = 0.0F,
                   std::vector<bool> const& columnsToLoad = std::vector<bool>()) const {
    return load(TableInfo(table), filepath, ',', false, true, defaultMissing, maxRows,
                missingChance, columnsToLoad);
  }

private:
  struct TableInfo {
    explicit TableInfo(Symbol const& table) : table(table) {
      auto const& tablePtr = TableSymbolRegistry::globalInstance().findSymbol(table);
      if(tablePtr == nullptr) {
        throw std::runtime_error("cannot find table " + table.getName() + " to load data into.");
      }
      auto const& schema = tablePtr->getSchema();
      columnNames.reserve(schema.size());
      std::transform(schema.begin(), schema.end(), std::back_inserter(columnNames),
                     [](auto const& field) { return field->name(); });
    }

    Symbol const& table;
    std::vector<std::string> columnNames;
  };

  bool load(TableInfo const& info, std::string const& filepath, char separator,
            bool eolHasSeparator, bool hasHeader, Expression const& defaultMissing,
            unsigned long long maxRows, float missingChance,
            std::vector<bool> const& columnsToLoad) const {
    return loadInternal(filepath, info.table, separator, eolHasSeparator, hasHeader,
                        info.columnNames, defaultMissing, maxRows, missingChance, columnsToLoad);
  }

  bool loadInternal(std::string const& filepath, Symbol const& table, char separator,
                    bool eolHasSeparator, bool hasHeader,
                    std::vector<std::string> const& columnNames, Expression const& defaultMissing,
                    unsigned long long maxRows, float missingChance,
                    std::vector<bool> const& columnsToLoad) const;

  int64_t batchSize;
  bool memoryMapped;
  bool forceNoOpForAtoms;
};

} // namespace boss::engines::bulk::serialization
