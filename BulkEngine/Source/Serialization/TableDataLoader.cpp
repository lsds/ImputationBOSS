#include "TableDataLoader.hpp"

#include <arrow/array/array_base.h>
#include <arrow/array/concatenate.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/memory_pool.h>
#include <arrow/scalar.h>
#include <arrow/visitor.h>
#include <arrow/visitor_inline.h>

// for debug info
#include <chrono>
#include <iostream>

namespace boss::engines::bulk::serialization {

#ifdef NDEBUG
bool constexpr VERBOSE_LOADING = false;
#else
bool constexpr VERBOSE_LOADING = true;
#endif

template <typename Func> struct ConvertToExpressionArgumentVisitor : public arrow::ScalarVisitor {
  Func func;
  explicit ConvertToExpressionArgumentVisitor(Func&& func) : func(func) {}
  template <typename ScalarType> arrow::Status Visit(ScalarType const& scalar) {
    if constexpr(std::is_base_of_v<arrow::internal::PrimitiveScalarBase, ScalarType>) {
      if constexpr(std::is_base_of_v<arrow::BaseBinaryScalar, ScalarType>) {
        func((std::string)scalar.view());
        return arrow::Status::OK();
      } else if constexpr(std::is_integral_v<typename ScalarType::ValueType>) {
        if constexpr(sizeof(scalar.value) == sizeof(int64_t)) {
          func(static_cast<int64_t>(scalar.value));
          return arrow::Status::OK();
        } else if constexpr(sizeof(scalar.value) == sizeof(int32_t)) {
          func(static_cast<int32_t>(scalar.value));
          return arrow::Status::OK();
        }
      } else if constexpr(std::is_floating_point_v<typename ScalarType::ValueType>) {
        if constexpr(sizeof(scalar.value) == sizeof(double_t)) {
          func(static_cast<double_t>(scalar.value));
          return arrow::Status::OK();
        } else if constexpr(sizeof(scalar.value) == sizeof(float_t)) {
          func(static_cast<float_t>(scalar.value));
          return arrow::Status::OK();
        }
      }
    }
    return arrow::Status::NotImplemented("Scalar visitor for type '", scalar.type->ToString(),
                                         "' not implemented");
  }
};

bool TableDataLoader::loadInternal(std::string const& filepath, Symbol const& table, char separator,
                                   bool eolHasSeparator, bool hasHeader,
                                   std::vector<std::string> const& columnNames,
                                   Expression const& defaultMissing, unsigned long long maxRows,
                                   float /*missingChance*/,
                                   std::vector<bool> const& columnsToLoad) const {
  // loading code which will be from a normal cvs file reader or from a memory-mapped file reader
  auto loadFromReader = [&, this](auto& reader) {
    static auto debugStart = std::chrono::high_resolution_clock::now();

    arrow::RecordBatchVector batches;
    auto readBatchesResult = reader->ReadAll(&batches);
    if(!readBatchesResult.ok()) {
      throw std::runtime_error(readBatchesResult.ToString());
    }
    std::vector<bool> hasNullValues;
    for(auto& batch : batches) {
      auto const& columns = batch->columns();
      if(hasNullValues.size() < columns.size()) {
        hasNullValues.resize(columns.size());
      }
      for(int64_t colIdx = 0; colIdx < columns.size(); ++colIdx) {
        auto const& column = columns[colIdx];
        if(column->null_count() > 0) {
          hasNullValues[colIdx] = true;
        }
      }
    }
    bool hasAnyMissingValue = std::count(hasNullValues.begin(), hasNullValues.end(), true) > 0;
    int64_t totalRows = 0;
    int64_t rowsLeftInBatch = batchSize;
    for(auto& batch : batches) {
      if(maxRows <= 0) {
        break;
      }
      auto numRows = batch->num_rows();
      if(numRows < maxRows) {
        maxRows -= numRows;
      } else {
        batch = batch->Slice(0, maxRows);
        numRows = maxRows;
        maxRows = 0;
      }

      auto handleBatch = [&table](auto const& batch) {
        auto const& columns = batch->columns();
        ExpressionArguments attachColumnArgs;
        attachColumnArgs.reserve(columns.size() + 1);
        attachColumnArgs.emplace_back(table);
        std::transform(
            columns.begin(), columns.end(), std::back_inserter(attachColumnArgs),
            [](auto const& column) { return genericArrowArrayToBulkExpression(column); });
        Engine::evaluateInternal(ComplexExpression("AttachColumns"_, std::move(attachColumnArgs)));
      };

      auto handleBatchWithMissingData = [&table, &defaultMissing, &hasNullValues,
                                         this](auto const& batch) {
        auto const& columns = batch->columns();
        auto numColumns = columns.size();
        auto numRows = batch->num_rows();
        for(int64_t index = 0; index < numRows; ++index) {
          ExpressionArguments row;
          row.reserve(numColumns + 1);
          row.emplace_back(table);
          ConvertToExpressionArgumentVisitor addToRowAsAtomValue(
              [&row](auto const& value) { row.emplace_back(value); });
          ConvertToExpressionArgumentVisitor addToRowAsNoOpExpression(
              [&row](auto const& value) { row.emplace_back("NoOp1"_(value)); });
          for(int64_t colIdx = 0; colIdx < columns.size(); ++colIdx) {
            auto const& column = columns[colIdx];
            auto const& scalarResult = column->GetScalar(index);
            if(scalarResult.ok()) {
              auto const& scalar = *scalarResult.ValueOrDie();
              if(scalar.type->id() != arrow::NullScalar::TypeClass::type_id && scalar.is_valid) {
                auto status = forceNoOpForAtoms && hasNullValues[colIdx]
                                  ? arrow::VisitScalarInline(scalar, &addToRowAsNoOpExpression)
                                  : arrow::VisitScalarInline(scalar, &addToRowAsAtomValue);
                if(!status.ok()) {
                  row.emplace_back("ParseError"_(status.ToString()));
                }
                continue;
              }
            }
            // fallback - add as a missing value
            if(auto const* missingExpression = std::get_if<ComplexExpression>(&defaultMissing)) {
              if(missingExpression->getHead().getName() == "Function") {
                auto unevaluatedMissingFunction = missingExpression->clone();
                // add the table and column symbols as arguments
                unevaluatedMissingFunction.getArguments().emplace_back(
                    "List"_(table, Symbol(batch->schema()->field(colIdx)->name())));
                auto e = Engine::evaluateInternal(std::move(unevaluatedMissingFunction));
                if(auto const* evaluatedMissingExpr = std::get_if<ComplexExpression>(&e)) {
                  if(evaluatedMissingExpr->getHead().getName() == "Unevaluated") {
                    row.emplace_back(std::move(e));
                    continue;
                  }
                }
                // add Unevaluated wrapper if missing (otherwise it won't stay unevaluated)
                row.emplace_back("Unevaluated"_(std::move(e)));
                continue;
              }
            }
            row.emplace_back(defaultMissing.clone());
          }
          Engine::evaluateInternal(ComplexExpression("InsertInto"_, std::move(row)));
        }
      };
      bool hasAnyMissing = false;
      do {
        hasAnyMissing = false;
        int64_t startIndex = batch->num_rows();
        int64_t endIndex = 0;
        for(auto const& column : batch->columns()) {
          if(column->null_count() > 0) {
            hasAnyMissing = true;
            for(int64_t i = 0; i < startIndex; ++i) {
              if(column->IsNull(i)) {
                startIndex = i;
                for(auto j = endIndex > i ? endIndex - 1 : i; j < column->length(); ++j) {
                  if(!column->IsNull(j)) {
                    break;
                  }
                  endIndex = j + 1;
                }
                break;
              }
            }
          }
        }

        if(hasAnyMissing) {
          while(startIndex >= rowsLeftInBatch) {
            if(forceNoOpForAtoms) {
              // we set an expression for every atom (a No-Op)
              handleBatchWithMissingData(batch->Slice(0, rowsLeftInBatch));
            } else {
              handleBatch(batch->Slice(0, rowsLeftInBatch));
            }
            batch = batch->Slice(rowsLeftInBatch);
            startIndex -= rowsLeftInBatch;
            endIndex -= rowsLeftInBatch;
            rowsLeftInBatch = batchSize;
          }
          if(startIndex > 0) {
            if(forceNoOpForAtoms) {
              // we set an expression for every atom (a No-Op)
              handleBatchWithMissingData(batch->Slice(0, startIndex));
            } else {
              handleBatch(batch->Slice(0, startIndex));
            }
            rowsLeftInBatch -= startIndex;
          }
          handleBatchWithMissingData(batch->Slice(startIndex, endIndex - startIndex));
          batch = batch->Slice(endIndex);
          if(batch->num_rows() == 0) {
            break;
          }
        }
      } while(hasAnyMissing);

      if(batch->num_rows() > 0) {
        while(batch->num_rows() >= rowsLeftInBatch) {
          if(forceNoOpForAtoms && hasAnyMissingValue) {
            // we set an expression for every atom (a No-Op)
            handleBatchWithMissingData(batch->Slice(0, rowsLeftInBatch));
          } else {
            handleBatch(batch->Slice(0, rowsLeftInBatch));
          }
          batch = batch->Slice(rowsLeftInBatch);
          rowsLeftInBatch = batchSize;
        }
        if(batch->num_rows() > 0) {
          if(forceNoOpForAtoms && hasAnyMissingValue) {
            // we set an expression for every atom (a No-Op)
            handleBatchWithMissingData(batch);
          } else {
            handleBatch(batch);
          }
          rowsLeftInBatch -= batch->num_rows();
        }
      }

      if constexpr(VERBOSE_LOADING) {
        auto debugEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = debugEnd - debugStart;
        auto speed = static_cast<int>(static_cast<float>(numRows) / elapsed.count());
        debugStart = debugEnd;
        std::cerr << " [speed:" << speed << "/s] inserting " << numRows << " rows." << std::endl;
      }

      totalRows += numRows;
    }
  };

  auto const& io_context = arrow::io::default_io_context();

  // try to load cached memory-mapped file
  std::shared_ptr<arrow::io::MemoryMappedFile> memoryMappedFile;
  if(memoryMapped) {
    auto memoryMappedFilepath = filepath + ".cached";
    auto maybeMemoryMappedFile =
        arrow::io::MemoryMappedFile::Open(memoryMappedFilepath, arrow::io::FileMode::READWRITE);
    if(!maybeMemoryMappedFile.ok()) {
      throw std::runtime_error("failed to open " + memoryMappedFilepath + " \n" +
                               maybeMemoryMappedFile.status().ToString());
    }
    memoryMappedFile = *maybeMemoryMappedFile;
  }

  // aligned to char size
  auto headerSize = ((sizeof(batchSize) + CHAR_BIT - 1) / CHAR_BIT) * CHAR_BIT;

  bool loadOriginalData = false;
  if(!memoryMappedFile || memoryMappedFile->GetSize() == 0) {
    loadOriginalData = true;
  } else {
    decltype(batchSize) cachedBatchSize = 0;
    auto result = memoryMappedFile->Read(sizeof(cachedBatchSize), &cachedBatchSize);
    if(!result.ok()) {
      throw std::runtime_error(result.status().ToString());
    }
    auto status = memoryMappedFile->Advance(headerSize - sizeof(cachedBatchSize));
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    if(cachedBatchSize == 0 || cachedBatchSize % batchSize != 0) {
      loadOriginalData = true;
      auto seekStatus = memoryMappedFile->Seek(0);
      if(!seekStatus.ok()) {
        throw std::runtime_error(seekStatus.ToString());
      }
    }
  }

  if(loadOriginalData) {
    // first time loading, load the original files
    auto maybeFileInput = arrow::io::ReadableFile::Open(filepath, io_context.pool());
    if(!maybeFileInput.ok()) {
      throw std::runtime_error("failed to find " + filepath + " \n" +
                               maybeFileInput.status().ToString());
    }
    auto cvsInput = *maybeFileInput;

    auto csvFilesizeResult = cvsInput->GetSize();
    if(!csvFilesizeResult.ok()) {
      throw std::runtime_error("failed to get size of " + filepath + " \n" +
                               csvFilesizeResult.status().ToString());
    }
    auto csvFilesize = *csvFilesizeResult;

    auto readOptions = arrow::csv::ReadOptions::Defaults();

    // loading the whole arrow array at once
    // and leave the logic to the loadFromReader() for splitting into batches
    // (otherwise it would force BOSS to do copies when batches are just in between two blocks)
    // BUT for size > ~2GB, we have to split the loading and re-assemble the batches
    readOptions.block_size = csvFilesize < std::numeric_limits<int32_t>::max()
                                 ? csvFilesize
                                 : std::numeric_limits<int32_t>::max();

    if(!hasHeader) {
      readOptions.column_names = columnNames;

      if(eolHasSeparator) {
        // need one more dummy column
        // to handle Arrow wrongly loading a value at the end of the line
        // (it will ignore during the loading...)
        readOptions.column_names.emplace_back();
      }

      // insert empty columns in between to filter out specific columns
      auto posIt = readOptions.column_names.begin();
      for(bool toLoad : columnsToLoad) {
        if(toLoad) {
          ++posIt;
        } else {
          posIt = readOptions.column_names.insert(posIt, "");
          ++posIt;
        }
      }
    }

    auto parseOptions = arrow::csv::ParseOptions::Defaults();
    parseOptions.delimiter = separator;

    auto convertOptions = arrow::csv::ConvertOptions::Defaults();
    convertOptions.include_columns = columnNames;
    convertOptions.include_missing_columns = true;

    auto maybeCvsReader = arrow::csv::StreamingReader::Make(io_context, cvsInput, readOptions,
                                                            parseOptions, convertOptions);
    if(!maybeCvsReader.ok()) {
      throw std::runtime_error("failed to open " + filepath + " \n" +
                               maybeCvsReader.status().ToString());
    }
    auto cvsReader = *maybeCvsReader;

    if(!memoryMappedFile) {
      if constexpr(VERBOSE_LOADING) {
        std::cerr << "Loading from file: " << table.getName() << std::endl;
      }
      loadFromReader(cvsReader);
      return true;
    }

    if constexpr(VERBOSE_LOADING) {
      std::cerr << "Caching: " << table.getName() << std::endl;
    }
    static auto debugStart = std::chrono::high_resolution_clock::now();

    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    auto write = [&](auto const& batch) {
      if(!writer) {
        auto const& schema = batch.schema();

        auto maybeWriter = arrow::ipc::MakeStreamWriter(memoryMappedFile, schema);
        if(!maybeWriter.ok()) {
          throw std::runtime_error("failed to open memory-mapped stream writer\n" +
                                   maybeWriter.status().ToString());
        }

        writer = *maybeWriter;
        auto resizeStatus = memoryMappedFile->Resize(headerSize);
        if(!resizeStatus.ok()) {
          throw std::runtime_error(resizeStatus.ToString());
        }

        auto writeStatus = memoryMappedFile->Write(&batchSize, sizeof(batchSize));
        if(!writeStatus.ok()) {
          throw std::runtime_error(writeStatus.ToString());
        }

        auto advanceStatus = memoryMappedFile->Advance(headerSize - sizeof(batchSize));
        if(!advanceStatus.ok()) {
          throw std::runtime_error(advanceStatus.ToString());
        }
      }

      int64_t recordBatchSize = 0;
      auto getSizeStatus = arrow::ipc::GetRecordBatchSize(batch, &recordBatchSize);
      if(!getSizeStatus.ok()) {
        throw std::runtime_error("failed to get record batch size\n" + getSizeStatus.ToString());
      }

      auto currentSize = *memoryMappedFile->GetSize();
      if(currentSize == headerSize) {
        // make space for schema
        currentSize = recordBatchSize;
      }

      auto resizeStatus = memoryMappedFile->Resize(currentSize + recordBatchSize);
      if(!resizeStatus.ok()) {
        throw std::runtime_error(resizeStatus.ToString());
      }

      auto writeStatus = writer->WriteRecordBatch(batch);
      if(!writeStatus.ok()) {
        throw std::runtime_error("failed to write\n" + writeStatus.ToString());
      }

      if constexpr(VERBOSE_LOADING) {
        auto numRows = batch.num_rows();
        auto debugEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = debugEnd - debugStart;
        auto speed = static_cast<int>(static_cast<float>(numRows) / elapsed.count());
        debugStart = debugEnd;
        std::cerr << " [speed:" << speed << "/s] caching " << numRows << " rows." << std::endl;
      }
    };

    std::shared_ptr<arrow::RecordBatch> mergedBatch;
    auto merge = [&mergedBatch](auto&& batch) {
      if(!mergedBatch) {
        mergedBatch = batch;
        return;
      }
      auto const& schema = batch->schema();
      auto const& numColumns = batch->columns().size();
      std::vector<std::shared_ptr<arrow::Array>> newColumns;
      newColumns.reserve(numColumns);
      for(int i = 0; i < numColumns; ++i) {
        auto const& curColumn = mergedBatch->column(i);
        auto const& newColumn = batch->column(i);
        auto columnsToMerge = arrow::ArrayVector{curColumn, newColumn};
        auto mergeResult = arrow::Concatenate(columnsToMerge);
        if(!mergeResult.ok()) {
          throw std::runtime_error(mergeResult.status().ToString());
        }
        newColumns.emplace_back(*mergeResult);
      }
      mergedBatch =
          arrow::RecordBatch::Make(schema, mergedBatch->num_rows() + batch->num_rows(), newColumns);
    };

    std::shared_ptr<arrow::RecordBatch> batch;
    while(cvsReader->ReadNext(&batch).ok() && batch) {
      int64_t mergedNumRows = mergedBatch ? mergedBatch->num_rows() : 0;
      while(mergedNumRows + batch->num_rows() > batchSize) {
        auto numRowsToMerge = batchSize - mergedNumRows;
        if(numRowsToMerge > 0) {
          merge(batch->Slice(0, numRowsToMerge));
          batch = batch->Slice(numRowsToMerge);
        }
        write(*mergedBatch);
        mergedBatch = nullptr;
        mergedNumRows = 0;
      }
      merge(batch);
    }
    batch = nullptr;

    if(mergedBatch) {
      write(*mergedBatch);
      mergedBatch = nullptr;
    }

    if(writer) {
      auto closeStatus = writer->Close();
      if(!closeStatus.ok()) {
        throw std::runtime_error(closeStatus.ToString());
      }
    }

    auto seekStatus = memoryMappedFile->Seek(headerSize);
    if(!seekStatus.ok()) {
      throw std::runtime_error(seekStatus.ToString());
    }
  }

  auto maybeReader = arrow::ipc::RecordBatchStreamReader::Open(memoryMappedFile);
  if(!maybeReader.ok()) {
    throw std::runtime_error("failed to open memory-mapped stream reader\n" +
                             maybeReader.status().ToString());
  }

  if constexpr(VERBOSE_LOADING) {
    std::cerr << "Loading from cache: " << table.getName() << std::endl;
  }
  loadFromReader(*maybeReader);

  return true;
}

} // namespace boss::engines::bulk::serialization
