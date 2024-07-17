#pragma once
#include "BulkExpression.hpp"
#include "BulkUtilities.hpp"
#include "Serialization/TableDataLoader.hpp"

using namespace std;

namespace boss::engines::bulk {

#ifdef NDEBUG
bool constexpr VERBOSE_LOADING = false;
#else
bool constexpr VERBOSE_LOADING = true;
#endif

template <typename... ArgumentTypes>
class ArrowArrayPtrOp : public boss::engines::bulk::Operator<ArrowArrayPtrOp, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<int64_t>>;
  using Operator<ArrowArrayPtrOp, ArgumentTypes...>::Operator;
  void operator()(int64_t addressAsLong) {
    auto const& arrowArrayPtr = boss::utilities::nasty::reconstructArrowArray(addressAsLong);
    this->pushUp(genericArrowArrayToBulkExpression(arrowArrayPtr));
  }
};
namespace {
boss::engines::bulk::Engine::Register<ArrowArrayPtrOp> const ser0("ArrowArrayPtr"); // NOLINT
}

template <typename... ArgumentTypes>
class LoadSchema : public boss::engines::bulk::Operator<LoadSchema, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol, string>>;
  using Operator<LoadSchema, ArgumentTypes...>::Operator;
  void operator()(Symbol const& /*table*/, string const& /*schemaFile*/) {
    throw runtime_error("Not implemented");
  }
};
namespace {
boss::engines::bulk::Engine::Register<LoadSchema> const ser1("LoadSchema"); // NOLINT
}

template <typename... ArgumentTypes>
class InsertIntoFromFile
    : public boss::engines::bulk::Operator<InsertIntoFromFile, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol, string>, tuple<Symbol, string, Symbol>,
                                 tuple<Symbol, string, ComplexExpression>>;
  using Operator<InsertIntoFromFile, ArgumentTypes...>::Operator;
  void operator()(Symbol const& table, string const& filename) {
    serialization::TableDataLoader loader;
    loader.setBatchSize(bulk::Properties::getMicroBatchesMaxSize());
    loader.setMemoryMapped(bulk::Properties::getUseMemoryMappedFiles());
    loader.setForceNoOpForAtoms(bulk::Properties::getForceNoOpForAtoms());
    this->pushUp(loader.load(table, filename));
  }
  void operator()(Symbol const& table, string const& filename,
                  ComplexExpression const& defaultMissingExpression) {
    serialization::TableDataLoader loader;
    loader.setBatchSize(bulk::Properties::getMicroBatchesMaxSize());
    loader.setMemoryMapped(bulk::Properties::getUseMemoryMappedFiles());
    loader.setForceNoOpForAtoms(bulk::Properties::getForceNoOpForAtoms());
    this->pushUp(loader.load(table, filename, defaultMissingExpression.clone()));
  }
  template <typename MissingType>
  void operator()(Symbol const& table, string const& filename,
                  MissingType const& defaultMissingValue) {
    serialization::TableDataLoader loader;
    loader.setBatchSize(bulk::Properties::getMicroBatchesMaxSize());
    loader.setMemoryMapped(bulk::Properties::getUseMemoryMappedFiles());
    loader.setForceNoOpForAtoms(bulk::Properties::getForceNoOpForAtoms());
    this->pushUp(loader.load(table, filename, defaultMissingValue));
  }
};
namespace {
boss::engines::bulk::Engine::Register<InsertIntoFromFile> const ser2("Load"); // NOLINT
}

} // namespace boss::engines::bulk
