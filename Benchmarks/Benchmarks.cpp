#include "DuckDB.hpp"
#include "ITTNotifySupport.hpp"
#include "MonetDB.hpp"
#include <BOSS.hpp>
#include <ExpressionUtilities.hpp>
#include <arrow/builder.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <map>

using namespace std;

using namespace boss::utilities;

static auto const vtune = VTuneAPIInterface{"BOSS"};

static bool VERBOSE_QUERY_OUTPUT = false;
static int VERBOSE_QUERY_OUTPUT_MAX_LINES = 1000;
static bool EXPLAIN_QUERY_OUTPUT = false;
static bool PROFILE_QUERY_OUTPUT = false;

static bool BOSS_FULL_VERBOSE = false;
static bool BOSS_BENCHMARK_BATCH_SIZE = false;
static int64_t BOSS_DEFAULT_BATCH_SIZE = 64000;
static bool BOSS_USE_MEMORY_MAPPED_FILES = true;
static bool BOSS_ENABLE_ORDER_PRESERVATION_CACHE = false;

static bool BOSS_FORCE_NO_OP_FOR_ATOMS = false;
static bool BOSS_DISABLE_EXPRESSION_PARTITIONING = false;
static bool BOSS_DISABLE_EXPRESSION_DECOMPOSITION = false;

static int BENCHMARK_NUM_WARMPUP_ITERATIONS = 3;

static bool MONETDB_MULTITHREADING = false;
static int DUCKDB_MAX_THREADS = 1;

static bool USE_FIXED_POINT_NUMERIC_TYPE = false;

static bool ENABLE_INDEXES = true;

static std::vector<string>
    librariesToTest{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

static auto& lastEngineLibrary() {
  static auto engineLibrary = std::string();
  return engineLibrary;
}

static auto& lastDataSize() {
  static int dataSize;
  return dataSize;
}

static auto& lastBatchSize() {
  static int64_t batchSize;
  return batchSize;
}

static auto& lastImputationMethod() {
  static int imputationMethod;
  return imputationMethod;
}

static auto& lastVariant() {
  static std::string v;
  return v;
}

static auto& lastExtension() {
  static std::string ext;
  return ext;
}

static auto& lastDataset() {
  static std::string ext;
  return ext;
}

static void resetBOSSEngine() {
  if(!lastEngineLibrary().empty()) {
    auto eval = [](auto const& expression) mutable {
      return boss::evaluate("EvaluateInEngines"_("List"_(lastEngineLibrary()), expression));
    };

    eval("DropTable"_("REGION"_));
    eval("DropTable"_("NATION"_));
    eval("DropTable"_("PART"_));
    eval("DropTable"_("SUPPLIER"_));
    eval("DropTable"_("PARTSUPP"_));
    eval("DropTable"_("CUSTOMER"_));
    eval("DropTable"_("ORDERS"_));
    eval("DropTable"_("LINEITEM"_));

    lastEngineLibrary() = "";
  }
}

enum DB_ENGINE { MONETDB, DUCKDB, BOSS_ENGINES_START };
static std::vector<string> const DBEngineNames{"MonetDB", "DuckDB", "BOSS"};

enum IMPUTATION_METHOD { MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP };

static int const IMPUTATION_MODE_OFFSET = 10;

enum IMPUTATION_MODE {
  LOCAL = IMPUTATION_MODE_OFFSET * 0,
  GLOBAL = IMPUTATION_MODE_OFFSET * 1,
  RANDOM_1P = IMPUTATION_MODE_OFFSET * 2,
  RANDOM_2P = IMPUTATION_MODE_OFFSET * 3,
  RANDOM_4P = IMPUTATION_MODE_OFFSET * 4,
  RANDOM_8P = IMPUTATION_MODE_OFFSET * 5,
  RANDOM_16P = IMPUTATION_MODE_OFFSET * 6,
  RANDOM_32P = IMPUTATION_MODE_OFFSET * 7,
  RANDOM_64P = IMPUTATION_MODE_OFFSET * 8,
};

static std::map<int, string> imputationMethodNames{{MEAN, "Mean"},
                                                         {HOTDECK, "HotDeck"},
                                                         {DTREE, "DecisionTree"},
                                                         {INTERPOLATE, "Interpolate"},
                                                         {NO_OP, "Minimal"}};

static std::map<int, string> imputationModeNames{
    {LOCAL, "Local"},    {GLOBAL, "Global"},  {RANDOM_1P, "1P"},
    {RANDOM_2P, "2P"},   {RANDOM_4P, "4P"},   {RANDOM_8P, "8P"},
    {RANDOM_16P, "16P"}, {RANDOM_32P, "32P"}, {RANDOM_64P, "64P"}};

static auto& bossImputationExpressions() {
  static std::map<int, boss::Expression> exprs;
  if(exprs.empty()) {
    exprs.try_emplace(MEAN + LOCAL, "ApproxMean"_());
    exprs.try_emplace(HOTDECK + LOCAL, "HotDeck"_());
    exprs.try_emplace(DTREE + LOCAL, "DecisionTree"_());
    exprs.try_emplace(INTERPOLATE + LOCAL, "Interpolate"_());
    exprs.try_emplace(NO_OP + LOCAL, "NoOp1"_(1));
    exprs.try_emplace(MEAN + GLOBAL,
                      "Function"_("List"_("table"_, "col"_), "ApproxMean"_("table"_, "col"_)));
    exprs.try_emplace(HOTDECK + GLOBAL,
                      "Function"_("List"_("table"_, "col"_), "HotDeck"_("table"_, "col"_)));
    exprs.try_emplace(DTREE + GLOBAL,
                      "Function"_("List"_("table"_, "col"_), "DecisionTree"_("table"_, "col"_)));
    exprs.try_emplace(INTERPOLATE + GLOBAL,
                      "Function"_("List"_("table"_, "col"_), "Interpolate"_("table"_, "col"_)));
    exprs.try_emplace(NO_OP + GLOBAL, "NoOp1"_(1));
    static std::map<int, boss::Expression> innerExprs;
    if(innerExprs.empty()) {
      innerExprs.try_emplace(MEAN, "ApproxMean"_());
      innerExprs.try_emplace(HOTDECK, "HotDeck"_());
      innerExprs.try_emplace(DTREE, "DecisionTree"_());
      innerExprs.try_emplace(INTERPOLATE, "Interpolate"_());
      innerExprs.try_emplace(NO_OP, 1);
    }
    for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
      auto const& innerExpr = innerExprs[imputationMethod];
      exprs.try_emplace(
          imputationMethod + RANDOM_1P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_2P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_4P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp2"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_8P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp3"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_16P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp4"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_32P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp6"_(innerExpr.clone()))))));
      exprs.try_emplace(
          imputationMethod + RANDOM_64P,
          "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp1"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp2"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp3"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp4"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp5"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp6"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp7"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp1"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp2"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp3"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp4"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp5"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp6"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp7"_("NoOp8"_(innerExpr.clone()))),
                                           "Unevaluated"_("NoOp8"_("NoOp8"_(innerExpr.clone()))))));
    }
  }
  return exprs;
}

static void initBOSSEngine_TPCH(std::string const& engineLibrary, int dataSize, int64_t batchSize,
                                int imputationMethod, std::string variant, std::string extension) {
  static auto dataset = std::string("TPCH");
  if(lastEngineLibrary().empty() || engineLibrary != lastEngineLibrary() ||
     lastDataset() != dataset || dataSize != lastDataSize() || batchSize != lastBatchSize() ||
     imputationMethod != lastImputationMethod() || variant != lastVariant() ||
     extension != lastExtension()) {
    resetBOSSEngine();

    lastEngineLibrary() = engineLibrary;
    lastDataset() = dataset;
    lastDataSize() = dataSize;
    lastBatchSize() = batchSize;
    lastImputationMethod() = imputationMethod;
    lastVariant() = variant;
    lastExtension() = extension;

    auto eval = [&engineLibrary](auto const& expression) mutable {
      return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
    };

    auto checkForErrors = [](auto&& output) {
      auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
      if(maybeComplexExpr == nullptr) {
        return;
      }
      if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
        std::cout << "Error: " << output << std::endl;
      }
    };

    checkForErrors(
        eval("Set"_("EnableOrderPreservationCache"_, BOSS_ENABLE_ORDER_PRESERVATION_CACHE)));
    checkForErrors(eval("Set"_("UseMemoryMappedFiles"_, BOSS_USE_MEMORY_MAPPED_FILES)));
    checkForErrors(eval("Set"_("ForceNoOpForAtoms"_, BOSS_FORCE_NO_OP_FOR_ATOMS)));
    checkForErrors(
        eval("Set"_("DisableExpressionPartitioning"_, BOSS_DISABLE_EXPRESSION_PARTITIONING)));
    checkForErrors(
        eval("Set"_("DisableExpressionDecomposition"_, BOSS_DISABLE_EXPRESSION_DECOMPOSITION)));
    checkForErrors(eval("Set"_("MicroBatchesMaxSize"_, batchSize)));
    checkForErrors(eval("Set"_("milliSF"_, dataSize))); // for cardinality calculation

    checkForErrors(
        eval("CreateTable"_("LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_,
                            "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_,
                            "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_, "L_COMMITDATE"_,
                            "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_)));
    checkForErrors(eval("CreateTable"_("REGION"_, "R_REGIONKEY"_, "R_NAME"_, "R_COMMENT"_)));
    checkForErrors(
        eval("CreateTable"_("NATION"_, "N_NATIONKEY"_, "N_NAME"_, "N_REGIONKEY"_, "N_COMMENT"_)));
    checkForErrors(
        eval("CreateTable"_("PART"_, "P_PARTKEY"_, "P_NAME"_, "P_MFGR"_, "P_BRAND"_, "P_TYPE"_,
                            "P_SIZE"_, "P_CONTAINER"_, "P_RETAILPRICE"_, "P_COMMENT"_)));
    checkForErrors(eval("CreateTable"_("SUPPLIER"_, "S_SUPPKEY"_, "S_NAME"_, "S_ADDRESS"_,
                                       "S_NATIONKEY"_, "S_PHONE"_, "S_ACCTBAL"_, "S_COMMENT"_)));
    checkForErrors(eval("CreateTable"_("PARTSUPP"_, "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_AVAILQTY"_,
                                       "PS_SUPPLYCOST"_, "PS_COMMENT"_)));
    checkForErrors(
        eval("CreateTable"_("CUSTOMER"_, "C_CUSTKEY"_, "C_NAME"_, "C_ADDRESS"_, "C_NATIONKEY"_,
                            "C_PHONE"_, "C_ACCTBAL"_, "C_MKTSEGMENT"_, "C_COMMENT"_)));
    checkForErrors(eval("CreateTable"_("ORDERS"_, "O_ORDERKEY"_, "O_CUSTKEY"_, "O_ORDERSTATUS"_,
                                       "O_TOTALPRICE"_, "O_ORDERDATE"_, "O_ORDERPRIORITY"_,
                                       "O_CLERK"_, "O_SHIPPRIORITY"_, "O_COMMENT"_)));
    auto filenames = std::vector<std::string>{"lineitem", "region",   "nation",   "part",
                                              "supplier", "partsupp", "customer", "orders"};
    auto tables = std::vector<boss::Symbol>{"LINEITEM"_, "REGION"_,   "NATION"_,   "PART"_,
                                            "SUPPLIER"_, "PARTSUPP"_, "CUSTOMER"_, "ORDERS"_};
    std::string lineitemPath = "../data/tpch_" + std::to_string(dataSize) + "MB/" + variant + "/" +
                               filenames[0] + extension;
    auto const& defaultMissing = bossImputationExpressions().find(imputationMethod)->second;
    checkForErrors(eval("Load"_(tables[0], lineitemPath, defaultMissing)));
    for(int i = 1; i < tables.size(); ++i) {
      std::string path = "../data/tpch_" + std::to_string(dataSize) + "MB/" + filenames[i] + ".tbl";
      checkForErrors(eval("Load"_(tables[i], path)));
    }

    eval("FinaliseTable"_("REGION"_));
    eval("FinaliseTable"_("NATION"_));
    eval("FinaliseTable"_("PART"_));
    eval("FinaliseTable"_("SUPPLIER"_));
    eval("FinaliseTable"_("PARTSUPP"_));
    eval("FinaliseTable"_("CUSTOMER"_));
    eval("FinaliseTable"_("ORDERS"_));
    eval("FinaliseTable"_("LINEITEM"_));
  }
};

static void initBOSSEngine_FCC(std::string const& engineLibrary, int batchSize,
                               int imputationMethod) {
  static auto dataset = std::string("FCC");
  if(lastEngineLibrary().empty() || engineLibrary != lastEngineLibrary() ||
     lastDataset() != dataset || batchSize != lastBatchSize() ||
     imputationMethod != lastImputationMethod()) {
    resetBOSSEngine();

    lastEngineLibrary() = engineLibrary;
    lastDataset() = dataset;
    lastBatchSize() = batchSize;
    lastImputationMethod() = imputationMethod;

    auto eval = [&engineLibrary](auto const& expression) mutable {
      return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
    };

    auto checkForErrors = [](auto&& output) {
      auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
      if(maybeComplexExpr == nullptr) {
        return;
      }
      if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
        std::cout << "Error: " << output << std::endl;
      }
    };

    checkForErrors(
        eval("Set"_("EnableOrderPreservationCache"_, BOSS_ENABLE_ORDER_PRESERVATION_CACHE)));
    checkForErrors(
        eval("Set"_("UseMemoryMappedFiles"_, false))); // cannot use cache for imputed data
    checkForErrors(eval("Set"_("MicroBatchesMaxSize"_, batchSize)));
    // eval("Set"_("milliSF"_, dataSize)); // for cardinality calculation

    checkForErrors(eval("CreateTable"_(
        "FCC"_, "index"_, "countrycitizen"_, "schooldegree"_, "hourslearning"_,
        "monthsprogramming"_, "gender"_, "age"_, "moneyforlearning"_, "attendedbootcamp"_,
        "commutetime"_, "citypopulation"_, "income"_, "studentdebtowe"_, "childrennumber"_,
        "bootcampfinish"_, "bootcampfulljobafter"_, "bootcamploanyesno"_, "bootcamppostsalary"_)));

    checkForErrors(eval("CreateTable"_("GDP"_, "index"_, "gdp_per_capita"_, "country"_)));

    auto filenames = std::vector<std::string>{"fcc", "gdp"};
    auto tables = std::vector<boss::Symbol>{"FCC"_, "GDP"_};

    auto const& defaultMissing = bossImputationExpressions().find(imputationMethod)->second;

    for(int i = 0; i < tables.size(); ++i) {
      std::string path = "../data/imputedb/" + filenames[i] + ".csv";
      checkForErrors(eval("Load"_(tables[i], path, defaultMissing)));
    }

    eval("FinaliseTable"_("FCC"_));
    eval("FinaliseTable"_("GDP"_));
  }
};

static void initBOSSEngine_CDC(std::string const& engineLibrary, int batchSize,
                               int imputationMethod) {
  static auto dataset = std::string("CDC");
  if(lastEngineLibrary().empty() || engineLibrary != lastEngineLibrary() ||
     lastDataset() != dataset || batchSize != lastBatchSize() ||
     imputationMethod != lastImputationMethod()) {
    resetBOSSEngine();

    lastEngineLibrary() = engineLibrary;
    lastDataset() = dataset;
    lastBatchSize() = batchSize;
    lastImputationMethod() = imputationMethod;

    auto eval = [&engineLibrary](auto const& expression) mutable {
      return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
    };

    auto checkForErrors = [](auto&& output) {
      auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
      if(maybeComplexExpr == nullptr) {
        return;
      }
      if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
        std::cout << "Error: " << output << std::endl;
      }
    };

    checkForErrors(
        eval("Set"_("EnableOrderPreservationCache"_, BOSS_ENABLE_ORDER_PRESERVATION_CACHE)));
    checkForErrors(
        eval("Set"_("UseMemoryMappedFiles"_, false))); // cannot use cache for imputed data
    checkForErrors(eval("Set"_("MicroBatchesMaxSize"_, batchSize)));
    // eval("Set"_("milliSF"_, dataSize)); // for cardinality calculation

    checkForErrors(eval("CreateTable"_("DEMO"_, "income"_, "num_people_household"_,
                                       "marital_status"_, "age_yrs"_, "gender"_, "id"_,
                                       "is_citizen"_, "years_edu"_, "age_months"_, "time_in_us"_)));

    checkForErrors(
        eval("CreateTable"_("EXAMS"_, "body_mass_index"_, "blood_pressure_secs"_, "cuff_size"_,
                            "id"_, "waist_circumference"_, "height"_, "arm_circumference"_,
                            "blood_pressure_systolic"_, "weight"_, "head_circumference"_)));

    checkForErrors(eval("CreateTable"_("LABS"_, "white_blood_cell_ct"_, "id"_, "hematocrit"_,
                                       "cholesterol"_, "vitamin_b12"_, "albumin"_, "blood_lead"_,
                                       "blood_selenium"_, "triglyceride"_, "creatine"_)));

    auto filenames = std::vector<std::string>{"demo", "exams", "labs"};
    auto tables = std::vector<boss::Symbol>{"DEMO"_, "EXAMS"_, "LABS"_};

    auto const& defaultMissing = bossImputationExpressions().find(imputationMethod)->second;

    for(int i = 0; i < tables.size(); ++i) {
      std::string path = "../data/imputedb/" + filenames[i] + ".csv";
      checkForErrors(eval("Load"_(tables[i], path, defaultMissing)));
    }

    eval("FinaliseTable"_("DEMO"_));
    eval("FinaliseTable"_("EXAMS"_));
    eval("FinaliseTable"_("LABS"_));
  }
};

static void initBOSSEngine_ACS(std::string const& engineLibrary, int batchSize,
                               int imputationMethod) {
  static auto dataset = std::string("ACS");
  if(lastEngineLibrary().empty() || engineLibrary != lastEngineLibrary() ||
     lastDataset() != dataset || batchSize != lastBatchSize() ||
     imputationMethod != lastImputationMethod()) {
    resetBOSSEngine();

    lastEngineLibrary() = engineLibrary;
    lastDataset() = dataset;
    lastBatchSize() = batchSize;
    lastImputationMethod() = imputationMethod;

    auto eval = [&engineLibrary](auto const& expression) mutable {
      return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
    };

    auto checkForErrors = [](auto&& output) {
      auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
      if(maybeComplexExpr == nullptr) {
        return;
      }
      if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
        std::cout << "Error: " << output << std::endl;
      }
    };

    checkForErrors(
        eval("Set"_("EnableOrderPreservationCache"_, BOSS_ENABLE_ORDER_PRESERVATION_CACHE)));
    checkForErrors(
        eval("Set"_("UseMemoryMappedFiles"_, false))); // cannot use cache for imputed data
    checkForErrors(eval("Set"_("MicroBatchesMaxSize"_, batchSize)));
    // eval("Set"_("milliSF"_, dataSize)); // for cardinality calculation

    checkForErrors(eval(
        "CreateTable"_("ACS"_, "c0"_, "c1"_, "c2"_, "c3"_, "c4"_, "c5"_, "c6"_, "c7"_, "c8"_, "c9"_,
                       "c10"_, "c11"_, "c12"_, "c13"_, "c14"_, "c15"_, "c16"_, "c17"_, "c18"_,
                       "c19"_, "c20"_, "c21"_, "c22"_, "c23"_, "c24"_, "c25"_, "c26"_, "c27"_,
                       "c28"_, "c29"_, "c30"_, "c31"_, "c32"_, "c33"_, "c34"_, "c35"_, "c36"_)));

    auto filenames = std::vector<std::string>{"acs_dirty"};
    auto tables = std::vector<boss::Symbol>{"ACS"_};

    auto const& defaultMissing = bossImputationExpressions().find(imputationMethod)->second;

    for(int i = 0; i < tables.size(); ++i) {
      std::string path = "../data/imputedb/" + filenames[i] + ".csv";
      checkForErrors(eval("Load"_(tables[i], path, defaultMissing)));
    }
    
    eval("FinaliseTable"_("ACS"_));
  }
};

enum TPCH_QUERIES { TPCH_Q1 = 1, TPCH_Q3 = 3, TPCH_Q6 = 6, TPCH_Q9 = 9, TPCH_Q18 = 18 };

enum TPCH_COMMON_PLAN_QUERIES {
  TPCH_COMMON_PLAN_Q3 = 23,
  TPCH_COMMON_PLAN_Q9 = 29,
  TPCH_COMMON_PLAN_Q18 = 38
};

enum TPCH_IMPUTATION_QUERIES {
  TPCH_Q1_WITH_EVAL = 41,
  TPCH_Q3_WITH_EVAL = 43,
  TPCH_Q6_WITH_EVAL = 46,
  TPCH_Q9_WITH_EVAL = 49,
  TPCH_Q18_WITH_EVAL = 58,
  TPCH_Q1_MODIFIED = 60,
  TPCH_Q6_MODIFIED,
  TPCH_Q1_FULL_EVAL,
  CDC_Q1,
  CDC_Q2,
  CDC_Q3,
  CDC_Q4,
  CDC_Q5,
  FCC_Q6,
  FCC_Q7,
  FCC_Q8,
  FCC_Q9,
  ACS
};

enum MICRO_BENCHMARKING {
  MICRO_BENCHMARKING_START = 100,

  EVALUATE_INT_ADDITIONS = MICRO_BENCHMARKING_START,
  EVALUATE_NESTED_INT_ADDITIONS,
  EVALUATE_INT_MULTIPLICATIONS,
  EVALUATE_INT_ADDMUL,
  EVALUATE_INT_ADD_CONSTANT,
  EVALUATE_INT_MULT_CONSTANT,
  EVALUATE_INT_ADDMULT_CONSTANTS,
  EVALUATE_FLOAT_ADDITIONS,
  EVALUATE_FLOAT_MULTIPLICATIONS,
  EVALUATE_FLOAT_ADDMUL,
  EVALUATE_FLOAT_ADD_CONSTANT,
  EVALUATE_FLOAT_MULT_CONSTANT,
  EVALUATE_FLOAT_ADDMULT_CONSTANTS,
  EVALUATE_STRING_JOIN,

  AGGREGATE_COUNT,
  AGGREGATE_SUM_INTS,
  AGGREGATE_SUM_FLOATS,
  AGGREGATE_AVERAGE_INTS,
  AGGREGATE_AVERAGE_FLOATS,

  PROJECT_5_DIFFERENT_COLUMNS,
  PROJECT_10_DIFFERENT_COLUMNS,
  PROJECT_15_DIFFERENT_COLUMNS,
  PROJECT_30_INT_COLUMNS,
  PROJECT_30_LARGE_STRING_COLUMNS,

  SELECT_LOW_INT,
  SELECT_LOW_FLOAT,
  SELECT_LOW_DATE,
  SELECT_LOW_STRING,
  SELECT_LOW_LARGE_STRING,
  // SELECT_LOW_MULTIPLE_INTS,
  // SELECT_LOW_MULTIPLE_FLOATS,
  // SELECT_LOW_MULTIPLE_STRINGS,
  // SELECT_LOW_INT_FLOAT_STRING,

  SELECT_HALF_INT,
  SELECT_HALF_FLOAT,
  SELECT_HALF_DATE,
  SELECT_HALF_STRING,
  SELECT_HALF_LARGE_STRING,
  // SELECT_HALF_MULTIPLE_INTS,
  // SELECT_HALF_MULTIPLE_FLOATS,
  SELECT_HALF_MULTIPLE_STRINGS,
  // SELECT_HALF_INT_FLOAT_STRING,

  SELECT_HIGH_INT,
  SELECT_HIGH_FLOAT,
  SELECT_HIGH_DATE,
  SELECT_HIGH_STRING,
  SELECT_HIGH_LARGE_STRING,
  // SELECT_HIGH_MULTIPLE_INTS,
  // SELECT_HIGH_MULTIPLE_FLOATS,
  // SELECT_HIGH_MULTIPLE_STRINGS,
  // SELECT_HIGH_INT_FLOAT_STRING,

  JOIN_INTS,
  JOIN_MANY_INTS,
  // JOIN_FLOAT,
  // JOIN_STRING,
  // JOIN_MULTIPLE_INTS,
  // JOIN_MULTIPLE_FLOATS,
  // JOIN_MULTIPLE_STRINGS,
  // JOIN_INT_FLOAT_STRING,

  GROUP_BY_MANY_INTS,
  GROUP_BY_FEW_INTS,
  // GROUP_BY_FLOAT,
  GROUP_BY_STRING,
  GROUP_BY_MULTIPLE_INTS,
  // GROUP_BY_MULTIPLE_FLOATS,
  GROUP_BY_MULTIPLE_STRINGS,
  // GROUP_BY_INT_FLOAT_STRING,

  TOP10_BY_INT,
  TOP10_BY_FLOAT,
  TOP10_BY_STRING,
  TOP10_BY_MULTIPLE_INTS,
  TOP10_BY_MULTIPLE_STRINGS,

  TOP100_BY_INT,
  TOP100_BY_FLOAT,
  TOP100_BY_STRING,
  TOP100_BY_MULTIPLE_INTS,
  TOP100_BY_MULTIPLE_STRINGS,

  TOP500_BY_INT,
  TOP500_BY_FLOAT,
  TOP500_BY_STRING,
  TOP500_BY_MULTIPLE_INTS,
  TOP500_BY_MULTIPLE_STRINGS,

  Q1_SELECT,
  Q1_CALC,
  Q1_PROJECT,
  Q1_SELECT_PROJECT,
  Q1_SUMS,
  Q1_GROUP_BY,
  Q1_SORT_BY,
  Q1_GROUP_SUM,
  Q1_GROUP_AVG,

  MICRO_BENCHMARKING_END
};

static auto& queryNames() {
  static std::map<int, std::string> names;
  if(names.empty()) {
    // Base features
    names.try_emplace(EVALUATE_INT_ADDITIONS, "Evaluate Int Additions");
    names.try_emplace(EVALUATE_NESTED_INT_ADDITIONS, "Evaluate Nested Int Additions");
    names.try_emplace(EVALUATE_INT_MULTIPLICATIONS, "Evaluate Int Multiplications");
    names.try_emplace(EVALUATE_INT_ADDMUL, "Evaluate Int Addmul");
    names.try_emplace(EVALUATE_INT_ADD_CONSTANT, "Evaluate Int Add Constant");
    names.try_emplace(EVALUATE_INT_MULT_CONSTANT, "Evaluate Int Mult Constant");
    names.try_emplace(EVALUATE_INT_ADDMULT_CONSTANTS, "Evaluate Int Addmult Constants");
    names.try_emplace(EVALUATE_FLOAT_ADDITIONS, "Evaluate Float Additions");
    names.try_emplace(EVALUATE_FLOAT_MULTIPLICATIONS, "Evaluate Float Multiplications");
    names.try_emplace(EVALUATE_FLOAT_ADDMUL, "Evaluate Float Addmul");
    names.try_emplace(EVALUATE_FLOAT_ADD_CONSTANT, "Evaluate Float Add Constant");
    names.try_emplace(EVALUATE_FLOAT_MULT_CONSTANT, "Evaluate Float Mult Constant");
    names.try_emplace(EVALUATE_FLOAT_ADDMULT_CONSTANTS, "Evaluate Float Addmult Constants");
    names.try_emplace(EVALUATE_STRING_JOIN, "Evaluate String Join");
    names.try_emplace(AGGREGATE_COUNT, "Aggregate Count");
    names.try_emplace(AGGREGATE_SUM_INTS, "Aggregate Sum Ints");
    names.try_emplace(AGGREGATE_SUM_FLOATS, "Aggregate Sum Floats");
    names.try_emplace(AGGREGATE_AVERAGE_INTS, "Aggregate Average Ints");
    names.try_emplace(AGGREGATE_AVERAGE_FLOATS, "Aggregate Average Floats");
    names.try_emplace(PROJECT_5_DIFFERENT_COLUMNS, "Project 5 Different Columns");
    names.try_emplace(PROJECT_10_DIFFERENT_COLUMNS, "Project 10 Different Columns");
    names.try_emplace(PROJECT_15_DIFFERENT_COLUMNS, "Project 15 Different Columns");
    names.try_emplace(PROJECT_30_INT_COLUMNS, "Project 30 Int Columns");
    names.try_emplace(PROJECT_30_LARGE_STRING_COLUMNS, "Project 30 Large String Columns");
    names.try_emplace(SELECT_LOW_INT, "Select Int with 5% selectivity");
    names.try_emplace(SELECT_LOW_FLOAT, "Select Float with 5% selectivity");
    names.try_emplace(SELECT_LOW_DATE, "Select Date with 5% selectivity");
    names.try_emplace(SELECT_LOW_STRING, "Select String with 5% selectivity");
    names.try_emplace(SELECT_LOW_LARGE_STRING, "Select From Large String with 5% selectivity");
    names.try_emplace(SELECT_HALF_INT, "Select Int with 50% selectivity");
    names.try_emplace(SELECT_HALF_FLOAT, "Select Float with 50% selectivity");
    names.try_emplace(SELECT_HALF_DATE, "Select Date with 50% selectivity");
    names.try_emplace(SELECT_HALF_STRING, "Select String with 50% selectivity");
    names.try_emplace(SELECT_HALF_LARGE_STRING, "Select From Large String with 50% selectivity");
    names.try_emplace(SELECT_HALF_MULTIPLE_STRINGS, "Select Multiple Strings with 50% selectivity");
    names.try_emplace(SELECT_HIGH_INT, "Select Int with 95% selectivity");
    names.try_emplace(SELECT_HIGH_FLOAT, "Select Float with 95% selectivity");
    names.try_emplace(SELECT_HIGH_DATE, "Select Date with 95% selectivity");
    names.try_emplace(SELECT_HIGH_STRING, "Select String with 95% selectivity");
    names.try_emplace(SELECT_HIGH_LARGE_STRING, "Select From Large String with 95% selectivity");
    names.try_emplace(JOIN_INTS, "Join Ints");
    names.try_emplace(JOIN_MANY_INTS, "Join Many Ints");
    names.try_emplace(GROUP_BY_MANY_INTS, "Group By Many Ints");
    names.try_emplace(GROUP_BY_FEW_INTS, "Group By Few Ints");
    names.try_emplace(GROUP_BY_STRING, "Group By String");
    names.try_emplace(GROUP_BY_MULTIPLE_INTS, "Group By Multiple Ints");
    names.try_emplace(GROUP_BY_MULTIPLE_STRINGS, "Group By Multiple Strings");
    names.try_emplace(TOP10_BY_INT, "Top 10 By Int");
    names.try_emplace(TOP10_BY_FLOAT, "Top 10 By Float");
    names.try_emplace(TOP10_BY_STRING, "Top 10 By String");
    names.try_emplace(TOP10_BY_MULTIPLE_INTS, "Top 10 By Multiple Ints");
    names.try_emplace(TOP10_BY_MULTIPLE_STRINGS, "Top 10 By Multiple Strings");
    names.try_emplace(TOP100_BY_INT, "Top 100 By Int");
    names.try_emplace(TOP100_BY_FLOAT, "Top 100 By Float");
    names.try_emplace(TOP100_BY_STRING, "Top 100 By String");
    names.try_emplace(TOP100_BY_MULTIPLE_INTS, "Top 100 By Multiple Ints");
    names.try_emplace(TOP100_BY_MULTIPLE_STRINGS, "Top 100 By Multiple Strings");
    names.try_emplace(TOP500_BY_INT, "Top 500 By Int");
    names.try_emplace(TOP500_BY_FLOAT, "Top 500 By Float");
    names.try_emplace(TOP500_BY_STRING, "Top 500 By String");
    names.try_emplace(TOP500_BY_MULTIPLE_INTS, "Top 500 By Multiple Ints");
    names.try_emplace(TOP500_BY_MULTIPLE_STRINGS, "Top 500 By Multiple Strings");

    // Break-down Q1
    names.try_emplace(Q1_SELECT, "Q1 Select");
    names.try_emplace(Q1_CALC, "Q1 Calc");
    names.try_emplace(Q1_PROJECT, "Q1 Project");
    names.try_emplace(Q1_SELECT_PROJECT, "Q1 Select+Project");
    names.try_emplace(Q1_SUMS, "Q1 Sums");
    names.try_emplace(Q1_GROUP_BY, "Q1 GroupBy");
    names.try_emplace(Q1_SORT_BY, "Q1 SortBy");
    names.try_emplace(Q1_GROUP_SUM, "Q1 Group+Sum");
    names.try_emplace(Q1_GROUP_AVG, "Q1 Group+Avg");

    // TPC-H
    names.try_emplace(TPCH_Q1, "TPC-H Q1");
    names.try_emplace(TPCH_Q3, "TPC-H Q3");
    names.try_emplace(TPCH_Q6, "TPC-H Q6");
    names.try_emplace(TPCH_Q9, "TPC-H Q9");
    names.try_emplace(TPCH_Q18, "TPC-H Q18");

    // TPC-H with common plan (i.e., DuckDB plan) for BOSS and MonetDB
    names.try_emplace(TPCH_COMMON_PLAN_Q3, "TPC-H CommonPlan Q3");
    names.try_emplace(TPCH_COMMON_PLAN_Q9, "TPC-H CommonPlan Q9");
    names.try_emplace(TPCH_COMMON_PLAN_Q18, "TPC-H CommonPlan Q18");

    // TPC-H modified (for imputation)
    names.try_emplace(TPCH_Q1_WITH_EVAL, "TPC-H-Q1-EVAL");
    names.try_emplace(TPCH_Q3_WITH_EVAL, "TPC-H-Q3-EVAL");
    names.try_emplace(TPCH_Q6_WITH_EVAL, "TPC-H-Q6-EVAL");
    names.try_emplace(TPCH_Q9_WITH_EVAL, "TPC-H-Q9-EVAL");
    names.try_emplace(TPCH_Q18_WITH_EVAL, "TPC-H-Q18-EVAL");
    names.try_emplace(TPCH_Q1_MODIFIED, "TPCH-Q1-MODIFIED");
    names.try_emplace(TPCH_Q6_MODIFIED, "TPCH-Q6-MODIFIED");
    names.try_emplace(TPCH_Q1_FULL_EVAL, "TPCH-Q1-FULL-EVAL");
    names.try_emplace(CDC_Q1, "CDC-Q1");
    names.try_emplace(CDC_Q2, "CDC-Q2");
    names.try_emplace(CDC_Q3, "CDC-Q3");
    names.try_emplace(CDC_Q4, "CDC-Q4");
    names.try_emplace(CDC_Q5, "CDC-Q5");
    names.try_emplace(FCC_Q6, "FCC-Q6");
    names.try_emplace(FCC_Q7, "FCC-Q7");
    names.try_emplace(FCC_Q8, "FCC-Q8");
    names.try_emplace(FCC_Q9, "FCC-Q9");
    names.try_emplace(ACS, "ACS");
  }
  return names;
}

static auto& bossQueries() {
  static std::map<int, boss::Expression> queries;
  if(queries.empty()) {
    queries.try_emplace(
        EVALUATE_INT_ADDITIONS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("L_ORDERKEY"_, "L_PARTKEY"_))));
    queries.try_emplace(
        EVALUATE_NESTED_INT_ADDITIONS,
        "Project"_(
            "Project"_(
                "Project"_(
                    "Project"_(
                        "Project"_(
                            "Project"_(
                                "Project"_(
                                    "Project"_(
                                        "Project"_("Project"_("LINEITEM"_,
                                                              "As"_("SUM"_, "Plus"_("L_ORDERKEY"_,
                                                                                    "L_PARTKEY"_))),
                                                   "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                                        "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                                    "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                                "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                            "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                        "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                    "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
                "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))),
            "As"_("SUM"_, "Plus"_("SUM"_, "SUM"_))));
    queries.try_emplace(
        EVALUATE_INT_MULTIPLICATIONS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Times"_("L_ORDERKEY"_, "L_PARTKEY"_))));
    queries.try_emplace(
        EVALUATE_INT_ADDMUL,
        "Project"_("LINEITEM"_,
                   "As"_("SUM"_, "Plus"_("L_ORDERKEY"_, "Times"_("L_PARTKEY"_, "L_SUPPKEY"_)))));
    queries.try_emplace(EVALUATE_INT_ADD_CONSTANT,
                        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("L_ORDERKEY"_, 1))));
    queries.try_emplace(EVALUATE_INT_MULT_CONSTANT,
                        "Project"_("LINEITEM"_, "As"_("SUM"_, "Times"_("L_ORDERKEY"_, 10))));
    queries.try_emplace(
        EVALUATE_INT_ADDMULT_CONSTANTS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("Times"_("L_ORDERKEY"_, 10), 1000))));
    queries.try_emplace(
        EVALUATE_FLOAT_ADDITIONS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("L_QUANTITY"_, "L_EXTENDEDPRICE"_))));
    queries.try_emplace(
        EVALUATE_FLOAT_MULTIPLICATIONS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Times"_("L_QUANTITY"_, "L_EXTENDEDPRICE"_))));
    queries.try_emplace(
        EVALUATE_FLOAT_ADDMUL,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("L_QUANTITY"_, "Times"_("L_EXTENDEDPRICE"_,
                                                                              "L_DISCOUNT"_)))));
    queries.try_emplace(EVALUATE_FLOAT_ADD_CONSTANT,
                        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("L_QUANTITY"_, 1.0))));
    queries.try_emplace(EVALUATE_FLOAT_MULT_CONSTANT,
                        "Project"_("LINEITEM"_, "As"_("SUM"_, "Times"_("L_QUANTITY"_, 10.0))));
    queries.try_emplace(
        EVALUATE_FLOAT_ADDMULT_CONSTANTS,
        "Project"_("LINEITEM"_, "As"_("SUM"_, "Plus"_("Times"_("L_QUANTITY"_, 10.0), 1000.0))));
    queries.try_emplace(
        EVALUATE_STRING_JOIN,
        "Project"_("LINEITEM"_, "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                      "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_))));

    queries.try_emplace(AGGREGATE_COUNT,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                                 "Count"_("L_ORDERKEY"_)));
    queries.try_emplace(AGGREGATE_SUM_INTS,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                                 "Sum"_("L_ORDERKEY"_)));
    queries.try_emplace(AGGREGATE_SUM_FLOATS,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_)),
                                 "Sum"_("L_QUANTITY"_)));
    queries.try_emplace(
        AGGREGATE_AVERAGE_INTS,
        "Project"_(
            "Group"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                     "As"_("SumKeys"_, "Sum"_("L_ORDERKEY"_), "Total"_, "Count"_("L_ORDERKEY"_))),
            "As"_("AVG_KEYS"_, "Divide"_("SumKeys"_, "Total"_))));
    queries.try_emplace(
        AGGREGATE_AVERAGE_FLOATS,
        "Project"_("Group"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_)),
                            "As"_("SumQuantity"_, "Sum"_("L_QUANTITY"_), "Total"_,
                                  "Count"_("L_QUANTITY"_))),
                   "As"_("AVG_QUANTITY"_, "Divide"_("SumQuantity"_, "Total"_))));
    queries.try_emplace(
        PROJECT_5_DIFFERENT_COLUMNS,
        "Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_PARTKEY"_,
                                      "L_SUPPKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                                      "L_QUANTITY"_, "L_QUANTITY"_)));

    queries.try_emplace(
        PROJECT_10_DIFFERENT_COLUMNS,
        "Project"_("LINEITEM"_,
                   "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                         "L_SUPPKEY"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_QUANTITY"_,
                         "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                         "L_DISCOUNT"_, "L_TAX"_, "L_TAX"_, "L_RETURNFLAG"_, "L_RETURNFLAG"_,
                         "L_LINESTATUS"_, "L_LINESTATUS"_)));
    queries.try_emplace(
        PROJECT_15_DIFFERENT_COLUMNS,
        "Project"_("LINEITEM"_,
                   "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                         "L_SUPPKEY"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_QUANTITY"_,
                         "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                         "L_DISCOUNT"_, "L_TAX"_, "L_TAX"_, "L_RETURNFLAG"_, "L_RETURNFLAG"_,
                         "L_LINESTATUS"_, "L_LINESTATUS"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                         "L_COMMITDATE"_, "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_RECEIPTDATE"_,
                         "L_SHIPINSTRUCT"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_SHIPMODE"_)));
    queries.try_emplace(
        PROJECT_30_INT_COLUMNS,
        "Project"_("LINEITEM"_,
                   "As"_("L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_,
                         "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_, "L_LINENUMBER"_)));
    queries.try_emplace(
        PROJECT_30_LARGE_STRING_COLUMNS,
        "Project"_("LINEITEM"_,
                   "As"_("L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_,
                         "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_, "L_COMMENT"_)));

    queries.try_emplace(SELECT_LOW_INT,
                        "Select"_("Project"_("PARTSUPP"_, "As"_("PS_AVAILQTY"_, "PS_AVAILQTY"_)),
                                  "Where"_("Greater"_(600, "PS_AVAILQTY"_))));
    queries.try_emplace(
        SELECT_LOW_FLOAT,
        "Select"_("Project"_("LINEITEM"_, "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                  "Where"_("Greater"_(4300.0, "L_EXTENDEDPRICE"_))));
    queries.try_emplace(SELECT_LOW_DATE,
                        "Select"_("Project"_("LINEITEM"_, "As"_("L_SHIPDATE"_, "L_SHIPDATE"_)),
                                  "Where"_("Greater"_("DateObject"_("1992-7-01"), "L_SHIPDATE"_))));
    queries.try_emplace(SELECT_LOW_STRING,
                        "Select"_("Project"_("ORDERS"_, "As"_("O_ORDERSTATUS"_, "O_ORDERSTATUS"_)),
                                  "Where"_("StringContainsQ"_("O_ORDERSTATUS"_, "P"))));
    queries.try_emplace(
        SELECT_LOW_LARGE_STRING,
        "Project"_("Select"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_,
                                                           "L_COMMENT"_, "L_COMMENT"_)),
                             "Where"_("StringContainsQ"_("L_COMMENT"_, "fo"))),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_)));

    queries.try_emplace(SELECT_HALF_INT,
                        "Select"_("Project"_("PARTSUPP"_, "As"_("PS_AVAILQTY"_, "PS_AVAILQTY"_)),
                                  "Where"_("Greater"_(5100, "PS_AVAILQTY"_))));
    queries.try_emplace(
        SELECT_HALF_FLOAT,
        "Select"_("Project"_("LINEITEM"_, "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                  "Where"_("Greater"_(37000.0, "L_EXTENDEDPRICE"_))));
    queries.try_emplace(SELECT_HALF_DATE,
                        "Select"_("Project"_("LINEITEM"_, "As"_("L_SHIPDATE"_, "L_SHIPDATE"_)),
                                  "Where"_("Greater"_("DateObject"_("1995-7-01"), "L_SHIPDATE"_))));
    queries.try_emplace(
        SELECT_HALF_STRING,
        "Project"_("Select"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_,
                                                           "L_RETURNFLAG"_, "L_RETURNFLAG"_)),
                             "Where"_("StringContainsQ"_("L_RETURNFLAG"_, "N"))),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_)));
    queries.try_emplace(
        SELECT_HALF_LARGE_STRING,
        "Project"_("Select"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_,
                                                           "L_COMMENT"_, "L_COMMENT"_)),
                             "Where"_("StringContainsQ"_("L_COMMENT"_, "p"))),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_)));
    queries.try_emplace(
        SELECT_HALF_MULTIPLE_STRINGS,
        "Project"_("Select"_("Project"_("LINEITEM"_,
                                        "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_RETURNFLAG"_,
                                              "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_)),
                             "Where"_("And"_("StringContainsQ"_("L_RETURNFLAG"_, "N"),
                                             "StringContainsQ"_("L_LINESTATUS"_, "O")))),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_)));

    queries.try_emplace(SELECT_HIGH_INT,
                        "Select"_("Project"_("PARTSUPP"_, "As"_("PS_AVAILQTY"_, "PS_AVAILQTY"_)),
                                  "Where"_("Greater"_(9600, "PS_AVAILQTY"_))));
    queries.try_emplace(
        SELECT_HIGH_FLOAT,
        "Select"_("Project"_("LINEITEM"_, "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                  "Where"_("Greater"_(80000.0, "L_EXTENDEDPRICE"_))));
    queries.try_emplace(SELECT_HIGH_DATE,
                        "Select"_("Project"_("LINEITEM"_, "As"_("L_SHIPDATE"_, "L_SHIPDATE"_)),
                                  "Where"_("Greater"_("DateObject"_("1998-6-15"), "L_SHIPDATE"_))));
    queries.try_emplace(SELECT_HIGH_STRING,
                        "Select"_("Project"_("ORDERS"_, "As"_("O_ORDERSTATUS"_, "O_ORDERSTATUS"_)),
                                  "Where"_("Not"_("StringContainsQ"_("O_ORDERSTATUS"_, "P")))));
    queries.try_emplace(
        SELECT_HIGH_LARGE_STRING,
        "Project"_("Select"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_,
                                                           "L_COMMENT"_, "L_COMMENT"_)),
                             "Where"_("StringContainsQ"_("L_COMMENT"_, "e"))),
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_)));

    queries.try_emplace(JOIN_INTS,
                        "Join"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                                "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_)),
                                "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))));
    queries.try_emplace(JOIN_MANY_INTS,
                        "Join"_("Project"_("LINEITEM"_, "As"_("L_SUPPKEY"_, "L_SUPPKEY"_)),
                                "Project"_("PARTSUPP"_, "As"_("PS_SUPPKEY"_, "PS_SUPPKEY"_)),
                                "Where"_("Equal"_("L_SUPPKEY"_, "PS_SUPPKEY"_))));

    queries.try_emplace(GROUP_BY_MANY_INTS,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                                 "By"_("L_ORDERKEY"_)));
    queries.try_emplace(
        GROUP_BY_FEW_INTS,
        "Group"_("Project"_("LINEITEM"_, "As"_("L_SUPPKEY"_, "L_SUPPKEY"_)), "By"_("L_SUPPKEY"_)));
    queries.try_emplace(GROUP_BY_STRING,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_)),
                                 "By"_("L_RETURNFLAG"_)));
    queries.try_emplace(GROUP_BY_MULTIPLE_INTS,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                               "L_SUPPKEY"_, "L_SUPPKEY"_)),
                                 "By"_("L_ORDERKEY"_, "L_SUPPKEY"_)));
    queries.try_emplace(GROUP_BY_MULTIPLE_STRINGS,
                        "Group"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                               "L_LINESTATUS"_, "L_LINESTATUS"_)),
                                 "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_)));

    queries.try_emplace(TOP10_BY_INT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                               "By"_("L_ORDERKEY"_), 10));
    queries.try_emplace(TOP10_BY_FLOAT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_)),
                               "By"_("L_QUANTITY"_), 10));
    queries.try_emplace(TOP10_BY_STRING,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_)),
                               "By"_("L_RETURNFLAG"_), 10));
    queries.try_emplace(TOP10_BY_MULTIPLE_INTS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                             "L_SUPPKEY"_, "L_SUPPKEY"_)),
                               "By"_("L_ORDERKEY"_, "L_SUPPKEY"_), 10));
    queries.try_emplace(TOP10_BY_MULTIPLE_STRINGS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                             "L_LINESTATUS"_, "L_LINESTATUS"_)),
                               "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_), 10));

    queries.try_emplace(TOP100_BY_INT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                               "By"_("L_ORDERKEY"_), 100));
    queries.try_emplace(TOP100_BY_FLOAT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_)),
                               "By"_("L_QUANTITY"_), 100));
    queries.try_emplace(TOP100_BY_STRING,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_)),
                               "By"_("L_RETURNFLAG"_), 100));
    queries.try_emplace(TOP100_BY_MULTIPLE_INTS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                             "L_SUPPKEY"_, "L_SUPPKEY"_)),
                               "By"_("L_ORDERKEY"_, "L_SUPPKEY"_), 100));
    queries.try_emplace(TOP100_BY_MULTIPLE_STRINGS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                             "L_LINESTATUS"_, "L_LINESTATUS"_)),
                               "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_), 100));

    queries.try_emplace(TOP500_BY_INT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_)),
                               "By"_("L_ORDERKEY"_), 500));
    queries.try_emplace(TOP500_BY_FLOAT,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_)),
                               "By"_("L_QUANTITY"_), 500));
    queries.try_emplace(TOP500_BY_STRING,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_)),
                               "By"_("L_RETURNFLAG"_), 500));
    queries.try_emplace(TOP500_BY_MULTIPLE_INTS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                             "L_SUPPKEY"_, "L_SUPPKEY"_)),
                               "By"_("L_ORDERKEY"_, "L_SUPPKEY"_), 500));
    queries.try_emplace(TOP500_BY_MULTIPLE_STRINGS,
                        "Top"_("Project"_("LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                             "L_LINESTATUS"_, "L_LINESTATUS"_)),
                               "By"_("L_RETURNFLAG"_, "L_LINESTATUS"_), 500));

    queries.try_emplace(
        Q1_SELECT, "Select"_("Project"_("LINEITEM"_, "As"_("L_SHIPDATE"_, "L_SHIPDATE"_)),
                             "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))));
    queries.try_emplace(
        Q1_CALC, "Project"_("Project"_("LINEITEM"_, "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                          "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_),
                                                          "calc2"_, "Plus"_(1.0, "L_TAX"_))),
                            "As"_("disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                                  "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_))));
    queries.try_emplace(
        Q1_PROJECT,
        "Project"_("Project"_("LINEITEM"_,
                              "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                    "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                    "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "calc1"_,
                                    "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_),
                                    "L_DISCOUNT"_, "L_DISCOUNT"_)),
                   "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "L_QUANTITY"_,
                         "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "disc_price"_,
                         "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                         "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                         "L_DISCOUNT"_)));
    queries.try_emplace(
        Q1_SELECT_PROJECT,
        "Project"_(
            "Project"_("Select"_("Project"_("LINEITEM"_,
                                            "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                  "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                  "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                  "L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_,
                                                  "L_LINESTATUS"_, "L_TAX"_, "L_TAX"_)),
                                 "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                       "As"_("RETURNFLAG_AND_LINESTATUS"_,
                             "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                             "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "calc1"_,
                             "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_),
                             "L_DISCOUNT"_, "L_DISCOUNT"_)),
            "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "L_QUANTITY"_,
                  "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "disc_price"_,
                  "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                  "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_, "L_DISCOUNT"_)));
    queries.try_emplace(
        Q1_SUMS,
        "Group"_(
            "Project"_(
                "Project"_("LINEITEM"_,
                           "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                                 "L_EXTENDEDPRICE"_, "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_),
                                 "calc2"_, "Plus"_(1.0, "L_TAX"_), "L_DISCOUNT"_, "L_DISCOUNT"_)),
                "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                      "disc_price"_, "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "charge"_,
                      "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                      "L_DISCOUNT"_)),
            "Sum"_("L_QUANTITY"_), "Sum"_("L_EXTENDEDPRICE"_), "Sum"_("disc_price"_),
            "Sum"_("charge"_), "Sum"_("L_DISCOUNT"_)));
    queries.try_emplace(
        Q1_GROUP_BY,
        "Group"_("Project"_("LINEITEM"_, "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                               "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_))),
                 "By"_("RETURNFLAG_AND_LINESTATUS"_)));
    queries.try_emplace(
        Q1_SORT_BY, "Sort"_("Group"_("Project"_("LINEITEM"_, "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                                                   "StringJoin"_("L_RETURNFLAG"_,
                                                                                 "L_LINESTATUS"_))),
                                     "By"_("RETURNFLAG_AND_LINESTATUS"_)),
                            "By"_("RETURNFLAG_AND_LINESTATUS"_)));
    queries.try_emplace(
        Q1_GROUP_SUM,
        "Group"_(
            "Project"_(
                "Project"_("LINEITEM"_,
                           "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                 "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                 "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "calc1"_,
                                 "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_),
                                 "L_DISCOUNT"_, "L_DISCOUNT"_)),
                "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "L_QUANTITY"_,
                      "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "DISC_PRICE"_,
                      "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "CHARGES"_,
                      "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                      "L_DISCOUNT"_)),
            "By"_("RETURNFLAG_AND_LINESTATUS"_),
            "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_, "Sum"_("L_EXTENDEDPRICE"_),
                  "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_), "SUM_CHARGES"_, "Sum"_("CHARGES"_),
                  "SUM_DISC"_, "Sum"_("L_DISCOUNT"_), "COUNT_ORDER"_, "Count"_("L_QUANTITY"_))));
    queries.try_emplace(
        Q1_GROUP_AVG,
        "Project"_(
            "Group"_(
                "Project"_(
                    "Project"_("LINEITEM"_,
                               "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                     "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                     "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                     "calc1"_, "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_,
                                     "Plus"_(1.0, "L_TAX"_), "L_DISCOUNT"_, "L_DISCOUNT"_)),
                    "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "L_QUANTITY"_,
                          "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "DISC_PRICE"_,
                          "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "CHARGES"_,
                          "Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_), "L_DISCOUNT"_,
                          "L_DISCOUNT"_)),
                "By"_("RETURNFLAG_AND_LINESTATUS"_),
                "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                      "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_),
                      "SUM_CHARGES"_, "Sum"_("CHARGES"_), "SUM_DISC"_, "Sum"_("L_DISCOUNT"_),
                      "COUNT_ORDER"_, "Count"_("L_QUANTITY"_))),
            "As"_("SUM_QTY"_, "SUM_QTY"_, "SUM_BASE_PRICE"_, "SUM_BASE_PRICE"_, "SUM_DISC_PRICE"_,
                  "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
                  "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
                  "Divide"_("SUM_BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
                  "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)));

    // TPC-H (without missing values)
    queries.try_emplace(
        TPCH_Q1,
        "Sort"_(
            "Project"_(
                "Group"_(
                    "Project"_(
                        "Project"_(
                            "Select"_("Project"_("LINEITEM"_,
                                                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                       "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                       "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                       "L_RETURNFLAG"_, "L_RETURNFLAG"_,
                                                       "L_LINESTATUS"_, "L_LINESTATUS"_, "L_TAX"_,
                                                       "L_TAX"_)),
                                      "Where"_(
                                          "Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_))),
                            "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                  "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                  "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                  "L_DISCOUNT"_, "L_DISCOUNT"_, "calc1"_,
                                  "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_))),
                        "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_,
                              "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                              "L_DISCOUNT"_, "L_DISCOUNT"_, "DISC_PRICE"_,
                              "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc"_, "calc2"_)),
                    "By"_("RETURNFLAG_AND_LINESTATUS"_),
                    "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                          "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_),
                          "SUM_CHARGES"_, "Sum"_("Times"_("DISC_PRICE"_, "calc"_)), "SUM_DISC"_,
                          "Sum"_("L_DISCOUNT"_), "COUNT_ORDER"_, "Count"_("L_QUANTITY"_))),
                "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "SUM_QTY"_,
                      "SUM_QTY"_, "SUM_BASE_PRICE"_, "SUM_BASE_PRICE"_, "SUM_DISC_PRICE"_,
                      "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
                      "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
                      "Divide"_("SUM_BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
                      "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)),
            "By"_("RETURNFLAG_AND_LINESTATUS"_)));
    queries.try_emplace(
        TPCH_Q3,
        "Top"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_(
                            "Join"_("Project"_(
                                        "Select"_("Project"_("CUSTOMER"_,
                                                             "As"_("C_CUSTKEY"_, "C_CUSTKEY"_,
                                                                   "C_MKTSEGMENT"_,
                                                                   "C_MKTSEGMENT"_)),
                                                  "Where"_("StringContainsQ"_("C_MKTSEGMENT"_,
                                                                              "BUILDING"))),
                                        "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
                                    "Select"_(
                                        "Project"_("ORDERS"_,
                                                   "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                         "O_ORDERDATE"_,
                                                         "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                         "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                                        "Where"_("Greater"_("DateObject"_("1995-03-15"),
                                                            "O_ORDERDATE"_))),
                                    "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_)),
                                    "Times"_("milliSF"_, 31)),
                            "As"_("O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_,
                                  "O_SHIPPRIORITY"_)),
                        "Project"_(
                            "Select"_(
                                "Project"_("LINEITEM"_,
                                           "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_,
                                                 "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                 "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1995-03-15")))),
                            "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                  "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                        "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_)),
                        "Times"_("milliSF"_, 750)),
                    "As"_("INV_DISC"_, "Minus"_(1.0, "L_DISCOUNT"_), "L_EXTENDEDPRICE"_,
                          "L_EXTENDEDPRICE"_, "L_ORDERKEY"_, "L_ORDERKEY"_, "O_ORDERDATE"_,
                          "O_ORDERDATE"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_),
                "As"_("revenue"_, "Sum"_("Times"_("L_EXTENDEDPRICE"_, "INV_DISC"_)))),
            "By"_(/*"Minus"_(*/ "revenue"_ /*)*/, "O_ORDERDATE"_), 10));
    queries.try_emplace(
        TPCH_Q6,
        "Group"_(
            "Project"_(
                "Select"_("Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                        "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                        "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                          "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                          "Greater"_("L_DISCOUNT"_, 0.0499),
                                          "Greater"_(0.07001, "L_DISCOUNT"_),
                                          "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                          "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
                "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
            "Sum"_("revenue"_)));
    queries.try_emplace(
        TPCH_Q9,
        "Sort"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                    "O_ORDERDATE"_)),
                        "Project"_(
                            "Join"_(
                                "Project"_(
                                    "Join"_(
                                        "Project"_(
                                            "Select"_("Project"_("PART"_,
                                                                 "As"_("P_PARTKEY"_, "P_PARTKEY"_,
                                                                       "P_NAME"_, "P_NAME"_)),
                                                      "Where"_(
                                                          "StringContainsQ"_("P_NAME"_, "green"))),
                                            "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                        "Project"_(
                                            "Join"_(
                                                "Project"_(
                                                    "Join"_(
                                                        "Project"_("NATION"_,
                                                                   "As"_("N_NAME"_, "N_NAME"_,
                                                                         "N_NATIONKEY"_,
                                                                         "N_NATIONKEY"_)),
                                                        "Project"_("SUPPLIER"_,
                                                                   "As"_("S_SUPPKEY"_, "S_SUPPKEY"_,
                                                                         "S_NATIONKEY"_,
                                                                         "S_NATIONKEY"_)),
                                                        "Where"_("Equal"_("N_NATIONKEY"_,
                                                                          "S_NATIONKEY"_)),
                                                        25),
                                                    "As"_("N_NAME"_, "N_NAME"_, "S_SUPPKEY"_,
                                                          "S_SUPPKEY"_)),
                                                "Project"_("PARTSUPP"_,
                                                           "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                 "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                 "PS_SUPPLYCOST"_,
                                                                 "PS_SUPPLYCOST"_)),
                                                "Where"_("Equal"_("S_SUPPKEY"_, "PS_SUPPKEY"_)),
                                                "Times"_("milliSF"_, 10)),
                                            "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_,
                                                  "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                  "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),

                                        "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_)),
                                        "Times"_("milliSF"_, 200)),
                                    "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_, "PS_PARTKEY"_,
                                          "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                          "PS_SUPPLYCOST"_)),
                                "Project"_("LINEITEM"_,
                                           "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                                 "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                 "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                 "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                 "L_QUANTITY"_)),
                                "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                  "List"_("L_PARTKEY"_, "L_SUPPKEY"_))),
                                "Times"_("milliSF"_, 800)),
                            "As"_("N_NAME"_, "N_NAME"_, "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_,
                                  "L_ORDERKEY"_, "L_ORDERKEY"_, "L_EXTENDEDPRICE"_,
                                  "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                  "L_QUANTITY"_)),
                        "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_)),
                        "Times"_("milliSF"_, 1500)),
                    "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                          "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                                   "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
            "By"_("nation"_, /*"Minus"_(*/ "o_year"_ /*)*/)));
    queries.try_emplace(
        TPCH_Q18,
        "Top"_(
            "Group"_(
                "Project"_(

                    "Join"_(
                        "Select"_("Group"_("Project"_("LINEITEM"_,

                                                      "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                            "L_QUANTITY"_, "L_QUANTITY"_)),
                                           "By"_("L_ORDERKEY"_), "Sum"_("L_QUANTITY"_)),
                                  "Where"_("Greater"_("Sum_L_QUANTITY"_, 300))),

                        "Project"_(
                            "Join"_("Project"_("CUSTOMER"_, "As"_("C_NAME"_, "C_NAME"_,
                                                                  "C_CUSTKEY"_, "C_CUSTKEY"_)),
                                    "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                                "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                                "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                                "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                                    "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_)),
                                    "Times"_("milliSF"_, 150)),
                            "As"_("C_NAME"_, "C_NAME"_, "O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_,
                                  "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_TOTALPRICE"_,
                                  "O_TOTALPRICE"_)),
                        "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),

                    "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                          "O_TOTALPRICE"_, "O_TOTALPRICE"_, "C_NAME"_, "C_NAME"_, "O_CUSTKEY"_,
                          "O_CUSTKEY"_, "L_QUANTITY"_, "Sum_L_QUANTITY"_)),
                "By"_("C_NAME"_, "O_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
                "Sum"_("L_QUANTITY"_)),
            "By"_(/*"Minus"_(*/ "O_TOTALPRICE"_ /*)*/, "O_ORDERDATE"_), 100));

    // TPC-H with common plan (i.e. duckdb plan)
    queries.try_emplace(TPCH_COMMON_PLAN_Q3, queries[TPCH_Q3].clone()); // matching with duckdb
    queries.try_emplace(
        TPCH_COMMON_PLAN_Q9,
        "Sort"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_(
                            "Join"_(
                                "Project"_(
                                    "Join"_(
                                        "Project"_(
                                            "Join"_(
                                                "Project"_("NATION"_,
                                                           "As"_("N_NAME"_, "N_NAME"_,
                                                                 "N_NATIONKEY"_,
                                                                 "N_NATIONKEY"_)),
                                                "Project"_("SUPPLIER"_,
                                                           "As"_("S_SUPPKEY"_, "S_SUPPKEY"_,
                                                                 "S_NATIONKEY"_,
                                                                 "S_NATIONKEY"_)),
                                                "Where"_("Equal"_("N_NATIONKEY"_, "S_NATIONKEY"_)),
                                                25),
                                            "As"_("N_NAME"_, "N_NAME"_, "S_SUPPKEY"_,
                                                  "S_SUPPKEY"_)),
                                        "Project"_(
                                            "Join"_(
                                                "Project"_(
                                                    "Select"_(
                                                        "Project"_("PART"_,
                                                                   "As"_("P_PARTKEY"_, "P_PARTKEY"_,
                                                                         "P_NAME"_, "P_NAME"_)),
                                                        "Where"_("StringContainsQ"_("P_NAME"_,
                                                                                    "green"))),
                                                    "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                                "Project"_("PARTSUPP"_,
                                                           "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                 "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                 "PS_SUPPLYCOST"_,
                                                                 "PS_SUPPLYCOST"_)),
                                                "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_)),
                                                "Times"_("milliSF"_, 11)),
                                            "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_,
                                                  "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                                  "PS_SUPPLYCOST"_)),
                                        "Where"_("Equal"_("S_SUPPKEY"_, "PS_SUPPKEY"_)),
                                        "Times"_("milliSF"_, 11)),
                                    "As"_("N_NAME"_, "N_NAME"_, "PS_PARTKEY"_, "PS_PARTKEY"_,
                                          "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                          "PS_SUPPLYCOST"_)),
                                "Project"_("LINEITEM"_,
                                           "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_,
                                                 "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                 "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                 "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                 "L_QUANTITY"_)),
                                "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                  "List"_("L_PARTKEY"_, "L_SUPPKEY"_))),
                                "Times"_("milliSF"_, 45)),
                            "As"_("N_NAME"_, "N_NAME"_, "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_,
                                  "L_ORDERKEY"_, "L_ORDERKEY"_, "L_EXTENDEDPRICE"_,
                                  "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                  "L_QUANTITY"_)),
                        "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                    "O_ORDERDATE"_)),
                        "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_)),
                        "Times"_("milliSF"_, 320)),
                    "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                          "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                                   "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
            "By"_("nation"_, /*"Minus"_(*/ "o_year"_ /*)*/)));
    queries.try_emplace(
        TPCH_COMMON_PLAN_Q18,
        "Top"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_(
                            "Select"_("Group"_("Project"_("LINEITEM"_,
                                                          "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                                "L_QUANTITY"_, "L_QUANTITY"_)),
                                               "By"_("L_ORDERKEY"_), "Sum"_("L_QUANTITY"_)),
                                      "Where"_("Greater"_("Sum_L_QUANTITY"_, 300))),
                            "As"_("INNER_L_ORDERKEY"_, "L_ORDERKEY"_)),
                        "Project"_(
                            "Join"_("Project"_(
                                        "Join"_("Project"_("CUSTOMER"_,
                                                           "As"_("C_NAME"_, "C_NAME"_, "C_CUSTKEY"_,
                                                                 "C_CUSTKEY"_)),
                                                "Project"_("ORDERS"_,
                                                           "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                                 "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                                 "O_ORDERDATE"_, "O_ORDERDATE"_,
                                                                 "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                                                "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_)),
                                                "Times"_("milliSF"_, 150)),
                                        "As"_("C_NAME"_, "C_NAME"_, "O_ORDERKEY"_, "O_ORDERKEY"_,
                                              "O_CUSTKEY"_, "O_CUSTKEY"_, "O_ORDERDATE"_,
                                              "O_ORDERDATE"_, "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                                    "Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                                  "L_QUANTITY"_, "L_QUANTITY"_)),
                                    "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_)),
                                    "Times"_("milliSF"_, 1500)),
                            "As"_("C_NAME"_, "C_NAME"_, "O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_,
                                  "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_TOTALPRICE"_,
                                  "O_TOTALPRICE"_, "L_QUANTITY"_, "L_QUANTITY"_)),
                        "Where"_("Equal"_("INNER_L_ORDERKEY"_, "O_ORDERKEY"_)),
                        "Times"_("milliSF"_, 60)),
                    "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                          "O_TOTALPRICE"_, "O_TOTALPRICE"_, "C_NAME"_, "C_NAME"_, "O_CUSTKEY"_,
                          "O_CUSTKEY"_, "L_QUANTITY"_, "L_QUANTITY"_)),
                "By"_("C_NAME"_, "O_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
                "Sum"_("L_QUANTITY"_)),
            "By"_(/*"Minus"_(*/ "O_TOTALPRICE"_ /*)*/, "O_ORDERDATE"_), 100));

    // TPC-H (with evaluation)
    queries.try_emplace(
        TPCH_Q1_WITH_EVAL,
        "Sort"_(
            "Project"_(
                "Group"_(
                    "Project"_(
                        "Project"_(
                            "Evaluate"_("Select"_(
                                "Project"_("LINEITEM"_,
                                           "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                 "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                 "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                 "L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_LINESTATUS"_,
                                                 "L_LINESTATUS"_, "L_TAX"_, "L_TAX"_)),
                                "Where"_("Greater"_("DateObject"_("1998-08-31"), "L_SHIPDATE"_)))),
                            "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                  "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                  "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                  "L_DISCOUNT"_, "L_DISCOUNT"_, "calc1"_,
                                  "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_))),
                        "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_,
                              "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                              "L_DISCOUNT"_, "L_DISCOUNT"_, "DISC_PRICE"_,
                              "Times"_("L_EXTENDEDPRICE"_, "calc1"_), "calc"_, "calc2"_)),
                    "By"_("RETURNFLAG_AND_LINESTATUS"_),
                    "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM_BASE_PRICE"_,
                          "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_, "Sum"_("DISC_PRICE"_),
                          "SUM_CHARGES"_, "Sum"_("Times"_("DISC_PRICE"_, "calc"_)), "SUM_DISC"_,
                          "Sum"_("L_DISCOUNT"_), "COUNT_ORDER"_, "Count"_("L_QUANTITY"_))),
                "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "SUM_QTY"_,
                      "SUM_QTY"_, "SUM_BASE_PRICE"_, "SUM_BASE_PRICE"_, "SUM_DISC_PRICE"_,
                      "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
                      "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
                      "Divide"_("SUM_BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
                      "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)),
            "By"_("RETURNFLAG_AND_LINESTATUS"_)));
    queries.try_emplace(
        TPCH_Q3_WITH_EVAL,
        "Top"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_(
                            "Select"_(
                                "Project"_("CUSTOMER"_, "As"_("C_CUSTKEY"_, "C_CUSTKEY"_,
                                                              "C_MKTSEGMENT"_, "C_MKTSEGMENT"_)),
                                "Where"_("StringContainsQ"_("C_MKTSEGMENT"_, "BUILDING"))),
                            "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
                        "Project"_(
                            "Join"_(
                                "Select"_("Project"_(
                                              "ORDERS"_,
                                              "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_,
                                                    "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                                                    "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                                          "Where"_("Greater"_("DateObject"_("1995-03-15"),
                                                              "O_ORDERDATE"_))),
                                "Project"_(
                                    "Evaluate"_("Select"_(
                                        "Project"_("LINEITEM"_,
                                                   "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                         "L_DISCOUNT"_, "L_DISCOUNT"_,
                                                         "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                         "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                        "Where"_("Greater"_("L_SHIPDATE"_,
                                                            "DateObject"_("1995-03-15"))))),
                                    "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_,
                                          "L_DISCOUNT"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                                "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_)),
                                "Times"_("milliSF"_, 750)),
                            "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "INV_DISC"_,
                                  "Minus"_(1.0, "L_DISCOUNT"_), "L_EXTENDEDPRICE"_,
                                  "L_EXTENDEDPRICE"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_,
                                  "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
                        "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_)), "Times"_("milliSF"_, 35)),
                    "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "INV_DISC"_), "L_ORDERKEY"_,
                          "L_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_,
                          "O_SHIPPRIORITY"_)),
                "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_), "Sum"_("revenue"_)),
            "By"_(/*"Minus"_(*/ "Sum_revenue"_ /*)*/, "O_ORDERDATE"_), 10));
    queries.try_emplace(
        TPCH_Q6_WITH_EVAL,
        "Group"_("Project"_(
                     "Evaluate"_("Select"_(
                         "Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                       "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                       "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                         "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                         "Greater"_("L_DISCOUNT"_, 0.0499),
                                         "Greater"_(0.07001, "L_DISCOUNT"_),
                                         "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                         "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31")))))),
                     "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
                 "Sum"_("revenue"_)));
    queries.try_emplace(
        TPCH_Q9_WITH_EVAL,
        "Sort"_(
            "Group"_(
                "Project"_(
                    "Join"_(
                        "Project"_("NATION"_,
                                   "As"_("N_NAME"_, "N_NAME"_, "N_NATIONKEY"_, "N_NATIONKEY"_)),
                        "Project"_(
                            "Join"_(
                                "Project"_(
                                    "Join"_(
                                        "Project"_("SUPPLIER"_,
                                                   "As"_("S_SUPPKEY"_, "S_SUPPKEY"_, "S_NATIONKEY"_,
                                                         "S_NATIONKEY"_)),
                                        "Project"_(
                                            "Evaluate"_("Join"_(
                                                "Project"_(
                                                    "Join"_(
                                                        "Project"_(
                                                            "Select"_("Project"_("PART"_,
                                                                                 "As"_("P_PARTKEY"_,
                                                                                       "P_PARTKEY"_,
                                                                                       "P_NAME"_,
                                                                                       "P_NAME"_)),
                                                                      "Where"_("StringContainsQ"_(
                                                                          "P_NAME"_, "green"))),
                                                            "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                                        "Project"_(
                                                            "PARTSUPP"_,
                                                            "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                                  "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                                  "PS_SUPPLYCOST"_,
                                                                  "PS_SUPPLYCOST"_)),
                                                        "Where"_(
                                                            "Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                                    "As"_("PS_PARTKEY"_, "PS_PARTKEY"_,
                                                          "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                                          "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_)),
                                                "Project"_("LINEITEM"_,
                                                           "As"_("L_PARTKEY"_, "L_PARTKEY"_,
                                                                 "L_SUPPKEY"_, "L_SUPPKEY"_,
                                                                 "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                                 "L_EXTENDEDPRICE"_,
                                                                 "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                                                 "L_DISCOUNT"_, "L_QUANTITY"_,
                                                                 "L_QUANTITY"_)),
                                                "Where"_("Equal"_(
                                                    "List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                    "List"_("L_PARTKEY"_, "L_SUPPKEY"_))))),
                                            "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_SUPPKEY"_,
                                                  "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                                  "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                                  "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                                  "L_QUANTITY"_)),
                                        "Where"_("Equal"_("S_SUPPKEY"_, "L_SUPPKEY"_))),
                                    "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_ORDERKEY"_,
                                          "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                          "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                          "L_QUANTITY"_, "S_NATIONKEY"_, "S_NATIONKEY"_)),
                                "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_,
                                                            "O_ORDERDATE"_, "O_ORDERDATE"_)),
                                "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                            "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_EXTENDEDPRICE"_,
                                  "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                                  "L_QUANTITY"_, "S_NATIONKEY"_, "S_NATIONKEY"_, "O_ORDERDATE"_,
                                  "O_ORDERDATE"_)),
                        "Where"_("Equal"_("N_NATIONKEY"_, "S_NATIONKEY"_))),
                    "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                          "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                                   "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
                "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
            "By"_("nation"_, /*"Minus"_(*/ "o_year"_ /*)*/)));
    queries.try_emplace(
        TPCH_Q18_WITH_EVAL,
        "Top"_(
            "Group"_(
                "Join"_(
                    "Project"_(
                        "Join"_(
                            "Evaluate"_("Join"_(
                                "Project"_(
                                    "Select"_(
                                        "Group"_("Project"_("LINEITEM"_,
                                                            "As"_("L_ORDERKEY"_, "L_ORDERKEY"_,
                                                                  "L_QUANTITY"_, "L_QUANTITY"_)),
                                                 "By"_("L_ORDERKEY"_), "Sum"_("L_QUANTITY"_)),
                                        "Where"_("Greater"_("Sum_L_QUANTITY"_, 300))),
                                    "As"_("INNER_L_ORDERKEY"_, "L_ORDERKEY"_)),
                                "Project"_("LINEITEM"_,
                                           "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_QUANTITY"_,
                                                 "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                                                 "L_EXTENDEDPRICE"_)),
                                "Where"_("Equal"_("INNER_L_ORDERKEY"_, "L_ORDERKEY"_)))),
                            "Project"_("ORDERS"_,
                                       "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_,
                                             "O_CUSTKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                                             "O_TOTALPRICE"_, "O_TOTALPRICE"_)),
                            "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                        "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                              "O_ORDERDATE"_, "O_ORDERDATE"_, "O_TOTALPRICE"_, "O_TOTALPRICE"_,
                              "L_QUANTITY"_, "L_QUANTITY"_, "L_EXTENDEDPRICE"_,
                              "L_EXTENDEDPRICE"_)),
                    "Project"_("CUSTOMER"_,
                               "As"_("C_NAME"_, "C_NAME"_, "C_CUSTKEY"_, "C_CUSTKEY"_)),
                    "Where"_("Equal"_("O_CUSTKEY"_, "C_CUSTKEY"_))),
                "By"_("C_NAME"_, "C_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
                "Sum"_("L_EXTENDEDPRICE"_)),
            "By"_(/*"Minus"_(*/ "O_TOTALPRICE"_ /*)*/, "O_ORDERDATE"_), 100));

    // imputation
    queries.try_emplace(
        TPCH_Q1_MODIFIED,
        "Group"_(
            "Project"_(
                "Evaluate"_("Select"_(
                    "Project"_("LINEITEM"_,
                               "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                     "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                    "Where"_("Greater"_(27354240, "L_SHIPDATE"_)))),
                "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_RETURNFLAG"_, "L_RETURNFLAG"_)),
            "By"_("L_RETURNFLAG"_), "Sum"_("L_EXTENDEDPRICE"_)));
    queries.try_emplace(
        TPCH_Q6_MODIFIED,
        "Group"_("Project"_(
                     "Evaluate"_("Select"_(
                         "Project"_("LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_,
                                                       "L_DISCOUNT"_, "L_SHIPDATE"_, "L_SHIPDATE"_,
                                                       "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                         "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                         "Greater"_("L_DISCOUNT"_, 4), "Greater"_(6, "L_DISCOUNT"_),
                                         "Greater"_(15776640, "L_SHIPDATE"_),
                                         "Greater"_("L_SHIPDATE"_, 12623039))))),
                     "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
                 "Sum"_("revenue"_)));
    queries.try_emplace(
        TPCH_Q1_FULL_EVAL,
        "Group"_(
            "Project"_(
                "Select"_(
                    "Evaluate"_("Project"_(
                        "LINEITEM"_, "As"_("L_RETURNFLAG"_, "L_RETURNFLAG"_, "L_SHIPDATE"_,
                                           "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_))),
                    "Where"_("Greater"_(27354240, "L_SHIPDATE"_))),
                "As"_("L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_RETURNFLAG"_, "L_RETURNFLAG"_)),
            "By"_("L_RETURNFLAG"_), "Sum"_("L_EXTENDEDPRICE"_)));

    queries.try_emplace(
        CDC_Q1,
        "Project"_(
            "Group"_(
                "Evaluate"_(
                    "Join"_("Select"_(
                                "Evaluate"_("Project"_("EXAMS"_,
                                                       "As"_("examId"_, "id"_, "height"_, "height"_,
                                                             "cuff_size"_, "cuff_size"_)),
                                            "height"_),
                                "Where"_("Greater"_("height"_, 149))),
                            "Project"_("DEMO"_, "As"_("demoId"_, "id"_, "income"_, "income"_)),
                            "Where"_("Equal"_("examId"_, "demoId"_))),
                    "cuff_size"_, "income"_),
                "By"_("income"_),
                "As"_("SUM_CUFF"_, "Sum"_("cuff_size"_), "COUNT_CUFF"_, "Count"_("cuff_size"_))),
            "As"_("income"_, "income"_, "AvgCuffSize"_, "Divide"_("SUM_CUFF"_, "COUNT_CUFF"_))));
    queries.try_emplace(
        CDC_Q2,
        "Project"_(
            "Group"_(
                "Evaluate"_(
                    "Join"_(
                        "Join"_(
                            "Select"_("Evaluate"_("Project"_("EXAMS"_, "As"_("examId"_, "id"_,
                                                                             "weight"_, "weight"_)),
                                                  "weight"_),
                                      "Where"_("Greater"_("weight"_, 62))),
                            "Project"_("LABS"_, "As"_("labId"_, "id"_, "creatine"_, "creatine"_)),
                            "Where"_("Equal"_("examId"_, "labId"_))),
                        "Select"_(
                            "Evaluate"_(
                                "Project"_("DEMO"_, "As"_("demoId"_, "id"_, "income"_, "income"_)),
                                "income"_),
                            "Where"_("And"_("Greater"_("income"_, 12), "Greater"_(16, "income"_)))),
                        "Where"_("Equal"_("examId"_, "demoId"_))),
                    "creatine"_),
                "By"_("income"_),
                "As"_("SUM_CREATINE"_, "Sum"_("creatine"_), "COUNT_CREATINE"_,
                      "Count"_("creatine"_))),
            "As"_("income"_, "income"_, "AvgCreatine"_,
                  "Divide"_("SUM_CREATINE"_, "COUNT_CREATINE"_))));
    queries.try_emplace(
        CDC_Q3,
        "Project"_(
            "Group"_(
                "Evaluate"_(
                    "Join"_("Join"_("Project"_("EXAMS"_, "As"_("examId"_, "id"_)),
                                    "Project"_("LABS"_, "As"_("labId"_, "id"_, "blood_lead"_,
                                                              "blood_lead"_)),
                                    "Where"_("Equal"_("examId"_, "labId"_))),
                            "Select"_("Project"_("DEMO"_,
                                                 "As"_("demoId"_, "id"_, "age_yrs"_, "age_yrs"_)),
                                      "Where"_("Greater"_(7, "age_yrs"_))),
                            "Where"_("Equal"_("examId"_, "demoId"_))),
                    "blood_lead"_),
                "As"_("SUM_LEAD"_, "Sum"_("blood_lead"_), "COUNT_LEAD"_, "Count"_("blood_lead"_))),
            "As"_("AvgLead"_, "Divide"_("SUM_LEAD"_, "COUNT_LEAD"_))));
    queries.try_emplace(
        CDC_Q4,
        "Project"_(
            "Group"_(
                "Evaluate"_(
                    "Join"_(
                        "Join"_("Select"_("Evaluate"_(
                                              "Project"_("EXAMS"_,
                                                         "As"_("examId"_, "id"_, "body_mass_index"_,
                                                               "body_mass_index"_,
                                                               "blood_pressure_systolic"_,
                                                               "blood_pressure_systolic"_)),
                                              "body_mass_index"_),
                                          "Where"_("Greater"_("body_mass_index"_, 29))),
                                "Project"_("LABS"_, "As"_("labId"_, "id"_)),
                                "Where"_("Equal"_("examId"_, "labId"_))),
                        "Project"_("DEMO"_, "As"_("demoId"_, "id"_, "gender"_, "gender"_)),
                        "Where"_("Equal"_("examId"_, "demoId"_))),
                    "blood_pressure_systolic"_),
                "By"_("gender"_),
                "As"_("SUM_PRESSURE"_, "Sum"_("blood_pressure_systolic"_), "COUNT_PRESSURE"_,
                      "Count"_("blood_pressure_systolic"_))),
            "As"_("gender"_, "gender"_, "AvgPressure"_,
                  "Divide"_("SUM_PRESSURE"_, "COUNT_PRESSURE"_))));
    queries.try_emplace(
        CDC_Q5,
        "Project"_(
            "Group"_("Evaluate"_(
                         "Join"_("Select"_(
                                     "Evaluate"_("Project"_("EXAMS"_,
                                                            "As"_("examId"_, "id"_, "waist"_,
                                                                  "waist_circumference"_, "weight"_,
                                                                  "weight"_, "height"_, "height"_)),
                                                 "weight"_, "height"_),
                                     "Where"_("And"_("Greater"_("weight"_, 99),
                                                     "Greater"_("height"_, 149)))),
                                 "Project"_("DEMO"_, "As"_("demoId"_, "id"_)),
                                 "Where"_("Equal"_("examId"_, "demoId"_))),
                         "waist"_),
                     "As"_("SUM_WAIST"_, "Sum"_("waist"_), "COUNT_WAIST"_, "Count"_("waist"_))),
            "As"_("AvgWaist"_, "Divide"_("SUM_WAIST"_, "COUNT_WAIST"_))));

    queries.try_emplace(
        FCC_Q6,
        "Project"_(
            "Group"_(
                "Evaluate"_("Select"_("Evaluate"_("Project"_("FCC"_, "As"_("attendedbootcamp"_,
                                                                           "attendedbootcamp"_,
                                                                           "income"_, "income"_)),
                                                  "income"_),
                                      "Where"_("Greater"_("income"_, 49999))),
                            "attendedbootcamp"_),
                "By"_("attendedbootcamp"_),
                "As"_("SUM_INCOME"_, "Sum"_("income"_), "COUNT_INCOME"_, "Count"_("income"_))),
            "As"_("attendedbootcamp"_, "attendedbootcamp"_, "AvgIncome"_,
                  "Divide"_("SUM_INCOME"_, "COUNT_INCOME"_))));
    queries.try_emplace(
        FCC_Q7,
        "Project"_("Group"_("Evaluate"_(
                                "Select"_("Evaluate"_(
                                              "Project"_("FCC"_,
                                                         "As"_("commutetime"_, "commutetime"_,
                                                               "countrycitizen"_, "countrycitizen"_,
                                                               "gender"_, "gender"_)),
                                              "countrycitizen"_, "gender"_),
                                          "Where"_("And"_("Equal"_("countrycitizen"_, 251),
                                                          "Equal"_("gender"_, 290)))),
                                "commutetime"_),
                            "As"_("SUM_TIME"_, "Sum"_("commutetime"_), "COUNT_TIME"_,
                                  "Count"_("commutetime"_))),
                   "As"_("AvgCommuteTime"_, "Divide"_("SUM_TIME"_, "COUNT_TIME"_))));
    queries.try_emplace(
        FCC_Q8,
        "Project"_("Group"_("Select"_("Evaluate"_("Project"_(
                                          "FCC"_, "As"_("studentdebtowe"_, "studentdebtowe"_,
                                                        "schooldegree"_, "schooldegree"_))),
                                      "Where"_("And"_("Greater"_("studentdebtowe"_, 0),
                                                      "Greater"_("schooldegree"_, -1)))),
                            "By"_("schooldegree"_),
                            "As"_("SUM_DEBT"_, "Sum"_("studentdebtowe"_), "COUNT_DEBT"_,
                                  "Count"_("studentdebtowe"_))),
                   "As"_("schooldegree"_, "schooldegree"_, "AvgDebt"_,
                         "Divide"_("SUM_DEBT"_, "COUNT_DEBT"_))));
    queries.try_emplace(
        FCC_Q9,
        "Project"_(
            "Group"_(
                "Evaluate"_(
                    "Join"_(
                        "Evaluate"_("Project"_("GDP"_, "As"_("gdp_per_capita"_, "gdp_per_capita"_,
                                                             "country"_, "country"_)),
                                    "country"_),
                        "Evaluate"_(
                            "Select"_(
                                "Evaluate"_(
                                    "Project"_("FCC"_, "As"_("attendedbootcamp"_,
                                                             "attendedbootcamp"_, "age"_, "age"_,
                                                             "countrycitizen"_, "countrycitizen"_)),
                                    "age"_),
                                "Where"_("Greater"_("age"_, 17))),
                            "countrycitizen"_),
                        "Where"_("Equal"_("country"_, "countrycitizen"_))),
                    "gdp_per_capita"_, "attendedbootcamp"_),
                "By"_("attendedbootcamp"_),
                "As"_("SUM_GDP"_, "Sum"_("gdp_per_capita"_), "COUNT_GDP"_,
                      "Count"_("gdp_per_capita"_))),
            "As"_("attendedbootcamp"_, "attendedbootcamp"_, "AvgGdp"_,
                  "Divide"_("SUM_GDP"_, "COUNT_GDP"_))));

    queries.try_emplace(
        ACS, "Project"_("Group"_("Evaluate"_("Project"_("ACS"_, "As"_("c0"_, "c0"_)), "c0"_),
                                 "As"_("SUM_C0"_, "Sum"_("c0"_), "COUNT_C0"_, "Count"_("c0"_))),
                        "As"_("AvgC0"_, "Divide"_("SUM_C0"_, "COUNT_C0"_))));
  }
  return queries;
}

static auto& monetdbQueries() {
  static std::map<int, std::string> queries = {
      {EVALUATE_INT_ADDITIONS, "select l_orderkey + l_partkey from lineitem; "s},
      {EVALUATE_NESTED_INT_ADDITIONS,
       "select (SUM + SUM) as SUM from "s
       "   (select (SUM + SUM) as SUM from "s
       "       (select (SUM + SUM) as SUM from "s
       "           (select (SUM + SUM) as SUM from "s
       "               (select (SUM + SUM) as SUM from "s
       "                   (select (SUM + SUM) as SUM from "s
       "                       (select (SUM + SUM) as SUM from "s
       "                           (select (SUM + SUM) as SUM from "s
       "                               (select (SUM + SUM) as SUM from "s
       "                                   (select (l_orderkey + l_partkey) as SUM from lineitem"s
       "                                   ) as inner_table"s
       "                               ) as inner_table"s
       "                           ) as inner_table"s
       "                       ) as inner_table"s
       "                   ) as inner_table"s
       "               ) as inner_table"s
       "           ) as inner_table"s
       "       ) as inner_table"s
       "   ) as inner_table; "s},
      {EVALUATE_INT_MULTIPLICATIONS, "select l_orderkey * l_partkey from lineitem; "s},
      {EVALUATE_INT_ADDMUL, "select l_orderkey + l_partkey * l_suppkey from lineitem; "s},
      {EVALUATE_INT_ADD_CONSTANT, "select l_orderkey + 1 from lineitem; "s},
      {EVALUATE_INT_MULT_CONSTANT, "select 10 * l_orderkey from lineitem; "s},
      {EVALUATE_INT_ADDMULT_CONSTANTS, "select 10 * l_orderkey + 1000 from lineitem; "s},
      {EVALUATE_FLOAT_ADDITIONS, "select l_quantity + l_extendedprice from lineitem; "s},
      {EVALUATE_FLOAT_MULTIPLICATIONS, "select l_quantity * l_extendedprice from lineitem; "s},
      {EVALUATE_FLOAT_ADDMUL, "select l_quantity + l_extendedprice * l_discount from lineitem; "s},
      {EVALUATE_FLOAT_ADD_CONSTANT, "select l_quantity + 1 from lineitem; "s},
      {EVALUATE_FLOAT_MULT_CONSTANT, "select 10 * l_quantity from lineitem; "s},
      {EVALUATE_FLOAT_ADDMULT_CONSTANTS, "select 10 * l_quantity + 1000 from lineitem; "s},
      {EVALUATE_STRING_JOIN, "select concat(l_returnflag, l_linestatus) from lineitem; "s},
      {AGGREGATE_COUNT, "select count(*) from lineitem; "s},
      {AGGREGATE_SUM_INTS, "select sum(l_orderkey) from lineitem; "s},
      {AGGREGATE_SUM_FLOATS, "select sum(l_quantity) from lineitem; "s},
      {AGGREGATE_AVERAGE_INTS, "select avg(l_orderkey) from lineitem; "s},
      {AGGREGATE_AVERAGE_FLOATS, "select avg(l_quantity) from lineitem; "s},
      {PROJECT_5_DIFFERENT_COLUMNS,
       "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity from lineitem; "s},
      {PROJECT_10_DIFFERENT_COLUMNS,
       "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, "s
       "l_discount, l_tax, l_returnflag, l_linestatus from lineitem; "s},
      {PROJECT_15_DIFFERENT_COLUMNS,
       "select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, "s
       "l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, "s
       "l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode from lineitem; "s},
      {PROJECT_30_INT_COLUMNS,
       "select l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber from lineitem; "s},
      {PROJECT_30_LARGE_STRING_COLUMNS,
       "select l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment from lineitem; "s},

      {SELECT_LOW_INT, "select ps_availqty from partsupp where ps_availqty < 600; "s},
      {SELECT_LOW_FLOAT, "select l_quantity from lineitem where l_extendedprice < 4300; "s},
      {SELECT_LOW_DATE, "select l_shipdate from lineitem where l_shipdate < '1992-7-01'; "s},
      {SELECT_LOW_STRING, "select o_orderstatus from orders where o_orderstatus like '%P%'; "s},
      {SELECT_LOW_LARGE_STRING, "select l_quantity from lineitem where l_comment like '%fo%'; "s},

      {SELECT_HALF_INT, "select ps_availqty from partsupp where ps_availqty < 5100; "s},
      {SELECT_HALF_FLOAT, "select l_quantity from lineitem where l_extendedprice < 37000; "s},
      {SELECT_HALF_DATE, "select l_shipdate from lineitem where l_shipdate < '1995-7-01'; "s},
      {SELECT_HALF_STRING, "select l_quantity from lineitem where l_returnflag like '%N%'; "s},
      {SELECT_HALF_LARGE_STRING, "select l_quantity from lineitem where l_comment like '%p%'; "s},
      {SELECT_HALF_MULTIPLE_STRINGS,
       "select l_quantity from lineitem where l_returnflag like '%N%' and l_linestatus like '%O%'; "s},

      {SELECT_HIGH_INT, "select ps_availqty from partsupp where ps_availqty < 9600; "s},
      {SELECT_HIGH_FLOAT, "select l_quantity from lineitem where l_extendedprice < 80000; "s},
      {SELECT_HIGH_DATE, "select l_shipdate from lineitem where l_shipdate < '1998-6-15'; "s},
      {SELECT_HIGH_STRING,
       "select o_orderstatus from orders where o_orderstatus not like '%P%'; "s},
      {SELECT_HIGH_LARGE_STRING, "select l_quantity from lineitem where l_comment like '%e%'; "s},

      {JOIN_INTS, "select l_orderkey from lineitem, orders where l_orderkey = o_orderkey; "s},
      {JOIN_MANY_INTS, "select l_suppkey from lineitem, partsupp where l_suppkey = ps_suppkey; "s},

      {GROUP_BY_MANY_INTS, "select l_orderkey from lineitem group by l_orderkey; "s},
      {GROUP_BY_FEW_INTS, "select l_suppkey from lineitem group by l_suppkey; "s},
      {GROUP_BY_STRING, "select l_returnflag from lineitem group by l_returnflag; "s},
      {GROUP_BY_MULTIPLE_INTS,
       "select l_orderkey, l_suppkey from lineitem group by l_orderkey, l_suppkey; "s},
      {GROUP_BY_MULTIPLE_STRINGS,
       "select l_returnflag, l_linestatus from lineitem group by l_returnflag, l_linestatus; "s},

      {TOP10_BY_INT, "select l_orderkey from lineitem order by l_orderkey limit 10; "s},
      {TOP10_BY_FLOAT, "select l_quantity from lineitem order by l_quantity limit 10; "s},
      {TOP10_BY_STRING, "select l_returnflag from lineitem order by l_returnflag limit 10; "s},
      {TOP10_BY_MULTIPLE_INTS,
       "select l_orderkey, l_suppkey from lineitem order by l_orderkey, l_suppkey limit 10; "s},
      {TOP10_BY_MULTIPLE_STRINGS,
       "select l_returnflag, l_linestatus from lineitem order by l_returnflag, l_linestatus limit 10; "s},

      {TOP100_BY_INT, "select l_orderkey from lineitem order by l_orderkey limit 100; "s},
      {TOP100_BY_FLOAT, "select l_quantity from lineitem order by l_quantity limit 100; "s},
      {TOP100_BY_STRING, "select l_returnflag from lineitem order by l_returnflag limit 100; "s},
      {TOP100_BY_MULTIPLE_INTS,
       "select l_orderkey, l_suppkey from lineitem order by l_orderkey, l_suppkey limit 100; "s},
      {TOP100_BY_MULTIPLE_STRINGS,
       "select l_returnflag, l_linestatus from lineitem order by l_returnflag, l_linestatus limit 100; "s},

      {TOP500_BY_INT, "select l_orderkey from lineitem order by l_orderkey limit 500; "s},
      {TOP500_BY_FLOAT, "select l_quantity from lineitem order by l_quantity limit 500; "s},
      {TOP500_BY_STRING, "select l_returnflag from lineitem order by l_returnflag limit 500; "s},
      {TOP500_BY_MULTIPLE_INTS,
       "select l_orderkey, l_suppkey from lineitem order by l_orderkey, l_suppkey limit 500; "s},
      {TOP500_BY_MULTIPLE_STRINGS,
       "select l_returnflag, l_linestatus from lineitem order by l_returnflag, l_linestatus limit 500; "s},

      {Q1_SELECT, "select "s
                  "    l_shipdate "s
                  "from "s
                  "    lineitem "s
                  "where "s
                  "    l_shipdate <= date '1998-12-01' - interval '90' day (3); "s},
      {Q1_CALC, "select "s
                "    (l_extendedprice * (1 - l_discount)) as disc_price, "s
                "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) as charge "s
                "from "s
                "    lineitem; "s},
      {Q1_PROJECT, "select "s
                   "    l_returnflag, "s
                   "    l_linestatus, "s
                   "    l_quantity, "s
                   "    l_extendedprice, "s
                   "    (l_extendedprice * (1 - l_discount)) as disc_price, "s
                   "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) as charge, "s
                   "    l_discount "s
                   "from "s
                   "    lineitem; "s},
      {Q1_SELECT_PROJECT, "select "s
                          "    l_returnflag, "s
                          "    l_linestatus, "s
                          "    l_quantity, "s
                          "    l_extendedprice, "s
                          "    (l_extendedprice * (1 - l_discount)) as disc_price, "s
                          "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) as charge, "s
                          "    l_discount "s
                          "from "s
                          "    lineitem;"s
                          "where "s
                          "    l_shipdate <= date '1998-12-01' - interval '90' day (3); "s},
      {Q1_SUMS, "select "s
                "    sum(l_quantity) as sum_qty, "s
                "    sum(l_extendedprice) as sum_base_price, "s
                "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "s
                "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "s
                "    sum(l_discount) as sum_disc "s
                "from "s
                "    lineitem; "s},
      {Q1_GROUP_BY, "select "s
                    "    l_returnflag, "s
                    "    l_linestatus "s
                    "from "s
                    "    lineitem "s
                    "group by "s
                    "    l_returnflag, "s
                    "    l_linestatus; "s},
      {Q1_SORT_BY, "select "s
                   "    l_returnflag, "s
                   "    l_linestatus "s
                   "from "s
                   "    lineitem "s
                   "group by "s
                   "    l_returnflag, "s
                   "    l_linestatus "s
                   "order by "s
                   "    l_returnflag, "s
                   "    l_linestatus; "s},
      {Q1_GROUP_SUM, "select "s
                     "    sum(l_quantity) as sum_qty, "s
                     "    sum(l_extendedprice) as sum_base_price, "s
                     "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "s
                     "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "s
                     "    sum(l_discount) as sum_disc, "s
                     "    count(*) as count_order "s
                     "from "s
                     "    lineitem "s
                     "group by "s
                     "    l_returnflag, "s
                     "    l_linestatus; "s},
      {Q1_GROUP_AVG, "select "s
                     "    sum(l_quantity) as sum_qty, "s
                     "    sum(l_extendedprice) as sum_base_price, "s
                     "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "s
                     "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "s
                     "    avg(l_quantity) as avg_qty, "s
                     "    avg(l_extendedprice) as avg_price, "s
                     "    avg(l_discount) as avg_disc, "s
                     "    count(*) as count_order "s
                     "from "s
                     "    lineitem "s
                     "group by "s
                     "    l_returnflag, "s
                     "    l_linestatus; "s},

      {TPCH_Q1, "select "s
                "    l_returnflag, "s
                "    l_linestatus, "s
                "    sum(l_quantity) as sum_qty, "s
                "    sum(l_extendedprice) as sum_base_price, "s
                "    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "s
                "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "s
                "    avg(l_quantity) as avg_qty, "s
                "    avg(l_extendedprice) as avg_price, "s
                "    avg(l_discount) as avg_disc, "s
                "    count(*) as count_order "s
                "from "s
                "    lineitem "s
                "where "s
                "    l_shipdate <= date '1998-12-01' - interval '90' day (3) "s
                "group by "s
                "    l_returnflag, "s
                "    l_linestatus "s
                "order by "s
                "    l_returnflag, "s
                "    l_linestatus; "s},
      {TPCH_Q3, "select "s
                "		l_orderkey, "s
                "		sum(l_extendedprice * (1 - l_discount)) as revenue, "s
                "		o_orderdate, "s
                "		o_shippriority "s
                "from customer, "s
                "		orders, "s
                "		lineitem "s
                "where c_mktsegment = 'BUILDING' "s
                "		and c_custkey = o_custkey "s
                "		and l_orderkey = o_orderkey "s
                "		and o_orderdate < date '1995-03-15' "s
                "		and l_shipdate > date '1995-03-15' "s
                "group by "s
                "		l_orderkey, "s
                "		o_orderdate, "s
                "		o_shippriority "s
                "order by "s
                "		revenue desc, "s
                "		o_orderdate "s
                "limit 10; "s},
      {TPCH_COMMON_PLAN_Q3, "select "s
                            "       l_orderkey, "s
                            "       o_orderdate, "s
                            "       o_shippriority, "s
                            "       sum(l_extendedprice * (1 - l_discount)) as revenue "s
                            "from "s
                            "       lineitem, "s
                            "       customer "s
                            "inner join orders on "s
                            "       c_mktsegment = 'BUILDING' "s
                            "       and o_custkey = c_custkey "s
                            "       and o_orderdate < date '1995-03-15' "s
                            "where "s
                            "       l_orderkey = o_orderkey "s
                            "       and l_shipdate > date '1995-03-15' "s
                            "group by "s
                            "       l_orderkey, "s
                            "       o_orderdate, "s
                            "       o_shippriority "s
                            "order by "s
                            "       revenue desc, "s
                            "       o_orderdate "s
                            "limit 10; "s},
      {TPCH_Q6, "select "s
                "    sum(l_extendedprice * l_discount) as revenue "s
                "from "s
                "    lineitem "s
                "where "s
                "    l_shipdate >= date '1994-01-01' "s
                "    and l_shipdate < date '1995-01-01'"s
                "    and l_discount between 0.05 and 0.07"s
                "    and l_quantity < 24; "s},
      {TPCH_Q9, "select "s
                "			nation, "s
                "			o_year, "s
                "			sum(amount) as sum_profit "s
                "from(select n_name as nation, "s
                "			extract(year from o_orderdate) as o_year, "s
                "			l_extendedprice* (1 - l_discount) - ps_supplycost * l_quantity as amount "s
                "		from part, "s
                "			supplier, "s
                "			lineitem, "s
                "			partsupp, "s
                "			orders, "s
                "			nation "s
                "		where s_suppkey = l_suppkey "s
                "			and ps_suppkey = l_suppkey "s
                "			and ps_partkey = l_partkey "s
                "			and p_partkey = l_partkey "s
                "			and o_orderkey = l_orderkey "s
                "			and s_nationkey = n_nationkey "s
                "			and p_name like '%green%' "s
                "	) as profit "s
                "group by nation, o_year "s
                "order by nation, o_year desc; "s},
      {TPCH_COMMON_PLAN_Q9,
       "select "s
       "      nation, "s
       "      o_year, "s
       "      sum(amount) as sum_profit "s
       "  from "s
       "      ( "s
       "          select "s
       "              n_name as nation, "s
       "              extract(year from o_orderdate) as o_year, "s
       "              l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount "s
       "          from "s
       "              supplier, "s
       "              nation, "s
       "              partsupp, "s
       "              part, "s
       "              lineitem, "s
       "              orders "s
       "          where "s
       "              s_nationkey = n_nationkey "s
       "              and ps_partkey = p_partkey "s
       "              and p_name like '%green%' "s
       "              and ps_suppkey = s_suppkey "s
       "              and ps_suppkey = l_suppkey "s
       "              and ps_partkey = l_partkey "s
       "              and o_orderkey = l_orderkey "s
       "      ) as profit "s
       "group by "s
       "      nation, "s
       "      o_year "s
       "order by "s
       "      nation, "s
       "      o_year desc; "s},
      {TPCH_Q18, "select "s
                 "	c_name, c_custkey, o_orderkey, "s
                 "	o_orderdate, o_totalprice, sum(l_quantity) "s
                 "from "s
                 "	customer, orders, lineitem "s
                 "where o_orderkey in ( "s
                 "		select l_orderkey "s
                 "		from lineitem "s
                 "		group by l_orderkey having "s
                 "				sum(l_quantity) > 300 "s
                 "	) "s
                 "	and c_custkey = o_custkey "s
                 "	and o_orderkey = l_orderkey "s
                 "group by "s
                 "	c_name, c_custkey, "s
                 "	o_orderkey, o_orderdate, o_totalprice "s
                 "order by "s
                 "	o_totalprice desc, o_orderdate "s
                 "limit 100; "s},
      {TPCH_COMMON_PLAN_Q18, "select "s
                             "     c_name, "s
                             "     c_custkey, "s
                             "     o_orderkey, "s
                             "     o_orderdate, "s
                             "     o_totalprice, "s
                             "     sum(l_quantity) "s
                             "from "s
                             "     customer, "s
                             "     orders, "s
                             "     lineitem "s
                             "where "s
                             "     o_orderkey in ( "s
                             "         select "s
                             "             l_orderkey "s
                             "         FROM "s
                             "             lineitem "s
                             "         group by "s
                             "             l_orderkey having "s
                             "                 sum(l_quantity) > 300 "s
                             "     ) "s
                             "     and o_custkey = c_custkey "s
                             "     and o_orderkey = l_orderkey "s
                             "group by "s
                             "     c_name, "s
                             "     c_custkey, "s
                             "     o_orderkey, "s
                             "     o_orderdate, "s
                             "     o_totalprice "s
                             "order by "s
                             "     o_totalprice desc, "s
                             "     o_orderdate "s
                             "limit 100; "s},
  };
  return queries;
}

static auto& duckdbQueries() {
  static std::map<int, std::string> queries = {
      {EVALUATE_INT_ADDITIONS, "SELECT l_orderkey + l_partkey FROM lineitem;"},
      {EVALUATE_NESTED_INT_ADDITIONS,
       "SELECT (SUM + SUM) AS SUM from "
       "   (SELECT (SUM + SUM) AS SUM FROM "
       "       (SELECT (SUM + SUM) AS SUM FROM "
       "           (SELECT (SUM + SUM) AS SUM FROM "
       "               (SELECT (SUM + SUM) AS SUM FROM "
       "                   (SELECT (SUM + SUM) AS SUM FROM "
       "                       (SELECT (SUM + SUM) AS SUM FROM "
       "                           (SELECT (SUM + SUM) AS SUM FROM "
       "                               (SELECT (SUM + SUM) AS SUM FROM "
       "                                   (SELECT (l_orderkey + l_partkey) AS SUM FROM lineitem"
       "                                   ) AS inner_table"
       "                               ) AS inner_table"
       "                           ) AS inner_table"
       "                       ) AS inner_table"
       "                   ) AS inner_table"
       "               ) AS inner_table"
       "           ) AS inner_table"
       "       ) AS inner_table"
       "   ) AS inner_table;"},
      {EVALUATE_INT_MULTIPLICATIONS, "SELECT l_orderkey * l_partkey FROM lineitem;"},
      {EVALUATE_INT_ADDMUL, "SELECT l_orderkey + l_partkey * l_suppkey FROM lineitem;"},
      {EVALUATE_INT_ADD_CONSTANT, "SELECT l_orderkey + 1 FROM lineitem;"},
      {EVALUATE_INT_MULT_CONSTANT, "SELECT 10 * l_orderkey FROM lineitem;"},
      {EVALUATE_INT_ADDMULT_CONSTANTS, "SELECT 10 * l_orderkey + 1000 FROM lineitem;"},
      {EVALUATE_FLOAT_ADDITIONS, "SELECT l_quantity + l_extendedprice FROM lineitem;"},
      {EVALUATE_FLOAT_MULTIPLICATIONS, "SELECT l_quantity * l_extendedprice FROM lineitem;"},
      {EVALUATE_FLOAT_ADDMUL, "SELECT l_quantity + l_extendedprice * l_discount FROM lineitem;"},
      {EVALUATE_FLOAT_ADD_CONSTANT, "SELECT l_quantity + 1 FROM lineitem;"},
      {EVALUATE_FLOAT_MULT_CONSTANT, "SELECT 10 * l_quantity FROM lineitem;"},
      {EVALUATE_FLOAT_ADDMULT_CONSTANTS, "SELECT 10 * l_quantity + 1000 FROM lineitem;"},
      {EVALUATE_STRING_JOIN, "SELECT concat(l_returnflag, l_linestatus) FROM lineitem;"},
      {AGGREGATE_COUNT, "SELECT count(*) FROM lineitem;"},
      {AGGREGATE_SUM_INTS, "SELECT sum(l_orderkey) FROM lineitem;"},
      {AGGREGATE_SUM_FLOATS, "SELECT sum(l_quantity) FROM lineitem;"},
      {AGGREGATE_AVERAGE_INTS, "SELECT avg(l_orderkey) FROM lineitem;"},
      {AGGREGATE_AVERAGE_FLOATS, "SELECT avg(l_quantity) FROM lineitem;"},
      {PROJECT_5_DIFFERENT_COLUMNS,
       "SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity FROM lineitem;"},
      {PROJECT_10_DIFFERENT_COLUMNS,
       "SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, "s
       "l_discount, l_tax, l_returnflag, l_linestatus FROM lineitem;"},
      {PROJECT_15_DIFFERENT_COLUMNS,
       "SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, "s
       "l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, "s
       "l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode FROM lineitem;"},
      {PROJECT_30_INT_COLUMNS,
       "SELECT l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber, "s
       "l_linenumber, l_linenumber, l_linenumber, l_linenumber, l_linenumber FROM lineitem;"},
      {PROJECT_30_LARGE_STRING_COLUMNS,
       "SELECT l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment, "s
       "l_comment, l_comment, l_comment, l_comment, l_comment FROM lineitem;"},

      {SELECT_LOW_INT, "SELECT ps_availqty FROM partsupp WHERE ps_availqty < 600;"},
      {SELECT_LOW_FLOAT, "SELECT l_quantity FROM lineitem WHERE l_extendedprice < 4300;"},
      {SELECT_LOW_DATE,
       "SELECT l_shipdate FROM lineitem WHERE l_shipdate < CAST('1992-7-01' AS DATE);"},
      {SELECT_LOW_STRING, "SELECT o_orderstatus FROM orders WHERE o_orderstatus LIKE '%P%';"},
      {SELECT_LOW_LARGE_STRING, "SELECT l_quantity FROM lineitem WHERE l_comment LIKE '%fo%';"},

      {SELECT_HALF_INT, "SELECT ps_availqty FROM partsupp WHERE ps_availqty < 5100;"},
      {SELECT_HALF_FLOAT, "SELECT l_quantity FROM lineitem WHERE l_extendedprice < 37000;"},
      {SELECT_HALF_DATE,
       "SELECT l_shipdate FROM lineitem WHERE l_shipdate < CAST('1995-7-01' AS DATE);"},
      {SELECT_HALF_STRING, "SELECT l_quantity FROM lineitem WHERE l_returnflag LIKE '%N%';"},
      {SELECT_HALF_LARGE_STRING, "SELECT l_quantity FROM lineitem WHERE l_comment LIKE '%p%';"},
      {SELECT_HALF_MULTIPLE_STRINGS, "SELECT l_quantity FROM lineitem WHERE l_returnflag LIKE "
                                     "'%N%' AND l_linestatus LIKE '%O%';"},

      {SELECT_HIGH_INT, "SELECT ps_availqty FROM partsupp WHERE ps_availqty < 9600;"},
      {SELECT_HIGH_FLOAT, "SELECT l_quantity FROM lineitem WHERE l_extendedprice < 80000;"},
      {SELECT_HIGH_DATE,
       "SELECT l_shipdate FROM lineitem WHERE l_shipdate < CAST('1998-6-15' AS DATE);"},
      {SELECT_HIGH_STRING, "SELECT o_orderstatus FROM orders WHERE o_orderstatus not LIKE '%P%';"},
      {SELECT_HIGH_LARGE_STRING, "SELECT l_quantity FROM lineitem WHERE l_comment LIKE '%e%';"},

      {JOIN_INTS, "SELECT l_orderkey FROM lineitem, orders WHERE l_orderkey = o_orderkey;"},
      {JOIN_MANY_INTS, "SELECT l_suppkey FROM lineitem, partsupp WHERE l_suppkey = ps_suppkey;"},

      {GROUP_BY_MANY_INTS, "SELECT l_orderkey FROM lineitem GROUP BY l_orderkey;"},
      {GROUP_BY_FEW_INTS, "SELECT l_suppkey FROM lineitem GROUP BY l_suppkey;"},
      {GROUP_BY_STRING, "SELECT l_returnflag FROM lineitem GROUP BY l_returnflag;"},
      {GROUP_BY_MULTIPLE_INTS,
       "SELECT l_orderkey, l_suppkey FROM lineitem GROUP BY l_orderkey, l_suppkey;"},
      {GROUP_BY_MULTIPLE_STRINGS,
       "SELECT l_returnflag, l_linestatus FROM lineitem GROUP BY l_returnflag, l_linestatus;"},

      {TOP10_BY_INT, "SELECT l_orderkey FROM lineitem ORDER BY l_orderkey LIMIT 10;"},
      {TOP10_BY_FLOAT, "SELECT l_quantity FROM lineitem ORDER BY l_quantity LIMIT 10;"},
      {TOP10_BY_STRING, "SELECT l_returnflag FROM lineitem ORDER BY l_returnflag LIMIT 10;"},
      {TOP10_BY_MULTIPLE_INTS,
       "SELECT l_orderkey, l_suppkey FROM lineitem ORDER BY l_orderkey, l_suppkey LIMIT 10;"},
      {TOP10_BY_MULTIPLE_STRINGS, "SELECT l_returnflag, l_linestatus FROM lineitem ORDER BY "
                                  "l_returnflag, l_linestatus LIMIT 10;"},

      {TOP100_BY_INT, "SELECT l_orderkey FROM lineitem ORDER BY l_orderkey LIMIT 100;"},
      {TOP100_BY_FLOAT, "SELECT l_quantity FROM lineitem ORDER BY l_quantity LIMIT 100;"},
      {TOP100_BY_STRING, "SELECT l_returnflag FROM lineitem ORDER BY l_returnflag LIMIT 100;"},
      {TOP100_BY_MULTIPLE_INTS,
       "SELECT l_orderkey, l_suppkey FROM lineitem ORDER BY l_orderkey, l_suppkey LIMIT 100;"},
      {TOP100_BY_MULTIPLE_STRINGS, "SELECT l_returnflag, l_linestatus FROM lineitem ORDER BY "
                                   "l_returnflag, l_linestatus LIMIT 100;"},

      {TOP500_BY_INT, "SELECT l_orderkey FROM lineitem ORDER BY l_orderkey LIMIT 500;"},
      {TOP500_BY_FLOAT, "SELECT l_quantity FROM lineitem ORDER BY l_quantity LIMIT 500;"},
      {TOP500_BY_STRING, "SELECT l_returnflag FROM lineitem ORDER BY l_returnflag LIMIT 500;"},
      {TOP500_BY_MULTIPLE_INTS,
       "SELECT l_orderkey, l_suppkey FROM lineitem ORDER BY l_orderkey, l_suppkey LIMIT 500;"},
      {TOP500_BY_MULTIPLE_STRINGS, "SELECT l_returnflag, l_linestatus FROM lineitem ORDER BY "
                                   "l_returnflag, l_linestatus LIMIT 500;"},

      {Q1_SELECT, "SELECT "
                  "    l_shipdate "
                  "FROM "
                  "    lineitem "
                  "WHERE "
                  "    l_shipdate <= DATE '1998-09-02';"},
      {Q1_CALC, "SELECT "
                "    (l_extendedprice * (1 - l_discount)) AS disc_price, "
                "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS charge "
                "FROM "
                "    lineitem;"},
      {Q1_PROJECT, "SELECT "
                   "    l_returnflag, "
                   "    l_linestatus, "
                   "    l_quantity, "
                   "    l_extendedprice, "
                   "    (l_extendedprice * (1 - l_discount)) AS disc_price, "
                   "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS charge, "
                   "    l_discount "
                   "FROM "
                   "    lineitem;"},
      {Q1_SELECT_PROJECT, "SELECT "
                          "    l_returnflag, "
                          "    l_linestatus, "
                          "    l_quantity, "
                          "    l_extendedprice, "
                          "    (l_extendedprice * (1 - l_discount)) AS disc_price, "
                          "    (l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS charge, "
                          "    l_discount "
                          "FROM "
                          "    lineitem "
                          "WHERE "
                          "    l_shipdate <= DATE '1998-09-02';"},
      {Q1_SUMS, "SELECT "
                "    sum(l_quantity) AS sum_qty, "
                "    sum(l_extendedprice) AS sum_base_price, "
                "    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
                "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
                "    sum(l_discount) AS sum_disc "
                "FROM "
                "    lineitem;"},
      {Q1_GROUP_BY, "SELECT "
                    "    l_returnflag, "
                    "    l_linestatus "
                    "FROM "
                    "    lineitem "
                    "GROUP BY "
                    "    l_returnflag, "
                    "    l_linestatus;"},
      {Q1_SORT_BY, "SELECT "
                   "    l_returnflag, "
                   "    l_linestatus "
                   "FROM "
                   "    lineitem "
                   "GROUP BY "
                   "    l_returnflag, "
                   "    l_linestatus "
                   "ORDER BY "
                   "    l_returnflag, "
                   "    l_linestatus;"},
      {Q1_GROUP_SUM, "SELECT "
                     "    sum(l_quantity) AS sum_qty, "
                     "    sum(l_extendedprice) AS sum_base_price, "
                     "    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
                     "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
                     "    sum(l_discount) AS sum_disc, "
                     "    count(*) AS count_order "
                     "FROM "
                     "    lineitem "
                     "GROUP BY "
                     "    l_returnflag, "
                     "    l_linestatus;"},
      {Q1_GROUP_AVG, "SELECT "
                     "    sum(l_quantity) AS sum_qty, "
                     "    sum(l_extendedprice) AS sum_base_price, "
                     "    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
                     "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
                     "    avg(l_quantity) AS avg_qty, "
                     "    avg(l_extendedprice) AS avg_price, "
                     "    avg(l_discount) AS avg_disc, "
                     "    count(*) AS count_order "
                     "FROM "
                     "    lineitem "
                     "GROUP BY "
                     "    l_returnflag, "
                     "    l_linestatus;"},

      {TPCH_Q1, "SELECT"
                "    l_returnflag, "
                "    l_linestatus, "
                "    sum(l_quantity) AS sum_qty, "
                "    sum(l_extendedprice) AS sum_base_price, "
                "    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
                "    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
                "    avg(l_quantity) AS avg_qty, "
                "    avg(l_extendedprice) AS avg_price, "
                "    avg(l_discount) AS avg_disc, "
                "    count(*) AS count_order "
                " FROM"
                "    lineitem"
                " WHERE"
                "    l_shipdate <= DATE '1998-09-02'"
                " GROUP BY"
                "     l_returnflag,"
                "     l_linestatus"
                " ORDER BY"
                "     l_returnflag,"
                "     l_linestatus;"},
      {TPCH_Q3, "SELECT"
                "       l_orderkey,"
                "       o_orderdate,"
                "       o_shippriority,"
                "       sum(l_extendedprice * (1 - l_discount)) AS revenue"
                " FROM"
                "       customer,"
                "       orders,"
                "       lineitem"
                " WHERE"
                "       c_mktsegment = 'BUILDING'"
                "       AND c_custkey = o_custkey"
                "       AND l_orderkey = o_orderkey"
                "       AND o_orderdate < date '1995-03-15'"
                "       AND l_shipdate > date '1995-03-15'"
                " GROUP BY"
                "       l_orderkey,"
                "       o_orderdate,"
                "       o_shippriority"
                " ORDER BY"
                "       revenue DESC,"
                "       o_orderdate"
                " LIMIT 10;"},
      {TPCH_Q6, "SELECT"
                "    sum(l_extendedprice * l_discount) AS revenue"
                " FROM"
                "    lineitem"
                " WHERE"
                "    l_shipdate >= CAST('1994-01-01' AS date)"
                "    AND l_shipdate < CAST('1995-01-01' AS date)"
                "    AND l_discount BETWEEN 0.05"
                "    AND 0.07"
                "    AND l_quantity < 24;"},
      {TPCH_Q9, "SELECT"
                "     nation,"
                "     o_year,"
                "     sum(amount) AS sum_profit"
                " FROM ("
                "     SELECT"
                "         n_name AS nation,"
                "         extract(year FROM o_orderdate) AS o_year,"
                "         l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount"
                "     FROM"
                "         part,"
                "         supplier,"
                "         lineitem,"
                "         partsupp,"
                "         orders,"
                "         nation"
                "     WHERE"
                "         s_suppkey = l_suppkey"
                "         AND ps_suppkey = l_suppkey"
                "         AND ps_partkey = l_partkey"
                "         AND p_partkey = l_partkey"
                "         AND o_orderkey = l_orderkey"
                "         AND s_nationkey = n_nationkey"
                "         AND p_name LIKE '%green%') AS profit"
                " GROUP BY"
                "     nation,"
                "     o_year"
                " ORDER BY"
                "     nation,"
                "     o_year DESC;"},
      {TPCH_Q18, "SELECT"
                 "       c_name,"
                 "       c_custkey,"
                 "       o_orderkey,"
                 "       o_orderdate,"
                 "       o_totalprice,"
                 "       sum(l_quantity)"
                 "   FROM"
                 "       customer,"
                 "       orders,"
                 "       lineitem"
                 "   WHERE"
                 "       o_orderkey IN ("
                 "           SELECT"
                 "               l_orderkey"
                 "           FROM"
                 "               lineitem"
                 "           GROUP BY"
                 "               l_orderkey"
                 "           HAVING"
                 "               sum(l_quantity) > 300)"
                 "       AND c_custkey = o_custkey"
                 "       AND o_orderkey = l_orderkey"
                 "   GROUP BY"
                 "       c_name,"
                 "       c_custkey,"
                 "       o_orderkey,"
                 "       o_orderdate,"
                 "       o_totalprice"
                 "   ORDER BY"
                 "       o_totalprice DESC,"
                 "       o_orderdate"
                 "   LIMIT 100;"},
  };
  return queries;
}

static void TPCH_BOSS(benchmark::State& state, int queryIdx, int dataSize,
                      std::string const& engineLibrary, int batchSize, int imputationMethod = 0,
                      std::string variant = "", std::string extension = ".tbl") {
  initBOSSEngine_TPCH(engineLibrary, dataSize, batchSize, imputationMethod, variant, extension);
  auto eval = [&engineLibrary](auto const& expression) mutable {
    return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
  };

  auto const& query = bossQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result) {
    if(!holds_alternative<boss::ComplexExpression>(result)) {
      return false;
    }
    if(get<boss::ComplexExpression>(result).getHead() == "List"_) {
      return false;
    }
    std::cout << queryName << " Error: " << result << std::endl;
    return true;
  };

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "BOSS ";
    eval("Set"_("DebugOutputRelationalOps"_, BOSS_FULL_VERBOSE));
    auto result = eval(query);
    eval("Set"_("DebugOutputRelationalOps"_, false));
    if(!error_handling(result)) {
      std::cout << queryName << " output = " << result << std::endl;
    }
    eval((boss::Expression)1); // dummy eval to cleanup output
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    auto result = eval(query);
    if(error_handling(result)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - BOSS");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      auto result = eval(query);
      if(error_handling(result)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
      eval((boss::Expression)1); // dummy eval to cleanup output
    }
  }
  vtune.stopSampling();
}

static void FCC_BOSS(benchmark::State& state, int queryIdx, std::string const& engineLibrary,
                     int batchSize, int imputationMethod) {
  initBOSSEngine_FCC(engineLibrary, batchSize, imputationMethod);
  auto eval = [&engineLibrary](auto const& expression) mutable {
    return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
  };

  auto const& query = bossQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result) {
    if(!holds_alternative<boss::ComplexExpression>(result)) {
      return false;
    }
    if(get<boss::ComplexExpression>(result).getHead() == "List"_) {
      return false;
    }
    std::cout << queryName << " Error: " << result << std::endl;
    return true;
  };

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "BOSS ";
    eval("Set"_("DebugOutputRelationalOps"_, BOSS_FULL_VERBOSE));
    auto result = eval(query);
    eval("Set"_("DebugOutputRelationalOps"_, false));
    if(!error_handling(result)) {
      std::cout << queryName << " output = " << result << std::endl;
    }
    eval((boss::Expression)1); // dummy eval to cleanup output
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    auto result = eval(query);
    if(error_handling(result)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - BOSS");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      auto result = eval(query);
      if(error_handling(result)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
      eval((boss::Expression)1); // dummy eval to cleanup output
    }
  }
  vtune.stopSampling();
}

static void CDC_BOSS(benchmark::State& state, int queryIdx, std::string const& engineLibrary,
                     int batchSize, int imputationMethod) {
  initBOSSEngine_CDC(engineLibrary, batchSize, imputationMethod);
  auto eval = [&engineLibrary](auto const& expression) mutable {
    return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
  };

  auto const& query = bossQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result) {
    if(!holds_alternative<boss::ComplexExpression>(result)) {
      return false;
    }
    if(get<boss::ComplexExpression>(result).getHead() == "List"_) {
      return false;
    }
    std::cout << queryName << " Error: " << result << std::endl;
    return true;
  };

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "BOSS ";
    eval("Set"_("DebugOutputRelationalOps"_, BOSS_FULL_VERBOSE));
    auto result = eval(query);
    eval("Set"_("DebugOutputRelationalOps"_, false));
    if(!error_handling(result)) {
      std::cout << queryName << " output = " << result << std::endl;
    }
    eval((boss::Expression)1); // dummy eval to cleanup output
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    auto result = eval(query);
    if(error_handling(result)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - BOSS");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      auto result = eval(query);
      if(error_handling(result)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
      eval((boss::Expression)1); // dummy eval to cleanup output
    }
  }
  vtune.stopSampling();
}

static void ACS_BOSS(benchmark::State& state, int queryIdx, std::string const& engineLibrary,
                     int batchSize, int imputationMethod) {
  initBOSSEngine_ACS(engineLibrary, batchSize, imputationMethod);
  auto eval = [&engineLibrary](auto const& expression) mutable {
    return boss::evaluate("EvaluateInEngines"_("List"_(engineLibrary), expression));
  };

  auto const& query = bossQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result) {
    if(!holds_alternative<boss::ComplexExpression>(result)) {
      return false;
    }
    if(get<boss::ComplexExpression>(result).getHead() == "List"_) {
      return false;
    }
    std::cout << queryName << " Error: " << result << std::endl;
    return true;
  };

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "BOSS ";
    eval("Set"_("DebugOutputRelationalOps"_, BOSS_FULL_VERBOSE));
    auto result = eval(query);
    eval("Set"_("DebugOutputRelationalOps"_, false));
    if(!error_handling(result)) {
      std::cout << queryName << " output = " << result << std::endl;
    }
    eval((boss::Expression)1); // dummy eval to cleanup output
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    auto result = eval(query);
    if(error_handling(result)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - BOSS");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      auto result = eval(query);
      if(error_handling(result)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
      eval((boss::Expression)1); // dummy eval to cleanup output
    }
  }
  vtune.stopSampling();
}

static void TPCH_monetdb(benchmark::State& state, int queryIdx, int dataSize) {
  auto& connection =
      initMonetDB(dataSize, ENABLE_INDEXES, MONETDB_MULTITHREADING, USE_FIXED_POINT_NUMERIC_TYPE);
  monetdb_result* result = nullptr;

  auto& query = monetdbQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result, auto&& error) {
    if(error) {
      std::cout << queryName << " Error: " << error << std::endl;
      return true;
    } else if(!result) {
      std::cout << queryName << " Error: NULL output" << std::endl;
      return true;
    }
    return false;
  };

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "MonetDB ";
    std::string verboseQuery = EXPLAIN_QUERY_OUTPUT ? "PLAN " + query : query;
    auto error = monetdb_query(connection, verboseQuery.data(), 1, &result, NULL, NULL);
    if(!error_handling(result, error)) {
      std::cout << queryName << " output = ";
      printResult(result, VERBOSE_QUERY_OUTPUT_MAX_LINES);
    }
    monetdb_cleanup_result(connection, result);
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    auto error = monetdb_query(connection, query.data(), 1, &result, NULL, NULL);
    if(error_handling(result, error)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - MonetDB");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      auto error = monetdb_query(connection, query.data(), 1, &result, NULL, NULL);
      if(error_handling(result, error)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
    }
  }
  vtune.stopSampling();

  monetdb_cleanup_result(connection, result);
}

static void TPCH_duckdb(benchmark::State& state, int queryIdx, int dataSize) {
  auto& connection =
      initDuckDB(dataSize, ENABLE_INDEXES, DUCKDB_MAX_THREADS, USE_FIXED_POINT_NUMERIC_TYPE);

  auto const& query = duckdbQueries().find(queryIdx)->second;
  auto const& queryName = queryNames().find(queryIdx)->second;

  auto error_handling = [&queryName](auto&& result) {
    if(result->HasError()) {
      std::cout << queryName << " Error: " << result->GetError() << std::endl;
      return true;
    }
    return false;
  };

  decltype(connection.Query("")) result;

  if(VERBOSE_QUERY_OUTPUT) {
    std::cout << "DuckDB ";
    result = connection.Query(EXPLAIN_QUERY_OUTPUT ? "EXPLAIN " + query : query);
    if(!error_handling(result)) {
      std::cout << queryName << " output = ";
      result->Print();
    }
  }

  bool failed = false;

  for(auto i = 0; i < BENCHMARK_NUM_WARMPUP_ITERATIONS; ++i) {
    result = connection.Query(query);
    if(error_handling(result)) {
      failed = true;
      break;
    }
  }

  vtune.startSampling(queryName + " - DuckDB");
  for(auto _ : state) { // NOLINT
    if(!failed) {
      result = connection.Query(PROFILE_QUERY_OUTPUT ? "EXPLAIN ANALYZE " + query : query);
      if(error_handling(result)) {
        failed = true;
      }
      benchmark::DoNotOptimize(result);
    }
  }
  vtune.stopSampling();

  if(!failed && PROFILE_QUERY_OUTPUT) {
    std::cout << queryName << " output = ";
    result->Print();
  }
}

static void TPCH_test(benchmark::State& state, int query, int dataSize, int engine, int batchSize,
                      int missingPercent, int imputationMethod, bool simplified) {
  static auto lastEngine = engine;
  if(lastEngine != engine) {
    switch(lastEngine) {
    case MONETDB:
      releaseMonetDB();
      break;
    case DUCKDB:
      releaseDuckDB();
      break;
    default:
      // assuming >= BOSS_ENGINES_START
      resetBOSSEngine();
      break;
    }
  }
  lastEngine = engine;

  switch(engine) {
  case MONETDB: {
    TPCH_monetdb(state, query, dataSize);
  } break;
  case DUCKDB: {
    TPCH_duckdb(state, query, dataSize);
  } break;
  default: {
    if(engine >= BOSS_ENGINES_START) {
      auto engineIndex = engine - BOSS_ENGINES_START;
      if(missingPercent <= 0 && !simplified) {
        TPCH_BOSS(state, query, dataSize, librariesToTest[engineIndex], batchSize);
      } else if(missingPercent == 1) {
        TPCH_BOSS(state, query, dataSize, librariesToTest[engineIndex], batchSize, imputationMethod,
                  simplified ? "simplified" : "original", "001.csv");
      } else {
        TPCH_BOSS(state, query, dataSize, librariesToTest[engineIndex], batchSize, imputationMethod,
                  simplified ? "simplified" : "original", std::to_string(missingPercent) + ".csv");
      }
    }
  } break;
  }
}

static void FCC_test(benchmark::State& state, int query, int engine, int batchSize,
                     int imputationMethod) {
  static auto lastEngine = engine;
  if(lastEngine != engine) {
    switch(lastEngine) {
    case MONETDB:
      releaseMonetDB();
      break;
    case DUCKDB:
      releaseDuckDB();
      break;
    default:
      // assuming >= BOSS_ENGINES_START
      resetBOSSEngine();
      break;
    }
  }
  lastEngine = engine;

  switch(engine) {
  case MONETDB: {
    // not supported
  } break;
  case DUCKDB: {
    // not supported
  } break;
  default: {
    if(engine >= BOSS_ENGINES_START) {
      auto engineIndex = engine - BOSS_ENGINES_START;
      FCC_BOSS(state, query, librariesToTest[engineIndex], batchSize, imputationMethod);
    }
  } break;
  }
}

static void CDC_test(benchmark::State& state, int query, int engine, int batchSize,
                     int imputationMethod) {
  static auto lastEngine = engine;
  if(lastEngine != engine) {
    switch(lastEngine) {
    case MONETDB:
      releaseMonetDB();
      break;
    case DUCKDB:
      releaseDuckDB();
      break;
    default:
      // assuming >= BOSS_ENGINES_START
      resetBOSSEngine();
      break;
    }
  }
  lastEngine = engine;

  switch(engine) {
  case MONETDB: {
    // not supported
  } break;
  case DUCKDB: {
    // not supported
  } break;
  default: {
    if(engine >= BOSS_ENGINES_START) {
      auto engineIndex = engine - BOSS_ENGINES_START;
      CDC_BOSS(state, query, librariesToTest[engineIndex], batchSize, imputationMethod);
    }
  } break;
  }
}

static void ACS_test(benchmark::State& state, int query, int engine, int batchSize,
                     int imputationMethod) {
  static auto lastEngine = engine;
  if(lastEngine != engine) {
    switch(lastEngine) {
    case MONETDB:
      releaseMonetDB();
      break;
    case DUCKDB:
      releaseDuckDB();
      break;
    default:
      // assuming >= BOSS_ENGINES_START
      resetBOSSEngine();
      break;
    }
  }
  lastEngine = engine;

  switch(engine) {
  case MONETDB: {
    // not supported
  } break;
  case DUCKDB: {
    // not supported
  } break;
  default: {
    if(engine >= BOSS_ENGINES_START) {
      auto engineIndex = engine - BOSS_ENGINES_START;
      ACS_BOSS(state, query, librariesToTest[engineIndex], batchSize, imputationMethod);
    }
  } break;
  }
}

template <typename... Args>
benchmark::internal::Benchmark* RegisterBenchmarkNolint([[maybe_unused]] Args... args) {
#ifdef __clang_analyzer__
  // There is not way to disable clang-analyzer-cplusplus.NewDeleteLeaks
  // even though it is perfectly safe. Let's just please clang analyzer.
  return nullptr;
#else
  return benchmark::RegisterBenchmark(args...);
#endif
}

template <typename Func0, typename Func1>
void RegisterForAllEngines(std::string const& name, Func0&& funcVectors, Func1&& funcBOSS,
                           int minRange, int maxRange) {
  auto nameTestVectors = name + "/Vectors";
  RegisterBenchmarkNolint(nameTestVectors.c_str(), funcVectors)->Range(minRange, maxRange);
  for(auto const& bossLibraryName : librariesToTest) {
    auto nameTestBOSS = name + "/";
    nameTestBOSS += bossLibraryName;
    RegisterBenchmarkNolint(nameTestBOSS.c_str(), funcBOSS, bossLibraryName)
        ->Range(minRange, maxRange);
  }
}

template <typename Func0, typename Func1>
void RegisterForAllEngines(std::string const& name, Func0&& funcVectors, Func1&& funcBOSS,
                           int minRange, int maxRange, int minSelectivity, int maxSelectivity) {
  auto nameTestVectors = name + "/Vectors";
  RegisterBenchmarkNolint(nameTestVectors.c_str(), funcVectors)
      ->Ranges({{minRange, maxRange}, {minSelectivity, maxSelectivity}});
  for(auto const& bossLibraryName : librariesToTest) {
    auto nameTestBOSS = name + "/";
    nameTestBOSS += bossLibraryName;
    RegisterBenchmarkNolint(nameTestBOSS.c_str(), funcBOSS, bossLibraryName)
        ->Ranges({{minRange, maxRange}, {minSelectivity, maxSelectivity}});
  }
}

void initAndRunBenchmarks(int argc, char** argv) {
  // read custom arguments
  for(int i = 0; i < argc; ++i) {
    if(std::string("--verbose") == argv[i] || std::string("-v") == argv[i]) {
      VERBOSE_QUERY_OUTPUT = true;
    } else if(std::string("--explain") == argv[i]) {
      EXPLAIN_QUERY_OUTPUT = true;
    } else if(std::string("--profile") == argv[i]) {
      PROFILE_QUERY_OUTPUT = true;
    } else if(std::string("--boss-full-verbose") == argv[i]) {
      BOSS_FULL_VERBOSE = true;
    } else if(std::string("--boss-benchmark-batch-size") == argv[i]) {
      BOSS_BENCHMARK_BATCH_SIZE = true;
    } else if(std::string("--boss-default-batch-size") == argv[i]) {
      if(++i < argc) {
        BOSS_DEFAULT_BATCH_SIZE = atoll(argv[i]);
      }
    } else if(std::string("--no-memory-mapped-files") == argv[i]) {
      BOSS_USE_MEMORY_MAPPED_FILES = false;
    } else if(std::string("--enable-order-preservation-cache") == argv[i]) {
      BOSS_ENABLE_ORDER_PRESERVATION_CACHE = true;
    } else if(std::string("--force-no-op-for-atoms") == argv[i]) {
      BOSS_FORCE_NO_OP_FOR_ATOMS = true;
    } else if(std::string("--disable-expression-partitioning") == argv[i]) {
      BOSS_DISABLE_EXPRESSION_PARTITIONING = true;
    } else if(std::string("--disable-expression-decomposition") == argv[i]) {
      BOSS_DISABLE_EXPRESSION_DECOMPOSITION = true;
    } else if(std::string("--benchmark-num-warmup-iterations") == argv[i]) {
      if(++i < argc) {
        BENCHMARK_NUM_WARMPUP_ITERATIONS = atoi(argv[i]);
      }
    } else if(std::string("--duckdb-max-threads") == argv[i]) {
      if(++i < argc) {
        DUCKDB_MAX_THREADS = atoi(argv[i]);
      }
    } else if(std::string("--monetdb-enable-multithreading") == argv[i]) {
      MONETDB_MULTITHREADING = true;
    } else if(std::string("--fixed-point-numeric-type") == argv[i]) {
      USE_FIXED_POINT_NUMERIC_TYPE = false;
    } else if(std::string("--disable-indexes") == argv[i]) {
      ENABLE_INDEXES = false;
    } else if(std::string("--library") == argv[i]) {
      if(++i < argc) {
        librariesToTest.emplace_back(argv[i]);
      }
    }
  }
  // register micro-benchmarks
  for(int dataSize : std::vector<int>{/*1, 10, 100, */ 1000, 2000, 5000, 10000 /*, 100000*/}) {
    for(int engine = 0; engine < BOSS_ENGINES_START + librariesToTest.size(); ++engine) {
      for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                  ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                  : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
        for(int queryId = MICRO_BENCHMARKING_START; queryId < MICRO_BENCHMARKING_END; ++queryId) {
          std::ostringstream testName;
          testName << "Micro_" << queryNames()[queryId] << "/";
          if(engine < BOSS_ENGINES_START) {
            testName << DBEngineNames[engine];
          } else {
            testName << librariesToTest[engine - BOSS_ENGINES_START];
            if(BOSS_BENCHMARK_BATCH_SIZE) {
              testName << "_batchSize" << batchSize;
            }
          }
          testName << "/" << dataSize << "MB";
          RegisterBenchmarkNolint(testName.str().c_str(), TPCH_test, queryId, dataSize, engine,
                                  batchSize, 0, -1, false);
        }
      }
    }
  }
  // register TPC-H benchmarks
  for(int dataSize : std::vector<int>{1, 10, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000}) {
    for(int engine = 0; engine < BOSS_ENGINES_START + librariesToTest.size(); ++engine) {
      for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                  ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                  : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
        for(int query : std::vector<int>{TPCH_Q1, TPCH_Q3, TPCH_Q6, TPCH_Q9, TPCH_Q18}) { // NOLINT
          std::ostringstream testName;
          testName << "TPCH_Q" << query << "/";
          if(engine < BOSS_ENGINES_START) {
            testName << DBEngineNames[engine];
          } else {
            testName << librariesToTest[engine - BOSS_ENGINES_START];
            if(BOSS_BENCHMARK_BATCH_SIZE) {
              testName << "_batchSize" << batchSize;
            }
          }
          testName << "/" << dataSize << "MB";
          RegisterBenchmarkNolint(testName.str().c_str(), TPCH_test, query, dataSize, engine,
                                  batchSize, 0, -1, false);
        }
      }
    }
  }
  // register TPC-H benchmarks for common query plans (i.e. same as duckdb)
  for(int dataSize : std::vector<int>{1, 10, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000}) {
    for(int engine = 0; engine < BOSS_ENGINES_START + librariesToTest.size(); ++engine) {
      for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                  ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                  : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
        for(int query : std::vector<int>{TPCH_COMMON_PLAN_Q3, TPCH_COMMON_PLAN_Q9,
                                         TPCH_COMMON_PLAN_Q18}) { // NOLINT
          std::ostringstream testName;
          testName << queryNames()[query] << "/";
          if(engine < BOSS_ENGINES_START) {
            testName << DBEngineNames[engine];
          } else {
            testName << librariesToTest[engine - BOSS_ENGINES_START];
            if(BOSS_BENCHMARK_BATCH_SIZE) {
              testName << "_batchSize" << batchSize;
            }
          }
          testName << "/" << dataSize << "MB";
          RegisterBenchmarkNolint(testName.str().c_str(), TPCH_test, query, dataSize, engine,
                                  batchSize, 0, -1, false);
        }
      }
    }
  }
  // register TPC-H benchmarks (with missing values)
  for(int dataSize : std::vector<int>{1, 10, 100, 1000, 2000, 5000, 10000, 100000}) {
    for(int engine = BOSS_ENGINES_START; engine < BOSS_ENGINES_START + librariesToTest.size();
        ++engine) {
      for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                  ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                  : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
        for(int missingPercent : std::vector<int>{0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90}) {
          for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
            for(int imputationMode :
                std::vector<int>{LOCAL, GLOBAL, RANDOM_1P, RANDOM_2P, RANDOM_4P, RANDOM_8P,
                                 RANDOM_16P, RANDOM_32P, RANDOM_64P}) {
              for(int query :
                  std::vector<int>{TPCH_Q1_WITH_EVAL, TPCH_Q3_WITH_EVAL, TPCH_Q6_WITH_EVAL,
                                   TPCH_Q9_WITH_EVAL, TPCH_Q18_WITH_EVAL}) { // NOLINT
                std::ostringstream testName;
                testName << "TPCH_I" << missingPercent << "%_"
                         << imputationMethodNames[imputationMethod] << "-"
                         << imputationModeNames[imputationMode];
                testName << "_Q" << query - (TPCH_Q1_WITH_EVAL - TPCH_Q1) << "/";
                if(engine < BOSS_ENGINES_START) {
                  testName << DBEngineNames[engine];
                } else {
                  testName << librariesToTest[engine - BOSS_ENGINES_START];
                  if(BOSS_BENCHMARK_BATCH_SIZE) {
                    testName << "_batchSize" << batchSize;
                  }
                }
                testName << "/" << dataSize << "MB";
                RegisterBenchmarkNolint(testName.str().c_str(), TPCH_test, query, dataSize, engine,
                                        batchSize, missingPercent,
                                        imputationMethod + imputationMode, false);
              }
            }
          }
        }
      }
    }
  }
  // register simplified TPC-H Q1/Q6 benchmarks with missing data and various imputation methods
  for(int dataSize : std::vector<int>{1, 10, 100, 1000, 2000, 5000, 10000, 100000}) {
    for(int engine = BOSS_ENGINES_START; engine < BOSS_ENGINES_START + librariesToTest.size();
        ++engine) {
      for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                  ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                  : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
        for(int missingPercent : std::vector<int>{0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90}) {
          for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
            for(int imputationMode :
                std::vector<int>{LOCAL, GLOBAL, RANDOM_1P, RANDOM_2P, RANDOM_4P, RANDOM_8P,
                                 RANDOM_16P, RANDOM_32P, RANDOM_64P}) {
              for(int query :
                  std::vector<int>{TPCH_Q1_MODIFIED, TPCH_Q6_MODIFIED, TPCH_Q1_FULL_EVAL}) {
                std::ostringstream testName;
                testName << "TPCH_I" << missingPercent << "%_"
                         << imputationMethodNames[imputationMethod] << "-"
                         << imputationModeNames[imputationMode];
                testName << "_" << queryNames()[query] << "/";
                if(engine < BOSS_ENGINES_START) {
                  testName << DBEngineNames[engine];
                } else {
                  testName << librariesToTest[engine - BOSS_ENGINES_START];
                  if(BOSS_BENCHMARK_BATCH_SIZE) {
                    testName << "_batchSize" << batchSize;
                  }
                }
                testName << "/" << dataSize << "MB";
                RegisterBenchmarkNolint(testName.str().c_str(), TPCH_test, query, dataSize, engine,
                                        batchSize, missingPercent,
                                        imputationMethod + imputationMode, true);
              }
            }
          }
        }
      }
    }
  }
  // register imputation benchmarks for CDC dataset
  for(int engine = BOSS_ENGINES_START; engine < BOSS_ENGINES_START + librariesToTest.size();
      ++engine) {
    for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
      for(int query : std::vector<int>{CDC_Q1, CDC_Q2, CDC_Q3, CDC_Q4, CDC_Q5}) {
        for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
          for(int imputationMode :
              std::vector<int>{LOCAL, GLOBAL, RANDOM_1P, RANDOM_2P, RANDOM_4P, RANDOM_8P,
                               RANDOM_16P, RANDOM_32P, RANDOM_64P}) {
            std::ostringstream testName;
            testName << "CDC_I_" << imputationMethodNames[imputationMethod] << "-"
                     << imputationModeNames[imputationMode];
            testName << "_" << queryNames()[query] << "/";
            if(engine < BOSS_ENGINES_START) {
              testName << DBEngineNames[engine];
            } else {
              testName << librariesToTest[engine - BOSS_ENGINES_START];
              if(BOSS_BENCHMARK_BATCH_SIZE) {
                testName << "_batchSize" << batchSize;
              }
            }
            RegisterBenchmarkNolint(testName.str().c_str(), CDC_test, query, engine, batchSize,
                                    imputationMethod + imputationMode);
          }
        }
      }
    }
  }
  // register imputation benchmarks for FCC dataset
  for(int engine = BOSS_ENGINES_START; engine < BOSS_ENGINES_START + librariesToTest.size();
      ++engine) {
    for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
      for(int query : std::vector<int>{FCC_Q6, FCC_Q7, FCC_Q8, FCC_Q9}) {
        for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
          for(int imputationMode :
              std::vector<int>{LOCAL, GLOBAL, RANDOM_1P, RANDOM_2P, RANDOM_4P, RANDOM_8P,
                               RANDOM_16P, RANDOM_32P, RANDOM_64P}) {
            std::ostringstream testName;
            testName << "FCC_I_" << imputationMethodNames[imputationMethod] << "-"
                     << imputationModeNames[imputationMode];
            testName << "_" << queryNames()[query] << "/";
            if(engine < BOSS_ENGINES_START) {
              testName << DBEngineNames[engine];
            } else {
              testName << librariesToTest[engine - BOSS_ENGINES_START];
              if(BOSS_BENCHMARK_BATCH_SIZE) {
                testName << "_batchSize" << batchSize;
              }
            }
            RegisterBenchmarkNolint(testName.str().c_str(), FCC_test, query, engine, batchSize,
                                    imputationMethod + imputationMode);
          }
        }
      }
    }
  }
  // register imputation benchmarks for ACS dataset
  for(int engine = BOSS_ENGINES_START; engine < BOSS_ENGINES_START + librariesToTest.size();
      ++engine) {
    for(int64_t batchSize : (BOSS_BENCHMARK_BATCH_SIZE && engine >= BOSS_ENGINES_START)
                                ? std::vector<int64_t>{1000, 2000, 5000, 10000, 50000, 100000}
                                : std::vector<int64_t>{BOSS_DEFAULT_BATCH_SIZE}) {
      for(int query : std::vector<int>{ACS}) {
        for(int imputationMethod : std::vector<int>{MEAN, HOTDECK, DTREE, INTERPOLATE, NO_OP}) {
          for(int imputationMode :
              std::vector<int>{LOCAL, GLOBAL, RANDOM_1P, RANDOM_2P, RANDOM_4P, RANDOM_8P,
                               RANDOM_16P, RANDOM_32P, RANDOM_64P}) {
            std::ostringstream testName;
            testName << "ACS_I_" << imputationMethodNames[imputationMethod] << "-"
                     << imputationModeNames[imputationMode];
            testName << "_" << queryNames()[query] << "/";
            if(engine < BOSS_ENGINES_START) {
              testName << DBEngineNames[engine];
            } else {
              testName << librariesToTest[engine - BOSS_ENGINES_START];
              if(BOSS_BENCHMARK_BATCH_SIZE) {
                testName << "_batchSize" << batchSize;
              }
            }
            RegisterBenchmarkNolint(testName.str().c_str(), ACS_test, query, engine, batchSize,
                                    imputationMethod + imputationMode);
          }
        }
      }
    }
  }
  // initialise and run google benchmark
  ::benchmark::Initialize(&argc, argv, ::benchmark::PrintDefaultHelp);
  ::benchmark::RunSpecifiedBenchmarks();
}

int main(int argc, char** argv) {
  try {
    initAndRunBenchmarks(argc, argv);
  } catch(std::exception& e) {
    std::cerr << "caught exception in main: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch(...) {
    std::cerr << "unhandled exception." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
