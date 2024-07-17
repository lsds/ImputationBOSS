#include "spdlog/cfg/env.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <BOSS.hpp>
#include <Engine.hpp>
#include <Expression.hpp>
#include <algorithm>
std::ostream& operator<<(std::ostream& s, std::vector<std::int64_t> const& input /*unused*/) {
  std::for_each(begin(input), prev(end(input)),
                [&s = s << "["](auto&& element) { s << element << ", "; });
  return (input.empty() ? s : (s << input.back())) << "]";
}
#include <ExpressionUtilities.hpp>
#include <arrow/array.h>
#include <cstring>
#include <iostream>
#include <regex>
#include <set>
#include <spdlog/common.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <wstp.h>

#define STRINGIFY(x) #x        // NOLINT
#define STRING(x) STRINGIFY(x) // NOLINT

namespace boss::engines::wolfram {
using std::move;
using WolframExpressionSystem = ExtensibleExpressionSystem<>;
using AtomicExpression = WolframExpressionSystem::AtomicExpression;
using ComplexExpression = WolframExpressionSystem::ComplexExpression;
using Expression = WolframExpressionSystem::Expression;
using ExpressionArguments = WolframExpressionSystem::ExpressionArguments;
using ExpressionSpanArguments = WolframExpressionSystem::ExpressionSpanArguments;
using expressions::generic::ArgumentWrapper;
using expressions::generic::ExpressionArgumentsWithAdditionalCustomAtomsWrapper;

class Engine : public boss::Engine {
private:
  class EngineImplementation& impl;
  friend class EngineImplementation;

public:
  Engine(Engine&) = delete;
  Engine& operator=(Engine&) = delete;
  Engine(Engine&&) = default;
  Engine& operator=(Engine&&) = delete;
  Engine();
  boss::Expression evaluate(Expression&& e);
  ~Engine();
};
} // namespace boss::engines::wolfram

namespace nasty = boss::utilities::nasty;
namespace boss::engines::wolfram {
using ExpressionBuilder = boss::utilities::ExtensibleExpressionBuilder<WolframExpressionSystem>;
static ExpressionBuilder operator""_(const char* name, size_t /*unused*/) {
  return ExpressionBuilder(name);
};

using std::set;
using std::string;
using std::to_string;
using std::vector;
using std::string_literals::operator""s;
using std::endl;

static class WolframLogStream {
public:
  WolframLogStream() { spdlog::cfg::load_env_levels(); }

  template <typename T> WolframLogStream& operator<<(T unused) {
    str << unused;
    return *this;
  }
  WolframLogStream& operator<<(std::ostream& (* /*pf*/)(std::ostream&)) {
    logger().trace(str.str());
    str.str("");
    return *this;
  };

private:
  static spdlog::logger& logger() {
    static auto instance = [] {
      try {
        return spdlog::basic_logger_mt("wolframlog.m", "wolframlog.m");
      } catch(const spdlog::spdlog_ex& ex) {
        return spdlog::stdout_color_mt("console");
      }
    }();
    static std::once_flag flag;
    std::call_once(flag, [&] { instance->sinks()[0]->set_pattern("%v"); });
    return *instance;
  }

  std::stringstream str;

} console; // NOLINT

struct EngineImplementation {
  constexpr static char const* const DefaultNamespace = "BOSS`";
  WSENV const environment = {}; // NOLINT(misc-misplaced-const)
  WSLINK const link = {};       // NOLINT(misc-misplaced-const)

  static char const* removeNamespace(char const* symbolName) {
    if(strncmp(DefaultNamespace, symbolName, strlen(DefaultNamespace)) == 0) {
      return symbolName + strlen(DefaultNamespace);
    }
    return symbolName;
  }

  static auto mangle(std::string normalizedName) {
    normalizedName = std::regex_replace(normalizedName, std::regex("_"), "$$0");
    normalizedName = std::regex_replace(normalizedName, std::regex("\\."), "$$1");
    return normalizedName;
  }

  static auto demangle(std::string normalizedName) {
    normalizedName = std::regex_replace(normalizedName, std::regex("$0"), "_");
    normalizedName = std::regex_replace(normalizedName, std::regex("$1"), ".");
    return normalizedName;
  }

  void putExpressionOnLink(Expression const& expression, std::string const& namespaceIdentifier) {
    std::visit(
        boss::utilities::overload(
            [&](bool a) {
              console << (a ? "True" : "False");
              WSPutSymbol(link, (a ? "True" : "False"));
            },
            [&](std::vector<bool>::reference a) {
              console << (a ? "True" : "False");
              WSPutSymbol(link, (a ? "True" : "False"));
            },
            [&](std::int64_t a) {
              console << a;
              WSPutInteger64(link, a);
            },
            [&](std::vector<int64_t> values) {
              ExpressionSpanArguments vs;
              vs.emplace_back(Span<std::int64_t>({values.begin(), values.end()}));
              putExpressionOnLink(ComplexExpression("List"_, {}, {}, std::move(vs)),
                                  namespaceIdentifier);
            },
            [&](char const* a) {
              console << a;
              WSPutString(link, a);
            },
            [&](std::double_t a) {
              console << a;
              WSPutDouble(link, a);
            },
            [&](Symbol const& a) {
              auto normalizedName = mangle(a.getName());
              auto unnamespacedSymbols = set<string>{"TimeZone"};
              auto namespaced =
                  (unnamespacedSymbols.count(normalizedName) > 0 ? "" : namespaceIdentifier) +
                  normalizedName;
              console << namespaced;
              WSPutSymbol(link, namespaced.c_str());
            },
            [&](std::string const& a) {
              console << "\"" << a << "\"";
              WSPutString(link, a.c_str());
            },
            [&](ComplexExpression const& expression) {
              auto headName = namespaceIdentifier + expression.getHead().getName();
              auto process = [&](auto const& arguments) {
                if(headName == namespaceIdentifier + "list" ||
                   headName == namespaceIdentifier + "ArrowArrayPtr") {
                  headName = "List";
                }
                console << (headName) << "[";
                WSPutFunction(link, (headName).c_str(), (int)arguments.size());
                for(auto it = arguments.begin(); it != arguments.end(); ++it) {
                  auto const& argument = *it;
                  if(it != arguments.begin()) {
                    console << ", ";
                  }
                  if constexpr(std::is_same_v<std::decay_t<decltype(argument)>, Expression>) {
                    putExpressionOnLink(argument, namespaceIdentifier);
                  } else {
                    std::visit(
                        [this, &namespaceIdentifier](auto&& argument) {
                          return putExpressionOnLink(argument, namespaceIdentifier);
                        },
                        argument.getArgument());
                  }
                }
                console << "]";
              };
              if((headName == namespaceIdentifier + "ArrowArrayPtr")) {
                vector<Expression> result;
                // ArgumentWrapper<true, std::vector<>>
                ExpressionArgumentsWithAdditionalCustomAtomsWrapper<std::tuple<> const, true> x =
                    expression.getArguments();
                ArgumentWrapper<true> v = x.at(0);
                auto const& arrowArray = nasty::reconstructArrowArray(get<std::int64_t>(v));
                auto int64_array = std::static_pointer_cast<arrow::Int64Array>(arrowArray);
                result.reserve(arrowArray->length());
                for(auto i = 0U; i < arrowArray->length(); i++) {
                  result.emplace_back(int64_array->Value(i));
                }
                process(result);
              } else {
                process(expression.getArguments());
              }
            },
            [](auto /*args*/) { throw std::runtime_error("unexpected argument type"); }),
        (Expression::SuperType const&)expression);
  }

  boss::Expression readExpressionFromLink() const {
    auto resultType = WSGetType(link);
    if(resultType == WSTKSTR) {
      char const* resultAsCString = nullptr;
      WSGetString(link, &resultAsCString);
      auto result = std::string(resultAsCString);
      WSReleaseString(link, resultAsCString);
      return result;
    }
    if(resultType == WSTKINT) {
      wsint64 result = 0;
      static_assert(sizeof(wsint64) == sizeof(std::int64_t),
                    "mathematica and c++ compiler don't agree on the size of a 64 bit integer");
      WSGetInteger64(link, &result);
      return result;
    }
    if(resultType == WSTKREAL) {
      double result = 0;
      WSGetDouble(link, &result);
      return result;
    }
    if(resultType == WSTKFUNC) {
      auto const* resultHead = "";
      auto numberOfArguments = 0;
      auto success = WSGetFunction(link, &resultHead, &numberOfArguments);
      if(success == 0) {
        throw std::runtime_error("error when getting function "s + WSErrorMessage(link));
      }
      auto resultArguments = vector<boss::Expression>();
      for(auto i = 0U; i < numberOfArguments; i++) {
        resultArguments.push_back(readExpressionFromLink());
      }
      auto result = boss::ComplexExpression(boss::Symbol(removeNamespace(resultHead)),
                                            std::move(resultArguments));
      WSReleaseSymbol(link, resultHead);
      return std::move(result);
    }
    if(resultType == WSTKSYM) {
      auto const* result = "";
      WSGetSymbol(link, &result);
      auto resultingSymbol = Symbol(demangle(removeNamespace(result)));
      WSReleaseSymbol(link, result);
      if(std::string("True") == resultingSymbol.getName()) {
        return true;
      }
      if(std::string("False") == resultingSymbol.getName()) {
        return false;
      }
      return resultingSymbol;
    }
    if(resultType == WSTKERROR) {
      const char* messageAsCString = WSErrorMessage(link);
      auto message = string(messageAsCString);
      WSReleaseErrorMessage(link, messageAsCString);
      throw std::runtime_error(message);
    }
    throw std::logic_error("unsupported return type: " + std::to_string(resultType));
  }

  static ExpressionBuilder namespaced(ExpressionBuilder const& builder) {
    return ExpressionBuilder(Symbol(DefaultNamespace + Symbol(builder).getName()));
  }
  static Symbol namespaced(Symbol const& name) {
    return std::move(ExpressionBuilder(Symbol(DefaultNamespace + name.getName())));
  }
  static ComplexExpression namespaced(ComplexExpression&& name) {
    return std::move(ComplexExpression(Symbol(DefaultNamespace + name.getHead().getName()),
                                       std::move(name.getArguments())));
  }

  void evalWithoutNamespace(Expression&& expression) { evaluate(std::move(expression), ""); };

  void DefineFunction(Symbol&& name, std::initializer_list<ComplexExpression>&& arguments,
                      Expression&& definition, vector<Symbol>&& attributes = {}) {
    ExpressionArguments args;
    std::transform(arguments.begin(), arguments.end(), back_inserter(args),
                   [](auto&& arg) { return arg.clone(); });
    evalWithoutNamespace("SetDelayed"_(
        std::move(namespaced(ComplexExpression(name, std::move(args)))), std::move(definition)));
    for(auto const& it : attributes) {
      evalWithoutNamespace("SetAttributes"_(namespaced(name), it));
    }
  };

  void loadRelationalOperators() {
    DefineFunction("Where"_, {"Pattern"_("condition"_, "Blank"_())},
                   "Function"_("tuple"_, "ReplaceAll"_("Unevaluated"_("condition"_), "tuple"_)),
                   {"HoldFirst"_});

    DefineFunction("Column"_,
                   {"Pattern"_("input"_, "Blank"_()), "Pattern"_("column"_, "Blank"_("Integer"_))},
                   "Extract"_("input"_, "column"_), {"HoldFirst"_});

    DefineFunction(
        "As"_, {"Pattern"_("projections"_, "BlankSequence"_())},
        "Function"_("tuple"_,
                    "Association"_("Thread"_("Rule"_(
                        "Part"_("List"_("projections"_), "Span"_(1, "All"_, 2)),
                        "ReplaceAll"_("Part"_("List"_("projections"_), "Span"_(2, "All"_, 2)),
                                      "tuple"_))))),
        {"HoldAll"_});

    DefineFunction("By"_, {"Pattern"_("projections"_, "BlankSequence"_())},
                   "Function"_("tuple"_, "ReplaceAll"_("List"_("projections"_), "tuple"_)),
                   {"HoldAll"_});

    DefineFunction(
        "GetPersistentTableIfSymbol"_, {"Pattern"_("input"_, "Blank"_("Symbol"_))},
        "If"_("Greater"_("Length"_("Database"_("input"_)), 0),
              "MovingMap"_(
                  "Function"_("Association"_("ReplaceAll"_(
                      "Normal"_("Part"_("Slot"_(1), 2)),
                      "RuleDelayed"_(
                          "HoldPattern"_(
                              "Rule"_("Pattern"_("attribute"_, "Blank"_()),
                                      namespaced("Interpolate"_)("Pattern"_("by"_, "Blank"_())))),
                          "Rule"_("attribute"_,
                                  "Divide"_("Total"_("Map"_("Extract"_("Key"_("attribute"_)),
                                                            ("Drop"_("Slot"_(1), "List"_(2))))),
                                            2)))))),
                  "Database"_("input"_), "List"_(2, "Center"_, "Automatic"_), "Fixed"),
              "Database"_("input"_)));
    DefineFunction("GetPersistentTableIfSymbol"_, {"Pattern"_("input"_, "Blank"_())}, "input"_,
                   {"HoldAll"_});

    DefineFunction(
        "Project"_, {"Pattern"_("input"_, "Blank"_()), "Pattern"_("projection"_, "Blank"_())},
        "Map"_("projection"_, namespaced("GetPersistentTableIfSymbol"_)("input"_)), {"HoldAll"_});

    DefineFunction(
        "ProjectAll"_, {"Pattern"_("input"_, "Blank"_()), "Pattern"_("projection"_, "Blank"_())},
        "Map"_("KeyMap"_("Function"_("oldname"_, "Symbol"_("StringJoin"_(
                                                     DefaultNamespace, "SymbolName"_("projection"_),
                                                     "$1", "SymbolName"_("oldname"_))))),
               namespaced("GetPersistentTableIfSymbol"_)("input"_)),
        {"HoldAll"_});

    DefineFunction(
        "Select"_, {"Pattern"_("input"_, "Blank"_()), "Pattern"_("predicate"_, "Blank"_())},
        "Select"_(namespaced("GetPersistentTableIfSymbol"_)("input"_), "predicate"_), {"HoldAll"_});

    DefineFunction(
        "Group"_,
        {"Pattern"_("inputName"_, "Blank"_()), "Pattern"_("groupFunction"_, "Blank"_()),
         "Pattern"_("aggregateFunctions"_, "BlankSequence"_())},
        "With"_(
            "List"_("Set"_("input"_, namespaced("GetPersistentTableIfSymbol"_)("inputName"_))),

            "KeyValueMap"_(
                "Function"_(
                    "List"_("groupkey"_, "groupresult"_),
                    "Append"_("groupresult"_,
                              "Thread"_("Rule"_(
                                  "Quiet"_("Check"_("Extract"_("groupFunction"_, "List"_(2, 1)),
                                                    "Unique"_("groupKey"_))),
                                  "groupkey"_)))),
                "GroupBy"_(
                    "input"_, "groupFunction"_,
                    "Function"_(
                        "groupedInput"_,
                        "Merge"_(
                            "Map"_(
                                "Function"_(
                                    "aggregateFunction"_,
                                    "Construct"_(
                                        "Switch"_("aggregateFunction"_, namespaced("Count"_),
                                                  "Composition"_(
                                                      "Association"_,
                                                      "Construct"_("CurryApplied"_("Rule"_, 2),
                                                                   "Count"_),
                                                      "Length"_),
                                                  "Blank"_(),
                                                  "Composition"_(
                                                      "Fold"_("Plus"_),
                                                      "Apply"_("KeyTake"_, "aggregateFunction"_))),
                                        "groupedInput"_)),
                                "List"_("aggregateFunctions"_)),
                            "First"_))))));

    DefineFunction(
        "Group"_,
        {"Pattern"_("inputName"_, "Blank"_()), "Pattern"_("aggregateFunctions"_, "Blank"_())},
        "With"_("List"_("Set"_("input"_, namespaced("GetPersistentTableIfSymbol"_)("inputName"_))),
                "List"_("Merge"_(
                    "Map"_("Function"_(
                               "aggregateFunction"_,
                               "Construct"_(
                                   "Switch"_(
                                       "aggregateFunction"_, namespaced("Count"_),
                                       "Composition"_(
                                           "Association"_,
                                           "Construct"_("CurryApplied"_("Rule"_, 2), "Count"_),
                                           "Length"_),
                                       "Blank"_(),
                                       "Composition"_("Fold"_("Plus"_),
                                                      "Apply"_("KeyTake"_, "aggregateFunction"_))),
                                   "input"_)),
                           "List"_("aggregateFunctions"_)),
                    "First"_))),
        {"HoldAll"_});

    DefineFunction("Order"_,
                   {"Pattern"_("input"_, "Blank"_()), "Pattern"_("orderFunction"_, "Blank"_())},
                   "SortBy"_(namespaced("GetPersistentTableIfSymbol"_)("input"_), "orderFunction"_),
                   {"HoldAll"_});

    DefineFunction("Top"_,
                   {"Pattern"_("input"_, "Blank"_()), "Pattern"_("orderFunction"_, "Blank"_()),
                    "Pattern"_("number"_, "Blank"_("Integer"_))},
                   "MinimalBy"_(namespaced("GetPersistentTableIfSymbol"_)("input"_),
                                "orderFunction"_, "UpTo"_("number"_)),
                   {"HoldAll"_});

    DefineFunction(
        "Join"_,
        {"Pattern"_("left"_, "Blank"_()), "Pattern"_("right"_, "Blank"_()),
         "Pattern"_("predicate"_, "Blank"_("Function"_))},
        "Select"_("Flatten"_("Outer"_("Composition"_("Merge"_("First"_), "List"_),
                                      namespaced("GetPersistentTableIfSymbol"_)("left"_),
                                      namespaced("GetPersistentTableIfSymbol"_)("right"_), 1),
                             1),
                  "predicate"_));

    DefineFunction(
        "Join"_,
        {"Pattern"_("leftInput"_, "Blank"_()), "Pattern"_("rightInput"_, "Blank"_()),
         namespaced("Where"_)(namespaced("Equal"_)("Pattern"_("leftAttribute"_, "Blank"_()),
                                                   "Pattern"_("rightAttribute"_, "Blank"_())))},
        "With"_(
            "List"_("Set"_("ht"_, "CreateDataStructure"_("HashTable"))),
            "CompoundExpression"_(
                "Map"_("Function"_(
                           "buildTuple"_,
                           "ht"_("Insert", "Rule"_("ReplaceAll"_("leftAttribute"_, "buildTuple"_),
                                                   "Append"_("ht"_("Lookup",
                                                                   "ReplaceAll"_("leftAttribute"_,
                                                                                 "buildTuple"_),
                                                                   "Function"_("List"_())),
                                                             "buildTuple"_)))),
                       namespaced("GetPersistentTableIfSymbol"_)("leftInput"_)),
                "Flatten"_(
                    "Map"_(
                        "Function"_(
                            "probeTuple"_,
                            "Map"_("Function"_(
                                       "buildTuple"_,
                                       "Merge"_("List"_("probeTuple"_, "buildTuple"_), "First"_)),
                                   "ht"_("Lookup", "ReplaceAll"_("rightAttribute"_, "probeTuple"_),
                                         "Function"_("List"_())))),
                        namespaced("GetPersistentTableIfSymbol"_)("rightInput"_)),
                    1))));
  }

  void loadDataLoadingOperators() {
    DefineFunction(
        "Load"_, {"Pattern"_("relation"_, "Blank"_()), "Pattern"_("from"_, "Blank"_("String"_))},
        "CompoundExpression"_(
            "Set"_("Database"_("relation"_),
                   "Map"_("Function"_("tuple"_,
                                      "Association"_("Thread"_("Rule"_(
                                          "Map"_("First"_, "Schema"_("relation"_)), "tuple"_)))),
                          "Normal"_("SemanticImport"_(
                              "from"_,
                              "Map"_("Function"_(
                                         "type"_,
                                         "Replace"_("Extract"_("type"_, 2),
                                                    "List"_( //
                                                        "Rule"_(namespaced("INTEGER"_), "Integer"),
                                                        "Rule"_(namespaced("CHAR"_), "String"),
                                                        "Rule"_(namespaced("VARCHAR"_), "String"),
                                                        "Rule"_(namespaced("DECIMAL"_), "Number"),
                                                        "Rule"_(namespaced("DATE"_), "Date")))),
                                     "Schema"_("relation"_)),
                              "Rule"_("Delimiters", "List"_(",")),
                              "Rule"_("ExcludedLines"_, "List"_(1)), "Rule"_("HeaderLines"_, 0))))),
            "List"_("List"_("relation"_))));
    DefineFunction("File"_, {"Pattern"_("pathComponents"_, "BlankSequence"_())},
                   "FileNameJoin"_("List"_("pathComponents"_)));
  }

  void loadDDLOperators() {
    DefineFunction(
        "CreateTable"_,
        {"Pattern"_("relation"_, "Blank"_()), "Pattern"_("attributes"_, "BlankSequence"_())},
        "CompoundExpression"_(
            "Set"_("Database"_("relation"_), "List"_()),
            "Set"_("Schema"_("relation"_),
                   "Map"_("Function"_("a"_, "If"_("SameQ"_(namespaced("List"_), "Head"_("a"_)),
                                                  "a"_, "List"_("a"_))),
                          "List"_("attributes"_))),
            "relation"_),
        {"HoldFirst"_});

    DefineFunction(
        "InsertInto"_,
        {"Pattern"_("relation"_, "Blank"_()), "Pattern"_("tuple"_, "BlankSequence"_())},
        "CompoundExpression"_(
            "AppendTo"_("Database"_("relation"_),
                        "Association"_("Thread"_(
                            "Rule"_("Map"_("First"_, "Schema"_("relation"_)), "List"_("tuple"_))))),
            "Null"_),
        {"HoldFirst"_});

    DefineFunction(
        "AttachColumns"_,
        {"Pattern"_("relation"_, "Blank"_()), "Pattern"_("columns"_, "BlankSequence"_())},
        "CompoundExpression"_(
            "Map"_("Function"_("tuple"_, "AppendTo"_("Database"_("relation"_),
                                                     "Association"_("Thread"_("Rule"_(
                                                         "Map"_("First"_, "Schema"_("relation"_)),
                                                         "tuple"_))))),
                   "Thread"_("List"_(("columns"_)))),
            "Null"_),
        {"HoldFirst"_});

    DefineFunction(
        "Column"_,
        {"Pattern"_("name"_, "Blank"_("Symbol"_)), "Pattern"_("data"_, "Blank"_("List"_))},
        "Rule"_("name"_, "data"_), {"HoldFirst"_});

    DefineFunction("ScanColumns"_, {"Pattern"_("columns"_, "BlankSequence"_())},
                   "Map"_("Association"_, "Transpose"_("KeyValueMap"_(
                                              "Function"_("List"_("key"_, "value"_),
                                                          "Thread"_("Rule"_("key"_, "value"_))),
                                              "Association"_("columns"_)))),
                   {"HoldFirst"_});
  }

  void loadSymbolicOperators() {
    DefineFunction(
        "Assuming"_,
        {"Pattern"_("input"_, "Blank"_()), "Pattern"_("assumptions"_, "BlankSequence"_())},
        "ReplaceAll"_(namespaced("GetPersistentTableIfSymbol"_)("input"_),
                      "Rule"_("First"_("assumptions"_), "Last"_("assumptions"_))),
        {"HoldAll"_});
  }

  void loadShimLayer() {
    evalWithoutNamespace("Set"_("BOSSVersion"_, 1));

    for(std::string const& it : vector{
            //
            "And",        "Apply",
            "DateObject", "Equal",
            "Evaluate",   "Extract",
            "Greater",    "Length",
            "List",       "Less",
            "Minus",      "Not",
            "Or",         "Plus",
            "Rule",       "Set",
            "SortBy",     "StringContainsQ",
            "StringJoin", "Symbol",
            "Times",      "UndefinedFunction",
            "UnixTime",   "Values",
            //
        }) {
      evalWithoutNamespace("Set"_(namespaced(Symbol(it)), Symbol("System`" + it)));
    }
    evalWithoutNamespace("Set"_(namespaced("Date"_), "System`DateObject"_));
    evalWithoutNamespace("Set"_(namespaced("StringContains"_), "System`StringContainsQ"_));

    DefineFunction("Function"_,
                   {"Pattern"_("arg"_, "Blank"_()), "Pattern"_("definition"_, "Blank"_())},
                   "Function"_("arg"_, "definition"_), {"HoldRest"_});
    DefineFunction("Function"_, {"Pattern"_("definition"_, "Blank"_())}, "Function"_("definition"_),
                   {"HoldRest"_});

    DefineFunction(
        "Return"_, {"Pattern"_("result"_, "Blank"_("List"_))},
        "ReplaceAll"_(
            "Map"_("Function"_("x"_, "If"_("MatchQ"_("x"_, "Blank"_("Association"_)),
                                           "Values"_("x"_), "x"_)),
                   "result"_),
            "Rule"_("DateObject"_("List"_("Pattern"_("year"_, "Blank"_()),
                                          "Pattern"_("month"_, "Blank"_()),
                                          "Pattern"_("day"_, "Blank"_()), "BlankSequence"_()),
                                  "BlankSequence"_()),
                    "Date"_("year"_, "month"_, "day"_))));

    DefineFunction(
        "Return"_, {"Pattern"_("input"_, "Blank"_("Symbol"_))},
        "ReplaceAll"_(
            "Map"_("Function"_("x"_, "If"_("MatchQ"_("x"_, "Blank"_("Association"_)),
                                           "Values"_("x"_), "x"_)),
                   namespaced("GetPersistentTableIfSymbol"_)("input"_)),
            "Rule"_("DateObject"_("List"_("Pattern"_("year"_, "Blank"_()),
                                          "Pattern"_("month"_, "Blank"_()),
                                          "Pattern"_("day"_, "Blank"_()), "BlankSequence"_()),
                                  "BlankSequence"_()),
                    "Date"_("year"_, "month"_, "day"_))));

    DefineFunction("Schema"_, {"Pattern"_("result"_, "Blank"_("List"_))},
                   "If"_("Greater"_("Length"_("result"_), 0),
                         "Map"_("Composition"_("List"_), "Keys"_("First"_("result"_))), "List"_()));

    loadDDLOperators();

    loadRelationalOperators();
    loadDataLoadingOperators();
    loadSymbolicOperators();
  };

  EngineImplementation()
      : environment([] {
          if(auto* environment = WSInitialize(nullptr)) {
            return environment;
          }
          throw std::runtime_error("could not initialize wstp environment");
        }()),
        link([this] {
          auto error = 0;
          auto* link = WSOpenString(
              environment,
              "-linkmode launch -linkname \"" STRING(MATHEMATICA_KERNEL_EXECUTABLE) "\" -wstp",
              &error);
          if(error != 0) {
            throw std::runtime_error("could not open wstp link -- error code: " + to_string(error));
          }
          return link;
        }()) {}

  EngineImplementation(EngineImplementation&&) = default;
  EngineImplementation(EngineImplementation const&) = delete;
  EngineImplementation& operator=(EngineImplementation&&) = delete;
  EngineImplementation& operator=(EngineImplementation const&) = delete;

  ~EngineImplementation() {
    WSClose(link);
    WSDeinitialize(environment);
  }

  boss::Expression evaluate(Expression&& e,
                            std::string const& namespaceIdentifier = DefaultNamespace) {
    putExpressionOnLink("Return"_(std::move(e)), namespaceIdentifier);
    console << ";" << endl;
    WSEndPacket(link);
    int pkt = 0;
    while(((pkt = WSNextPacket(link)) != 0) && (pkt != RETURNPKT)) {
      WSNewPacket(link);
    }
    return readExpressionFromLink();
  }
};

Engine::Engine() : impl([]() -> EngineImplementation& { return *(new EngineImplementation()); }()) {
  impl.loadShimLayer();
}
Engine::~Engine() { delete &impl; }

boss::Expression Engine::evaluate(Expression&& e) { return impl.evaluate(std::move(e)); }
} // namespace boss::engines::wolfram

#ifdef _WIN32
#define BOSS_WOLFRAM_API __declspec(dllexport)
#else
#define BOSS_WOLFRAM_API
#endif // _WIN32

static auto& enginePtr(bool initialise = true) {
  static auto engine = std::unique_ptr<boss::engines::wolfram::Engine>();
  if(!engine && initialise) {
    engine.reset(new boss::engines::wolfram::Engine());
  }
  return engine;
}

extern "C" BOSS_WOLFRAM_API BOSSExpression* evaluate(BOSSExpression* e) {
  static std::mutex m;
  std::lock_guard lock(m);
  auto* r = new BOSSExpression{enginePtr()->evaluate(e->delegate.clone())};
  return r;
};

extern "C" BOSS_WOLFRAM_API void reset() { enginePtr(false).reset(nullptr); }

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
  switch(fdwReason) {
  case DLL_PROCESS_ATTACH:
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
    break;
  case DLL_PROCESS_DETACH:
    // Make sure to call reset instead of letting destructors to be called.
    // It leaves the engine unique_ptr in a non-dangling state
    // in case the depending process still want to call reset() during its own destruction
    // (which does happen in a unpredictable order if it is itself a dll:
    // https://devblogs.microsoft.com/oldnewthing/20050523-05/?p=35573)
    reset();
    break;
  }
  return TRUE;
}
#endif
