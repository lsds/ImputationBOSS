#define CATCH_CONFIG_RUNNER
#include "../Source/BOSS.hpp"
#include "../Source/BootstrapEngine.hpp"
#include "../Source/ExpressionUtilities.hpp"
#include <arrow/array.h>
#include <arrow/builder.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <numeric>
#include <variant>
using boss::Expression;
using std::string;
using std::literals::string_literals::operator""s;
using boss::utilities::operator""_;
using Catch::Generators::random;
using Catch::Generators::take;
using Catch::Generators::values;
using Catch::literals::operator""_a;
using std::vector;
using namespace Catch::Matchers;
using boss::expressions::generic::get;
using boss::expressions::generic::get_if;
using boss::expressions::generic::holds_alternative;

static std::vector<string>
    librariesToTest{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static int defaultBatchSize = 1000;
static int TpchDatasetSizeMb = 1;
static int TpchImputedDatasetSizeMb = 1;

TEST_CASE("Expressions", "[expressions]") {
  auto v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto v2 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto const& e = "UnevaluatedPlus"_(v1, v2);
  CHECK(e.getHead().getName() == "UnevaluatedPlus");
  CHECK(e.getArguments().at(0) == v1);
  CHECK(e.getArguments().at(1) == v2);
}

TEST_CASE("Expression Transformation", "[expressions]") {
  auto v1 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto v2 = GENERATE(take(3, random<std::int64_t>(1, 100)));
  auto e = "UnevaluatedPlus"_(v1, v2);
  REQUIRE(*begin(e.getArguments()) == v1);
  get<std::int64_t>(*begin(e.getArguments()))++;
  REQUIRE(*begin(e.getArguments()) == v1 + 1);
  std::transform(std::make_move_iterator(begin(e.getArguments())),
                 std::make_move_iterator(end(e.getArguments())), e.getArguments().begin(),
                 [](auto&& e) { return get<std::int64_t>(e) + 1; });

  CHECK(e.getArguments().at(0) == v1 + 2);
  CHECK(e.getArguments().at(1) == v2 + 1);
}

TEST_CASE("Expression without arguments", "[expressions]") {
  auto const& e = "UnevaluatedPlus"_();
  CHECK(e.getHead().getName() == "UnevaluatedPlus");
}

class DummyAtom {
public:
  friend std::ostream& operator<<(std::ostream& s, DummyAtom const& /*unused*/) {
    return s << "dummy";
  }
};

TEST_CASE("Expression cast to more general expression system", "[expressions]") {
  auto a = boss::ExtensibleExpressionSystem<>::Expression("howdie"_());
  auto b = (boss::ExtensibleExpressionSystem<DummyAtom>::Expression)std::move(a);
  CHECK(
      get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b).getHead().getName() ==
      "howdie");
}

TEST_CASE("Complex expression's argument cast to more general expression system", "[expressions]") {
  auto a = "List"_("howdie"_(1, 2, 3));
  auto const& b1 =
      (boss::ExtensibleExpressionSystem<DummyAtom>::Expression)(std::move(a).getArgument(0));
  CHECK(
      get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b1).getHead().getName() ==
      "howdie");
  auto b2 =
      get<boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression>(b1).cloneArgument(1);
  CHECK(get<int64_t>(b2) == 2);
}

TEST_CASE("Extract typed arguments from complex expression (using std::accumulate)",
          "[expressions]") {
  auto exprBase = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& expr0 =
      boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression(std::move(exprBase));
  auto str = [](auto const& expr) {
    auto const& args = expr.getArguments();
    return std::accumulate(
        args.begin(), args.end(), expr.getHead().getName(),
        [](auto const& accStr, auto const& arg) {
          return accStr + "_" +
                 visit(boss::utilities::overload(
                           [](auto const& value) { return std::to_string(value); },
                           [](DummyAtom const& /*value*/) { return ""s; },
                           [](boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression const&
                                  expr) { return expr.getHead().getName(); },
                           [](boss::Symbol const& symbol) { return symbol.getName(); },
                           [](std::string const& str) { return str; }),
                       arg);
        });
  }(expr0);
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Extract typed arguments from complex expression (manual iteration)", "[expressions]") {
  auto exprBase = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& expr0 =
      boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression(std::move(exprBase));
  auto str = [](auto const& expr) {
    auto const& args = expr.getArguments();
    auto size = args.size();
    auto accStr = expr.getHead().getName();
    for(int idx = 0; idx < size; ++idx) {
      accStr +=
          "_" +
          visit(boss::utilities::overload(
                    [](auto const& value) { return std::to_string(value); },
                    [](DummyAtom const& /*value*/) { return ""s; },
                    [](boss::ExtensibleExpressionSystem<DummyAtom>::ComplexExpression const& expr) {
                      return expr.getHead().getName();
                    },
                    [](boss::Symbol const& symbol) { return symbol.getName(); },
                    [](std::string const& str) { return str; }),
                args.at(idx));
    }
    return accStr;
  }(expr0);
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Merge two complex expressions", "[expressions]") {
  auto delimeters = "List"_("_"_(), "_"_(), "_"_(), "_"_());
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto delimetersIt = std::make_move_iterator(delimeters.getArguments().begin());
  auto delimetersItEnd = std::make_move_iterator(delimeters.getArguments().end());
  auto exprIt = std::make_move_iterator(expr.getArguments().begin());
  auto exprItEnd = std::make_move_iterator(expr.getArguments().end());
  auto args = boss::ExpressionArguments();
  for(; delimetersIt != delimetersItEnd && exprIt != exprItEnd; ++delimetersIt, ++exprIt) {
    args.emplace_back(std::move(*delimetersIt));
    args.emplace_back(std::move(*exprIt));
  }
  auto e = boss::ComplexExpression("List"_, std::move(args));
  auto str = std::accumulate(
      e.getArguments().begin(), e.getArguments().end(), e.getHead().getName(),
      [](auto const& accStr, auto const& arg) {
        return accStr + visit(boss::utilities::overload(
                                  [](auto const& value) { return std::to_string(value); },
                                  [](boss::ComplexExpression const& expr) {
                                    return expr.getHead().getName();
                                  },
                                  [](boss::Symbol const& symbol) { return symbol.getName(); },
                                  [](std::string const& str) { return str; }),
                              arg);
      });
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("Merge a static and a dynamic complex expressions", "[expressions]") {
  auto delimeters = "List"_("_"s, "_"s, "_"s, "_"s);
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto delimetersIt = std::make_move_iterator(delimeters.getArguments().begin());
  auto delimetersItEnd = std::make_move_iterator(delimeters.getArguments().end());
  auto exprIt = std::make_move_iterator(expr.getArguments().begin());
  auto exprItEnd = std::make_move_iterator(expr.getArguments().end());
  auto args = boss::ExpressionArguments();
  for(; delimetersIt != delimetersItEnd && exprIt != exprItEnd; ++delimetersIt, ++exprIt) {
    args.emplace_back(std::move(*delimetersIt));
    args.emplace_back(std::move(*exprIt));
  }
  auto e = boss::ComplexExpression("List"_, std::move(args));
  auto str = std::accumulate(
      e.getArguments().begin(), e.getArguments().end(), e.getHead().getName(),
      [](auto const& accStr, auto const& arg) {
        return accStr + visit(boss::utilities::overload(
                                  [](auto const& value) { return std::to_string(value); },
                                  [](boss::ComplexExpression const& expr) {
                                    return expr.getHead().getName();
                                  },
                                  [](boss::Symbol const& symbol) { return symbol.getName(); },
                                  [](std::string const& str) { return str; }),
                              arg);
      });
  CHECK(str == "List_howdie_1_unknown_hello world");
}

TEST_CASE("holds_alternative for complex expression's arguments", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  CHECK(holds_alternative<boss::ComplexExpression>(expr.getArguments().at(0)));
  CHECK(holds_alternative<int64_t>(expr.getArguments().at(1)));
  CHECK(holds_alternative<boss::Symbol>(expr.getArguments().at(2)));
  CHECK(holds_alternative<std::string>(expr.getArguments().at(3)));
}

TEST_CASE("get_if for complex expression's arguments", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto const& arg0 = expr.getArguments().at(0);
  auto const& arg1 = expr.getArguments().at(1);
  auto const& arg2 = expr.getArguments().at(2);
  auto const& arg3 = expr.getArguments().at(3);
  CHECK(get_if<boss::ComplexExpression>(&arg0) != nullptr);
  CHECK(get_if<int64_t>(&arg1) != nullptr);
  CHECK(get_if<boss::Symbol>(&arg2) != nullptr);
  CHECK(get_if<std::string>(&arg3) != nullptr);
}

TEST_CASE("move expression's arguments to a new expression", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto&& movedExpr = std::move(expr);
  auto head = movedExpr.getHead();
  boss::ExpressionArguments args = movedExpr.getArguments();
  auto expr2 = boss::ComplexExpression(std::move(head), std::move(args)); // NOLINT
  CHECK(get<boss::ComplexExpression>(expr2.getArguments().at(0)) == "howdie"_());
  CHECK(get<int64_t>(expr2.getArguments().at(1)) == 1);
  CHECK(get<boss::Symbol>(expr2.getArguments().at(2)) == "unknown"_);
  CHECK(get<std::string>(expr2.getArguments().at(3)) == "hello world"s);
}

TEST_CASE("copy expression's arguments to a new expression", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  auto args =
      expr.getArguments(); // TODO: this one gets the reference to the arguments
                           // when it should be a copy.
                           // Any modification/move of args will be reflected in expr's arguments!
  get<int64_t>(args.at(1)) = 2;
  auto expr2 = boss::ComplexExpression(expr.getHead(), args);
  get<int64_t>(args.at(1)) = 3;
  auto expr3 = boss::ComplexExpression(expr.getHead(), std::move(args)); // NOLINT
  // CHECK(get<int64_t>(expr.getArguments().at(1)) == 1); // fails for now (see above TODO)
  CHECK(get<int64_t>(expr2.getArguments().at(1)) == 2);
  CHECK(get<int64_t>(expr3.getArguments().at(1)) == 3);
}

TEST_CASE("copy non-const expression's arguments to ExpressionArguments", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  boss::ExpressionArguments args = expr.getArguments(); // TODO: why is it moved?
  // get<int64_t>(args.at(1)) = 2;
  // auto expr2 = boss::ComplexExpression(expr.getHead(), args); // TODO: do we need to support
  // this?
  get<int64_t>(args.at(1)) = 3;
  auto expr3 = boss::ComplexExpression(expr.getHead(), std::move(args));
  // CHECK(get<int64_t>(expr.getArguments().at(1)) == 1); // fails because args was moved (see first
  // TODO) CHECK(get<int64_t>(expr2.getArguments().at(1)) == 2);
  CHECK(get<int64_t>(expr3.getArguments().at(1)) == 3);
}

TEST_CASE("copy const expression's arguments to ExpressionArguments)", "[expressions]") {
  auto const& expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  boss::ExpressionArguments args = expr.getArguments();
  // get<int64_t>(args.at(1)) = 2;
  //  auto expr2 = boss::ComplexExpression(expr.getHead(), args); // TODO: fails to compile
  get<int64_t>(args.at(1)) = 3;
  auto expr3 = boss::ComplexExpression(expr.getHead(), std::move(args));
  CHECK(get<int64_t>(expr.getArguments().at(1)) == 1);
  // CHECK(get<int64_t>(expr2.getArguments().at(1)) == 2);
  CHECK(get<int64_t>(expr3.getArguments().at(1)) == 3);
}

TEST_CASE("move and dispatch expression's arguments", "[expressions]") {
  auto expr = "List"_("howdie"_(), 1, "unknown"_, "hello world"s);
  std::vector<boss::Symbol> symbols;
  std::vector<boss::Expression> otherExpressions;
  for(auto&& arg : (boss::ExpressionArguments)std::move(expr).getArguments()) {
    visit(boss::utilities::overload(
              [&otherExpressions](auto&& value) {
                otherExpressions.emplace_back(std::forward<decltype(value)>(value));
              },
              [&symbols](boss::Symbol&& symbol) { symbols.emplace_back(std::move(symbol)); }),
          std::move(arg));
  }
  CHECK(symbols.size() == 1);
  CHECK(symbols[0] == "unknown"_);
  CHECK(otherExpressions.size() == 3);
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with numeric Spans", "[spans]", std::int64_t,
                   std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1L, 1000L))));
  auto v = vector<TestType>(input);
  auto s = boss::Span<TestType>(std::move(v));
  auto vectorExpression = "duh"_(std::move(s));
  REQUIRE(vectorExpression.getArguments().size() == input.size());
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(vectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with numeric Arrow Spans", "[spans][arrow]", std::int64_t,
                   std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1L, 1000L))));
  std::conditional_t<std::is_same_v<TestType, std::int64_t>, arrow::Int64Builder,
                     arrow::DoubleBuilder>
      builder;
  auto status = builder.AppendValues(begin(input), end(input));
  auto thingy = builder.Finish().ValueOrDie();
  auto* v = thingy->data()->template GetMutableValues<TestType>(1);
  auto s = boss::Span<TestType>(v, thingy->length(), [thingy](void* /* unused */) {});
  auto vectorExpression = "duh"_(std::move(s));
  REQUIRE(vectorExpression.getArguments().size() == input.size());
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(vectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Cloning Expressions with numeric Spans", "[spans][clone]", std::int64_t,
                   std::double_t) {
  auto input = GENERATE(take(3, chunk(5, random<TestType>(1, 1000))));
  auto vectorExpression = "duh"_(boss::Span<TestType>(vector(input)));
  auto clonedVectorExpression = vectorExpression.clone();
  for(auto i = 0U; i < input.size(); i++) {
    CHECK(clonedVectorExpression.getArguments().at(i) == input.at(i));
    CHECK(vectorExpression.getArguments()[i] == input[i]);
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Complex Expressions with Spans", "[spans]", std::string, boss::Symbol) {
  using std::literals::string_literals::operator""s;
  auto vals = GENERATE(take(3, chunk(5, values({"a"s, "b"s, "c"s, "d"s, "e"s, "f"s, "g"s, "h"s}))));
  auto input = vector<TestType>();
  std::transform(begin(vals), end(vals), std::back_inserter(input),
                 [](auto v) { return TestType(v); });
  auto vectorExpression = "duh"_(boss::Span<TestType>(move(input)));
  for(auto i = 0U; i < vals.size(); i++) {
    CHECK(vectorExpression.getArguments().at(0) == TestType(vals.at(0)));
    CHECK(vectorExpression.getArguments()[0] == TestType(vals[0]));
  }
}

TEST_CASE("Basics", "[basics]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](boss::Expression&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  SECTION("CatchingErrors") {
    CHECK_THROWS_MATCHES(
        engine.evaluate("EvaluateInEngines"_("List"_(9), 5)), std::bad_variant_access,
        Message("expected and actual type mismatch in expression \"9\", expected string"));
  }

  SECTION("Atomics") {
    CHECK(get<std::int64_t>(eval(boss::Expression(9))) == 9); // NOLINT
  }

  SECTION("Addition") {
    CHECK(get<std::int64_t>(eval("Plus"_(5, 4))) == 9); // NOLINT
    CHECK(get<std::int64_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int64_t>(eval("Plus"_(5, 2, 2))) == 9);
    CHECK(get<std::int64_t>(eval("Plus"_("Plus"_(2, 3), 2, 2))) == 9);
    CHECK(get<std::int64_t>(eval("Plus"_("Plus"_(3, 2), 2, 2))) == 9);
  }

  SECTION("Strings") {
    CHECK(get<string>(eval("StringJoin"_((string) "howdie", (string) " ", (string) "world"))) ==
          "howdie world");
  }

  SECTION("Doubles") {
    auto const twoAndAHalf = 2.5F;
    auto const two = 2.0F;
    auto const quantum = 0.001F;
    CHECK(std::fabs(get<double>(eval("Plus"_(twoAndAHalf, twoAndAHalf))) - two * twoAndAHalf) <
          quantum);
  }

  SECTION("Booleans") {
    CHECK(get<bool>(eval("Greater"_(5, 2))));
    CHECK(!get<bool>(eval("Greater"_(2, 5))));
  }

  SECTION("Symbols") {
    CHECK(get<boss::Symbol>(eval("Symbol"_((string) "x"))).getName() == "x");
    auto expression = get<boss::ComplexExpression>(
        eval("UndefinedFunction"_(9))); // NOLINT(readability-magic-numbers)

    CHECK(expression.getHead().getName() == "UndefinedFunction");
    CHECK(get<std::int64_t>(expression.getArguments().at(0)) == 9);

    CHECK(get<std::string>(
              get<boss::ComplexExpression>(eval("UndefinedFunction"_((string) "Hello World!")))
                  .getArguments()
                  .at(0)) == "Hello World!");

    eval("Set"_("TestSymbol"_, 10));
    CHECK(get<int64_t>(eval("TestSymbol"_)) == 10);
    eval("Set"_("TestSymbol"_, 20));
    CHECK(get<int64_t>(eval("TestSymbol"_)) == 20);
  }

  SECTION("Evaluate") {
    /*eval("CreateTable"_("EvaluateTable"_, "x"_, "y"_));
    eval("InsertInto"_("EvaluateTable"_, 1, 1));
    eval("InsertInto"_("EvaluateTable"_, 1, "Unevaluated"_("Interpolate"_(1, "Plus"_(1, 0)))));
    eval("InsertInto"_("EvaluateTable"_, 1, "Unevaluated"_("Interpolate"_(2, "Plus"_(0, 1)))));

    // all columns

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_)) ==
          "List"_("List"_(1, 1), "List"_(1, 2), "List"_(1, 3)));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))))) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))))) ==
          "List"_("List"_(1), "List"_(2), "List"_(3)));

    // specify the column

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_, "y"_)) ==
          "List"_("List"_(1, 1), "List"_(1, 2), "List"_(1, 3)));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))), "y"_)) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))), "y"_)) ==
          "List"_("List"_(1), "List"_(2), "List"_(3)));

    // specify wrong column

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_, "x"_)) ==
          "List"_("List"_(1, 1), "List"_(1, "Interpolate"_(1, "Plus"_(1, 0))),
                  "List"_(1, "Interpolate"_(2, "Plus"_(0, 1)))));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))), "x"_)) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))), "x"_)) ==
          "List"_("List"_(1), "List"_("Interpolate"_(1, "Plus"_(1, 0))),
                  "List"_("Interpolate"_(2, "Plus"_(0, 1)))));

    // project without evaluation needed
    /*CHECK(eval("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))) == "List"_("List"_(1),
    "List"_(1)));
    // implicitely evaluate the projected column
    CHECK(eval("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))) == "List"_("List"_(2),
    "List"_(3)));

    // example with Project using Function
    //CHECK(eval("Project"_("EvaluateTable"_,
    //                      "Function"_("tuple"_, "List"_("Column"_("tuple"_, 1))))) ==
    //      "List"_("List"_(1), "List"_(1)));
    //CHECK(eval("Project"_("EvaluateTable"_,
    //                      "Function"_("tuple"_, "List"_("Column"_("tuple"_, 2))))) ==
    //      "List"_("List"_(2), "List"_(3)));

    // grouping without evaluation needed
    CHECK(eval("Group"_("EvaluateTable"_, "Sum"_("x"_))) == "List"_("List"_(2)));
    // evaluate the grouping aggregate
    CHECK(eval("Group"_("EvaluateTable"_, "Sum"_("y"_))) == "List"_("List"_(5)));
    // evaluate the grouping keys
    CHECK(eval("Group"_("EvaluateTable"_, "By"_("y"_), "Count"_)) ==
          "List"_("List"_(2, 1), "List"_(3, 1)));

    // evaluate the selected rows
    CHECK(eval("Select"_("EvaluateTable"_, "Where"_("Greater"_("x"_, 0)))) ==
          "List"_("List"_(1, 2), "List"_(1, 3)));
    // evaluate the selection predicate
    CHECK(eval("Select"_("EvaluateTable"_, "Where"_("Greater"_("y"_, 2)))) ==
          "List"_("List"_(1, 3)));

    // evaluate the joined rows
    CHECK(eval("Join"_("EvaluateTable"_, "EvaluateTable"_, "Where"_("Equal"_("x"_, "x"_)))) ==
          "List"_("List"_(1, 2, 1, 2), "List"_(1, 3, 1, 2), "List"_(1, 2, 1, 3),
                  "List"_(1, 3, 1, 3)));
    // evaluate the join predicate
    CHECK(eval("Join"_("EvaluateTable"_, "EvaluateTable"_, "Where"_("Equal"_("y"_, "y"_)))) ==
          "List"_("List"_(1, 2, 1, 2), "List"_(1, 3, 1, 3)));

    // evaluate the sorted rows
    CHECK(eval("Sort"_("EvaluateTable"_, "By"_("x"_))) == "List"_("List"_(1, 2), "List"_(1, 3)));
    // evaluate the sorting keys
    CHECK(eval("Sort"_("EvaluateTable"_, "By"_("y"_))) == "List"_("List"_(1, 2), "List"_(1, 3)));
    // evaluate the sorting keys wrapping them with a more complex expression
    CHECK(eval("Sort"_("EvaluateTable"_, "By"_("Minus"_("y"_)))) ==
          "List"_("List"_(1, 3), "List"_(1, 2)));*/
  }

  SECTION("evaluate noargs") {
    /*eval("CreateTable"_("EvaluateTable"_, "x"_, "y"_));
    eval("InsertInto"_("EvaluateTable"_, 1, 1));
    eval("InsertInto"_("EvaluateTable"_, 1, "Interpolate"_()));
    eval("InsertInto"_("EvaluateTable"_, 1, "Interpolate"_()));

    // all columns

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_)) ==
          "List"_("List"_(1, 1), "List"_(1, 10), "List"_(1, 10)));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))))) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))))) ==
          "List"_("List"_(1), "List"_(10), "List"_(10)));

    // specify the column

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_, "y"_)) ==
          "List"_("List"_(1, 1), "List"_(1, 10), "List"_(1, 10)));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))), "y"_)) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))), "y"_)) ==
          "List"_("List"_(1), "List"_(10), "List"_(10)));

    // specify wrong column

    // one clean column, one dirty column
    CHECK(eval("Evaluate"_("EvaluateTable"_, "x"_)) ==
          "List"_("List"_(1, 1), "List"_(1, "Interpolate"_()), "List"_(1, "Interpolate"_())));
    // only clean column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("x"_, "x"_))), "x"_)) ==
          "List"_("List"_(1), "List"_(1), "List"_(1)));
    // only dirty column
    CHECK(eval("Evaluate"_(("Project"_("EvaluateTable"_, "As"_("y"_, "y"_))), "x"_)) ==
          "List"_("List"_(1), "List"_("Interpolate"_()), "List"_("Interpolate"_())));*/
  }

  /*SECTION("Interpolation") {
    auto thing = GENERATE(
        take(1, chunk(3, filter([](int i) { return i % 2 == 1; }, random(1, 1000))))); // NOLINT
    std::sort(begin(thing), end(thing));
    auto y = GENERATE(
        take(1, chunk(3, filter([](int i) { return i % 2 == 1; }, random(1, 1000))))); // NOLINT

    eval("CreateTable"_("InterpolationTable"_, "x"_, "y"_));
    eval("InsertInto"_("InterpolationTable"_, thing[0], y[0]));
    eval("InsertInto"_("InterpolationTable"_, thing[1], "Interpolate"_("x"_)));
    eval("InsertInto"_("InterpolationTable"_, thing[2], y[2]));
    REQUIRE(eval("Project"_("InterpolationTable"_, "As"_("y"_, "y"_))) ==
            "List"_("List"_(y[0]), "List"_((y[0] + y[2]) / 2), "List"_(y[2])));
    REQUIRE(eval("Project"_("InterpolationTable"_, "As"_("x"_, "x"_))) ==
            "List"_("List"_(thing[0]), "List"_(thing[1]), "List"_(thing[2])));
  }*/

  /*SECTION("Relational on ephemeral tables") {

    SECTION("Selection") {
      auto const& result =
          eval("Select"_("ScanColumns"_("Column"_("Size"_, "List"_(2, 3, 1, 4, 1))),
                         "Where"_("Greater"_("Size"_, 3))));
      REQUIRE(result == "List"_("List"_(4)));
    }
  }*/

  SECTION("Relational (simple)") {
    eval("CreateTable"_("Customer"_, "FirstName"_, "LastName"_));
    eval("InsertInto"_("Customer"_, "John", "McCarthy"));
    eval("InsertInto"_("Customer"_, "Sam", "Madden"));
    eval("InsertInto"_("Customer"_, "Barbara", "Liskov"));
    SECTION("Selection") {
      auto const& sam = eval(
          "Select"_("Customer"_,
                    "Function"_("tuple"_, "StringContainsQ"_("Madden", "Column"_("tuple"_, 2)))));
      REQUIRE(sam == "List"_("List"_("Sam", "Madden")));
      REQUIRE(sam != "List"_("List"_("Barbara", "Liskov")));
    }

    SECTION("Aggregation") {
      REQUIRE(eval("Group"_("Customer"_, "Function"_(0), "Count"_)) == "List"_("List"_(0, 3)));
      REQUIRE(eval("Group"_("Customer"_, "Count"_)) == "List"_("List"_(3)));
      REQUIRE(
          eval("Group"_(("Select"_("Customer"_,
                                   "Function"_("tuple"_, "StringContainsQ"_(
                                                             "Madden", "Column"_("tuple"_, 2))))),
                        "Function"_(0), "Count"_)) == "List"_("List"_(0, 1)));
    }

    SECTION("Join") {
      eval("CreateTable"_("Adjacency1"_, "From"_, "To"_));
      eval("CreateTable"_("Adjacency2"_, "From2"_, "To2"_));
      auto const dataSetSize = 10;
      for(int i = 0U; i < dataSetSize; i++) {
        eval("InsertInto"_("Adjacency1"_, i, dataSetSize + i));
        eval("InsertInto"_("Adjacency2"_, dataSetSize + i, i));
      }
      auto const& result =
          eval("Join"_("Adjacency1"_, "Adjacency2"_,
                       "Function"_("List"_("tuple"_),
                                   "Equal"_("Column"_("tuple"_, 2), "Column"_("tuple"_, 1)))));
      INFO(get<boss::ComplexExpression>(result));
      REQUIRE(get<boss::ComplexExpression>(result).getArguments().size() == dataSetSize);
    }
  }

  SECTION("Inserting") {
    eval("CreateTable"_("InsertTable"_, "duh"_));
    eval("InsertInto"_("InsertTable"_, "Plus"_(1, 2)));
    REQUIRE(eval("Select"_("InsertTable"_, "Function"_(true))) == "List"_("List"_(3)));
  }

  SECTION("Relational (with multiple column types)") {
    eval("CreateTable"_("Customer"_, "ID"_, "FirstName"_, "LastName"_, "BirthYear"_, "Country"_));
    INFO(eval("Length"_("Select"_("Customer"_, "Function"_(true)))));

    REQUIRE(get<std::int64_t>(eval("Length"_("Select"_("Customer"_, "Function"_(true))))) == 0);
    auto const& emptyTable = eval("Select"_("Customer"_, "Function"_(true)));
    CHECK(get<std::int64_t>(eval("Length"_(emptyTable))) == 0);
    eval("InsertInto"_("Customer"_, 1, "John", "McCarthy", 1927, "USA"));  // NOLINT
    eval("InsertInto"_("Customer"_, 2, "Sam", "Madden", 1976, "USA"));     // NOLINT
    eval("InsertInto"_("Customer"_, 3, "Barbara", "Liskov", 1939, "USA")); // NOLINT
    INFO("Select"_("Customer"_, "Function"_(true)));
    CHECK(eval("Length"_("Select"_("Customer"_, "Function"_(true)))) == Expression(3));
    auto const& fullTable = eval("Select"_("Customer"_, "Function"_(true)));
    CHECK(get<std::int64_t>(eval("Length"_(fullTable))) == 3);
    CHECK(get<std::string>(eval("Extract"_("Extract"_("Select"_("Customer"_, "Function"_(true)), 2),
                                           3))) == "Madden");

    SECTION("Selection") {
      auto const& sam = eval("Select"_(
          "Customer"_,
          "Function"_("List"_("tuple"_), "StringContainsQ"_("Madden", "Column"_("tuple"_, 3)))));
      CHECK(get<std::int64_t>(eval("Length"_(sam))) == 1);
      auto const& samRow = eval("Extract"_(sam, 1));
      CHECK(get<std::int64_t>(eval("Length"_(samRow))) == 5);
      CHECK(get<string>(eval("Extract"_(samRow, 2))) == "Sam");
      CHECK(get<string>(eval("Extract"_(samRow, 3))) == "Madden");
      auto const& none = eval("Select"_("Customer"_, "Function"_(false)));
      CHECK(get<std::int64_t>(eval("Length"_(none))) == 0);
      auto const& all = eval("Select"_("Customer"_, "Function"_(true)));
      CHECK(get<std::int64_t>(eval("Length"_(all))) == 3);
      auto const& johnRow = eval("Extract"_(all, 1));
      auto const& barbaraRow = eval("Extract"_(all, 3));
      CHECK(get<string>(eval("Extract"_(johnRow, 2))) == "John");
      CHECK(get<string>(eval("Extract"_(barbaraRow, 2))) == "Barbara");
    }

    SECTION("Projection") {
      auto const& fullnames = eval(
          "Project"_("Customer"_, "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
      INFO("Project"_("Customer"_, "As"_("FirstName"_, "FirstName"_, "LastName"_, "LastName"_)));
      INFO(fullnames);
      CHECK(get<std::int64_t>(eval("Length"_(fullnames))) == 3);
      auto const& firstNames = eval("Project"_("Customer"_, "As"_("FirstName"_, "FirstName"_)));
      INFO(eval("Extract"_("Extract"_(fullnames, 1), 1)));
      CHECK(get<string>(eval("Extract"_("Extract"_(firstNames, 1), 1))) ==
            get<string>(eval("Extract"_("Extract"_(fullnames, 1), 1))));
      auto const& lastNames = eval("Project"_("Customer"_, "As"_("LastName"_, "LastName"_)));
      INFO("lastnames=" << eval("Extract"_("Extract"_(lastNames, 1), 1)));
      INFO("fullnames=" << eval("Extract"_("Extract"_(fullnames, 1), 2)));
      CHECK(get<string>(eval("Extract"_("Extract"_(lastNames, 1), 1))) ==
            get<string>(eval("Extract"_("Extract"_(fullnames, 1), 2))));
    }

    SECTION("Sorting") {
      auto const& sortedByLastName = eval("SortBy"_("Select"_("Customer"_, "Function"_(true)),
                                                    "Function"_("tuple"_, "Column"_("tuple"_, 3))));
      auto const& liskovRow = eval("Extract"_(sortedByLastName, 1));
      auto const& MaddenRow = eval("Extract"_(sortedByLastName, 2));
      INFO(sortedByLastName);
      CHECK(get<string>(eval("Extract"_(liskovRow, 3))) == "Liskov");
      CHECK(get<string>(eval("Extract"_(MaddenRow, 3))) == "Madden");
    }

    SECTION("Aggregation") {
      auto const& countRows = eval("Group"_("Customer"_, "Function"_(0), "Count"_));
      INFO("countRows=" << countRows << "\n" << eval("Extract"_("Extract"_(countRows, 1))));
      CHECK(get<std::int64_t>(eval("Extract"_("Extract"_(countRows, 1), 2))) == 3);
      CHECK(get<std::int64_t>(eval("Extract"_(
                "Extract"_("Group"_(("Select"_("Customer"_, "Where"_("StringContainsQ"_(
                                                                "Madden", "LastName"_)))),
                                    "Count"_),
                           1),
                1))) == 1);

      // auto const& sumRows = eval("Group"_("Customer"_, "Function"_(0), "Sum"_("BirthYear"_)));
      // CHECK(get<std::int64_t>(eval("Extract"_("Extract"_(sumRows, 1), 2))) == (1927 + 1976 +
      // 1939));
    }
  }
}

TEST_CASE("Arrays", "[arrays]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  namespace nasty = boss::utilities::nasty;
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), expression));
  };

  std::vector<int64_t> ints{10, 20, 30, 40, 50}; // NOLINT
  std::shared_ptr<arrow::Array> intArrayPtr(
      new arrow::Int64Array((long long)ints.size(), arrow::Buffer::Wrap(ints)));

  std::vector<int32_t> smallints{10, 20, 30, 40, 50}; // NOLINT
  std::shared_ptr<arrow::Array> smallIntArrayPtr(
      new arrow::Int32Array((long long)smallints.size(), arrow::Buffer::Wrap(smallints)));

  std::vector<float_t> floats{10.f, 20.f, 30.f, 40.f, 50.f}; // NOLINT
  std::shared_ptr<arrow::Array> floatArrayPtr(
      new arrow::FloatArray((long long)floats.size(), arrow::Buffer::Wrap(floats)));

  std::vector<double_t> doubles{10.0, 20.0, 30.0, 40.0, 50.0}; // NOLINT
  std::shared_ptr<arrow::Array> doubleArrayPtr(
      new arrow::DoubleArray((long long)doubles.size(), arrow::Buffer::Wrap(doubles)));

  auto intArrayPtrExpr = nasty::arrowArrayToExpression(intArrayPtr);
  auto smallIntArrayPtrExpr = nasty::arrowArrayToExpression(smallIntArrayPtr);
  auto floatArrayPtrExpr = nasty::arrowArrayToExpression(floatArrayPtr);
  auto doubleArrayPtrExpr = nasty::arrowArrayToExpression(doubleArrayPtr);

  eval("CreateTable"_("Thingy"_, "Value"_, "Value32"_, "ValueFloat"_, "ValueDouble"_));
  eval("AttachColumns"_("Thingy"_, intArrayPtrExpr, smallIntArrayPtrExpr, floatArrayPtrExpr,
                        doubleArrayPtrExpr));

  SECTION("ArrowArrays") {
    CHECK(get<std::int64_t>(eval("Extract"_(intArrayPtrExpr, 1))) == 10);
    CHECK(get<std::int64_t>(eval("Extract"_(intArrayPtrExpr, 2))) == 20);
    CHECK(get<std::int64_t>(eval("Extract"_(intArrayPtrExpr, 3))) == 30);
    CHECK(get<std::int64_t>(eval("Extract"_(intArrayPtrExpr, 4))) == 40);
    CHECK(get<std::int64_t>(eval("Extract"_(intArrayPtrExpr, 5))) == 50);
    CHECK(eval(intArrayPtrExpr) == "List"_(10, 20, 30, 40, 50));
    CHECK(eval(smallIntArrayPtrExpr) == "List"_(10, 20, 30, 40, 50));
    CHECK(eval(floatArrayPtrExpr) == "List"_(10.f, 20.f, 30.f, 40.f, 50.f));
    CHECK(eval(doubleArrayPtrExpr) == "List"_(10.0, 20.0, 30.0, 40.0, 50.0));
  }

  auto compareColumn = [&eval](boss::Expression const& expression, auto const& results) {
    for(auto i = 0; i < results.size(); i++) {
      CHECK(get<typename std::remove_reference_t<decltype(results)>::value_type>(
                eval("Extract"_("Extract"_(expression, i + 1), 1))) == results[i]);
    }
  };

  SECTION("Plus") {
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value"_, "Value"_))),
                  vector<std::int64_t>{20, 40, 60, 80, 100}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value32"_, "Value32"_))),
                  vector<std::int64_t>{20, 40, 60, 80, 100}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("ValueFloat"_, "ValueFloat"_))),
                  vector<std::double_t>{20, 40, 60, 80, 100}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("ValueDouble"_, "ValueDouble"_))),
                  vector<std::double_t>{20, 40, 60, 80, 100}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value"_, 1))),
                  vector<std::int64_t>{11, 21, 31, 41, 51}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value32"_, "Int32"_(1)))),
                  vector<std::int64_t>{11, 21, 31, 41, 51}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value"_, "Float"_(1.0)))),
                  vector<std::double_t>{11, 21, 31, 41, 51}); // NOLINT(readability-magic-numbers)
    compareColumn("Project"_("Thingy"_, "As"_("Result"_, "Plus"_("Value"_, 1.0))),
                  vector<std::double_t>{11, 21, 31, 41, 51}); // NOLINT(readability-magic-numbers)
  }

  SECTION("Greater") {
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_("Value"_, 25))), // NOLINT(readability-magic-numbers)
        vector<bool>{false, false, true, true, true});
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_("Value32"_,
                                               "Int32"_(25)))), // NOLINT(readability-magic-numbers)
        vector<bool>{false, false, true, true, true});
    compareColumn(
        "Project"_(
            "Thingy"_,
            "As"_("Result"_, "Greater"_("ValueFloat"_,
                                        "Float"_(25.0)))), // NOLINT(readability-magic-numbers)
        vector<bool>{false, false, true, true, true});
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_("ValueDouble"_,
                                               25.0))), // NOLINT(readability-magic-numbers)
        vector<bool>{false, false, true, true, true});
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_(45, "Value"_))), // NOLINT(readability-magic-numbers)
        vector<bool>{true, true, true, true, false});
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_("Int32"_(45), // NOLINT(readability-magic-numbers)
                                               "Value32"_))),
        vector<bool>{true, true, true, true, false});
    compareColumn(
        "Project"_("Thingy"_,
                   "As"_("Result"_, "Greater"_("Float"_(45.0), // NOLINT(readability-magic-numbers)
                                               "ValueFloat"_))),
        vector<bool>{true, true, true, true, false});
    compareColumn(
        "Project"_("Thingy"_, "As"_("Result"_, "Greater"_(45.0, // NOLINT(readability-magic-numbers)
                                                          "ValueDouble"_))),
        vector<bool>{true, true, true, true, false});
  }

  SECTION("Logic") {
    compareColumn("Project"_("Thingy"_,
                             "As"_("Result"_, "Not"_("Greater"_(
                                                  "Value"_, 25) // NOLINT(readability-magic-numbers)
                                                     ))),
                  vector<bool>{true, true, false, false, false});

    compareColumn(
        "Project"_(
            "Thingy"_,
            "As"_("Result"_, "And"_("Greater"_("Value"_, 25), // NOLINT(readability-magic-numbers)
                                    "Greater"_(45, "Value"_)  // NOLINT(readability-magic-numbers)
                                    ))),
        vector<bool>{false, false, true, true, false});
  }
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("Summation of numeric Spans", "[spans]", std::int64_t, std::double_t) {
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  auto input = GENERATE(take(3, chunk(20, random<TestType>(1, 1000))));
  auto sum = std::accumulate(begin(input), end(input), TestType());

  if constexpr(std::is_same_v<TestType, std::double_t>) {
    CHECK(get<std::double_t>(eval("Plus"_(boss::Span<TestType>(vector(input))))) ==
          Catch::Detail::Approx((std::double_t)sum));
  } else {
    CHECK(get<TestType>(eval("Plus"_(boss::Span<TestType>(vector(input))))) == sum);
  }
}

TEST_CASE("Imputation (interpolation)", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("EnableOrderPreservationCache"_, true));

  eval("CreateTable"_("Customer"_, "Name"_, "Age"_));
  eval("InsertInto"_("Customer"_, "John", 32));    // NOLINT
  eval("InsertInto"_("Customer"_, "Sam", 42));     // NOLINT
  eval("InsertInto"_("Customer"_, "Barbara", 37)); // NOLINT

  eval("InsertInto"_("Customer"_, "Holger",
                     "Unevaluated"_("Interpolate"_("Customer"_, "Age"_)))); // NOLINT
  eval("InsertInto"_("Customer"_, "Andrea",
                     "Unevaluated"_("Interpolate"_("Customer"_, "Age"_)))); // NOLINT

  eval("InsertInto"_("Customer"_, "George", 28)); // NOLINT

  CHECK(eval("Column"_("Partition"_("Customer"_, 2), 1)) == "List"_("Holger", "Andrea"));

  auto const query = "Evaluate"_("Project"_("Customer"_, "As"_("Age"_, "Age"_)));

  INFO(eval(query))
  CHECK(eval(query) ==
        "List"_("List"_(32), "List"_(42), "List"_(37), "List"_(28), "List"_(34), "List"_(31)));
}

TEST_CASE("Imputation (hot deck)", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("CreateTable"_("Customer"_, "Name"_, "Age"_));
  eval("InsertInto"_("Customer"_, "John", 32));    // NOLINT
  eval("InsertInto"_("Customer"_, "Sam", 42));     // NOLINT
  eval("InsertInto"_("Customer"_, "Barbara", 37)); // NOLINT

  eval("InsertInto"_("Customer"_, "Holger",
                     "Unevaluated"_("HotDeck"_("Customer"_, "Age"_)))); // NOLINT
  eval("InsertInto"_("Customer"_, "Andrea",
                     "Unevaluated"_("HotDeck"_("Customer"_, "Age"_)))); // NOLINT

  eval("InsertInto"_("Customer"_, "George", 28)); // NOLINT

  CHECK(eval("Column"_("Partition"_("Customer"_, 2), 1)) == "List"_("Holger", "Andrea"));

  auto const query = "Evaluate"_("Project"_("Customer"_, "As"_("Age"_, "Age"_)));

  INFO(eval(query));

  auto imputed1 = get<int64_t>(eval("Extract"_("Extract"_(query, 5), 1)));
  auto imputed2 = get<int64_t>(eval("Extract"_("Extract"_(query, 6), 1)));

  CHECK((imputed1 == 32 || imputed1 == 42 || imputed1 == 37 || imputed1 == 28));
  CHECK((imputed2 == 32 || imputed2 == 42 || imputed2 == 37 || imputed2 == 28));
}

TEST_CASE("Imputation (approximate mean)", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("CreateTable"_("Customer"_, "Name"_, "Age"_));
  eval("InsertInto"_("Customer"_, "John", 32));    // NOLINT
  eval("InsertInto"_("Customer"_, "Sam", 42));     // NOLINT
  eval("InsertInto"_("Customer"_, "Barbara", 37)); // NOLINT

  eval("InsertInto"_("Customer"_, "Holger",
                     "Unevaluated"_("ApproxMean"_("Customer"_, "Age"_)))); // NOLINT
  eval("InsertInto"_("Customer"_, "Andrea",
                     "Unevaluated"_("ApproxMean"_("Customer"_, "Age"_)))); // NOLINT

  eval("InsertInto"_("Customer"_, "George", 28)); // NOLINT

  CHECK(eval("Column"_("Partition"_("Customer"_, 2), 1)) == "List"_("Holger", "Andrea"));

  auto const query = "Evaluate"_("Project"_("Customer"_, "As"_("Age"_, "Age"_)));

  INFO(eval(query))
  CHECK(eval(query) ==
        "List"_("List"_(32), "List"_(42), "List"_(37), "List"_(28), "List"_(34), "List"_(34)));
}

TEST_CASE("Imputation (decision tree)", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("CreateTable"_("Customer"_, "Id"_, "Age"_, "Height"_));
  eval("InsertInto"_("Customer"_, 1, 32, 2.1));  // NOLINT
  eval("InsertInto"_("Customer"_, 2, 42, 1.65)); // NOLINT
  eval("InsertInto"_("Customer"_, 3, 37, 1.85)); // NOLINT

  eval("InsertInto"_("Customer"_, 4, "Unevaluated"_("DecisionTree"_("Customer"_, "Age"_)),
                     1.9)); // NOLINT
  eval("InsertInto"_("Customer"_, 5, "Unevaluated"_("DecisionTree"_("Customer"_, "Age"_)),
                     1.7)); // NOLINT

  eval("InsertInto"_("Customer"_, 6, 28, 1.75)); // NOLINT

  CHECK(eval("Column"_("Partition"_("Customer"_, 2), 1)) == "List"_(4, 5));

  auto const query =
      "Evaluate"_("Project"_("Customer"_, "As"_("Age"_, "Age"_, "Height"_, "Height"_)));

  INFO(eval(query))

  auto imputed1 = get<int64_t>(eval("Extract"_("Extract"_(query, 5), 1)));
  auto imputed2 = get<int64_t>(eval("Extract"_("Extract"_(query, 6), 1)));

  INFO(imputed1);
  INFO(imputed2);
  CHECK((imputed1 >= 28 && imputed1 <= 42));
  CHECK((imputed2 >= 28 && imputed2 <= 42));
}

TEST_CASE("TPC-H-RND", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Function"_(
      "List"_(), "Random"_("Unevaluated"_("Plus"_(10.0, 0.0)), "Unevaluated"_("Minus"_(10.0, 10.0)),
                           "Unevaluated"_("Times"_(2.0, 2.0)), "Unevaluated"_("Divide"_(10.0, 2.0)),
                           "Unevaluated"_("Plus"_(10, 5.0)), "Unevaluated"_("Plus"_(10.0, 10)),
                           "Unevaluated"_("Minus"_(10, 5.0)), "Unevaluated"_("Minus"_(10.0, 2))));
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-RND output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-RND2", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute =
      "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_(1)), "Unevaluated"_("NoOp2"_(2)),
                                       "Unevaluated"_("NoOp3"_(3)), "Unevaluated"_("NoOp4"_(4)),
                                       "Unevaluated"_("NoOp5"_(5)), "Unevaluated"_("NoOp6"_(6)),
                                       "Unevaluated"_("NoOp7"_(7)), "Unevaluated"_("NoOp8"_(8))));
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-RND2 output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-DTree", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("DecisionTree"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Evaluate"_("Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)));

  auto const queryQ6 = "Group"_(
      "Project"_("Select"_(lineitemTable, "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                                          "Greater"_("L_DISCOUNT"_, 4),
                                                          "Greater"_(6, "L_DISCOUNT"_),
                                                          "Greater"_(15776640, "L_SHIPDATE"_),
                                                          "Greater"_("L_SHIPDATE"_, 12623039)))),
                 "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-DTree output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-HotDeck", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("HotDeck"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-HotDeck output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-Mean", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("ApproxMean"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-Mean output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-Interp", "[imputation]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));
  eval("Set"_("EnableOrderPreservationCache"_, true));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("Interpolate"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Project"_(
      "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_)),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-Interp output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem0.csv");

  REQUIRE(checkSuccess(eval("Load"_(table, filepath))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable =
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 =
      "Group"_("ParallelCombine"_("Project"_(
                   "Select"_(lineitemTable, "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                                            "Greater"_("L_DISCOUNT"_, 4),
                                                            "Greater"_(6, "L_DISCOUNT"_),
                                                            "Greater"_(15776640, "L_SHIPDATE"_),
                                                            "Greater"_("L_SHIPDATE"_, 12623039)))),
                   "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
               "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-Parallel output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-RND-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Function"_(
      "List"_(), "Random"_("Unevaluated"_("Plus"_(10.0, 0.0)), "Unevaluated"_("Minus"_(10.0, 10.0)),
                           "Unevaluated"_("Times"_(2.0, 2.0)), "Unevaluated"_("Divide"_(10.0, 2.0)),
                           "Unevaluated"_("Plus"_(10, 5.0)), "Unevaluated"_("Plus"_(10.0, 10)),
                           "Unevaluated"_("Minus"_(10, 5.0)), "Unevaluated"_("Minus"_(10.0, 2))));
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable =
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "ParallelCombine"_("Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-RND-Parallel output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-RND2-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute =
      "Function"_("List"_(), "Random"_("Unevaluated"_("NoOp1"_(1)), "Unevaluated"_("NoOp2"_(2)),
                                       "Unevaluated"_("NoOp3"_(3)), "Unevaluated"_("NoOp4"_(4)),
                                       "Unevaluated"_("NoOp5"_(5)), "Unevaluated"_("NoOp6"_(6)),
                                       "Unevaluated"_("NoOp7"_(7)), "Unevaluated"_("NoOp8"_(8))));
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable =
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "ParallelCombine"_("Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-RND2-Parallel " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-DTree-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("DecisionTree"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable = "Evaluate"_(
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)));

  auto const queryQ6 =
      "Group"_("ParallelCombine"_("Project"_(
                   "Select"_(lineitemTable, "Where"_("And"_("Greater"_(24, "L_QUANTITY"_),
                                                            "Greater"_("L_DISCOUNT"_, 4),
                                                            "Greater"_(6, "L_DISCOUNT"_),
                                                            "Greater"_(15776640, "L_SHIPDATE"_),
                                                            "Greater"_("L_SHIPDATE"_, 12623039)))),
                   "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
               "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6); // eval("Extract"_("Extract"_(queryQ6_, 1), 1));
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-DTree-Parallel output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-HotDeck-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("HotDeck"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable =
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "ParallelCombine"_("Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-HotDeck-Parallel output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("TPC-H-Mean-Parallel", "[parallel]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), move(expression)));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  boss::Symbol table = "LINEITEM"_;
  auto filepath = std::string("../../data/tpch_" + std::to_string(TpchImputedDatasetSizeMb) +
                              "MB/simplified/lineitem10.csv");

  auto const impute = "Unevaluated"_("ApproxMean"_());
  REQUIRE(checkSuccess(eval("Load"_(table, filepath, impute))));

  Catch::Timer timer;
  double duration = 0.0; // NOLINT
  auto const& lineitemTable =
      "Project"_("ParallelDispatch"_("LINEITEM"_),
                 "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                       "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

  auto const queryQ6 = "Group"_(
      "ParallelCombine"_("Project"_(
          "Evaluate"_("Select"_(
              lineitemTable,
              "Where"_("And"_("Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 4),
                              "Greater"_(6, "L_DISCOUNT"_), "Greater"_(15776640, "L_SHIPDATE"_),
                              "Greater"_("L_SHIPDATE"_, 12623039))))),
          "As"_("revenue"_, "L_EXTENDEDPRICE"_))),
      "Sum"_("revenue"_));

  timer.start();
  auto output = eval(queryQ6);
  duration = timer.getElapsedSeconds();
  std::cout << "TPC-H-Mean-Parallel output " << output << " duration " << duration << std::endl;
  INFO(eval("Group"_(queryQ6, "Count"_)));
  CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
}

TEST_CASE("Handling Dates", "[date]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto&& expression) mutable {
    return engine.evaluate("EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))),
                                                std::move(expression)));
  };

  CHECK(get<bool>(eval("Greater"_("DateObject"_("1997-01-01"), "DateObject"_("1996-12-31")))) ==
        true);
  CHECK(get<bool>(eval("Greater"_("DateObject"_("1996-01-02"), "DateObject"_("1996-01-01")))) ==
        true);
  CHECK(get<bool>(eval("Greater"_("DateObject"_("2020-01-01"), "DateObject"_("1970-01-01")))) ==
        true);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1970-01-01")))) == 1970);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-01-01")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-01-02")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-01-15")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-02-01")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-03-01")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1998-12-31")))) == 1998);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1999-01-01")))) == 1999);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("1970-01-01")))) == 1970);
  CHECK(get<std::int64_t>(eval("Year"_("DateObject"_("2020-06-15")))) == 2020);
}

TEST_CASE("TPC-H", "[tpch]") { // NOLINT
  auto engine = boss::engines::BootstrapEngine();
  REQUIRE(!librariesToTest.empty());
  auto eval = [&engine](auto const& expression) mutable {
    return engine.evaluate(
        "EvaluateInEngines"_("List"_(GENERATE(from_range(librariesToTest))), expression.clone()));
  };

  eval("Set"_("MicroBatchesMaxSize"_, defaultBatchSize));

  auto checkSuccess = [](auto&& output) {
    auto* maybeComplexExpr = std::get_if<boss::ComplexExpression>(&output);
    if(maybeComplexExpr == nullptr) {
      return true;
    }
    if(maybeComplexExpr->getHead() == "ErrorWhenEvaluatingExpression"_) {
      for(auto const& arg : maybeComplexExpr->getArguments()) {
        UNSCOPED_INFO(arg);
      }
      return false;
    }
    return true;
  };

  // create schema
  REQUIRE(checkSuccess(eval("CreateTable"_("REGION"_, "R_REGIONKEY"_, "R_NAME"_, "R_COMMENT"_))));
  REQUIRE(checkSuccess(
      eval("CreateTable"_("NATION"_, "N_NATIONKEY"_, "N_NAME"_, "N_REGIONKEY"_, "N_COMMENT"_))));
  REQUIRE(checkSuccess(
      eval("CreateTable"_("PART"_, "P_PARTKEY"_, "P_NAME"_, "P_MFGR"_, "P_BRAND"_, "P_TYPE"_,
                          "P_SIZE"_, "P_CONTAINER"_, "P_RETAILPRICE"_, "P_COMMENT"_))));
  REQUIRE(
      checkSuccess(eval("CreateTable"_("SUPPLIER"_, "S_SUPPKEY"_, "S_NAME"_, "S_ADDRESS"_,
                                       "S_NATIONKEY"_, "S_PHONE"_, "S_ACCTBAL"_, "S_COMMENT"_))));
  REQUIRE(checkSuccess(eval("CreateTable"_("PARTSUPP"_, "PS_PARTKEY"_, "PS_SUPPKEY"_,
                                           "PS_AVAILQTY"_, "PS_SUPPLYCOST"_, "PS_COMMENT"_))));
  REQUIRE(checkSuccess(
      eval("CreateTable"_("CUSTOMER"_, "C_CUSTKEY"_, "C_NAME"_, "C_ADDRESS"_, "C_NATIONKEY"_,
                          "C_PHONE"_, "C_ACCTBAL"_, "C_MKTSEGMENT"_, "C_COMMENT"_))));
  REQUIRE(checkSuccess(eval("CreateTable"_("ORDERS"_, "O_ORDERKEY"_, "O_CUSTKEY"_, "O_ORDERSTATUS"_,
                                           "O_TOTALPRICE"_, "O_ORDERDATE"_, "O_ORDERPRIORITY"_,
                                           "O_CLERK"_, "O_SHIPPRIORITY"_, "O_COMMENT"_))));
  REQUIRE(checkSuccess(eval("CreateTable"_(
      "LINEITEM"_, "L_ORDERKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_LINENUMBER"_, "L_QUANTITY"_,
      "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_TAX"_, "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_SHIPDATE"_,
      "L_COMMITDATE"_, "L_RECEIPTDATE"_, "L_SHIPINSTRUCT"_, "L_SHIPMODE"_, "L_COMMENT"_))));

  // load data
  REQUIRE(checkSuccess(eval("Load"_(
      "REGION"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/region.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "NATION"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/nation.tbl"))));
  REQUIRE(checkSuccess(eval(
      "Load"_("PART"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/part.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "SUPPLIER"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/supplier.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "PARTSUPP"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/partsupp.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "CUSTOMER"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/customer.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "ORDERS"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/orders.tbl"))));
  REQUIRE(checkSuccess(eval("Load"_(
      "LINEITEM"_, "../../data/tpch_" + std::to_string(TpchDatasetSizeMb) + "MB/lineitem.tbl"))));

  SECTION("LoadCheck") {
    CHECK(eval("Group"_("REGION"_, "Count"_)) == "List"_("List"_(5)));
    CHECK(eval("Group"_("NATION"_, "Count"_)) == "List"_("List"_(25)));
    CHECK(eval("Group"_("PART"_, "Count"_)) == "List"_("List"_(200)));
    CHECK(eval("Group"_("SUPPLIER"_, "Count"_)) == "List"_("List"_(10)));
    CHECK(eval("Group"_("PARTSUPP"_, "Count"_)) == "List"_("List"_(800)));
    CHECK(eval("Group"_("CUSTOMER"_, "Count"_)) == "List"_("List"_(150)));
    CHECK(eval("Group"_("ORDERS"_, "Count"_)) == "List"_("List"_(1500)));
    CHECK(eval("Group"_("LINEITEM"_, "Count"_)) == "List"_("List"_(6005)));
  }

  SECTION("Q1") {
    auto const& lineitemTable =
        "Project"_("LINEITEM"_,
                   "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_SHIPDATE"_,
                         "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_RETURNFLAG"_,
                         "L_RETURNFLAG"_, "L_LINESTATUS"_, "L_LINESTATUS"_, "L_TAX"_, "L_TAX"_));
    auto const queryQ1 = "Sort"_(
        "Project"_(
            "Group"_(
                "Project"_("Select"_(lineitemTable, "Where"_("Greater"_("DateObject"_("1998-08-31"),
                                                                        "L_SHIPDATE"_))),
                           "As"_("RETURNFLAG_AND_LINESTATUS"_,
                                 "StringJoin"_("L_RETURNFLAG"_, "L_LINESTATUS"_), "L_QUANTITY"_,
                                 "L_QUANTITY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                 "L_DISCOUNT"_, "L_DISCOUNT"_, "calc1"_,
                                 "Minus"_(1.0, "L_DISCOUNT"_), "calc2"_, "Plus"_(1.0, "L_TAX"_))),
                "By"_("RETURNFLAG_AND_LINESTATUS"_),
                "As"_("SUM_QTY"_, "Sum"_("L_QUANTITY"_), "SUM__BASE_PRICE"_,
                      "Sum"_("L_EXTENDEDPRICE"_), "SUM_DISC_PRICE"_,
                      "Sum"_("Times"_("L_EXTENDEDPRICE"_, "calc1"_)), "SUM_CHARGES"_,
                      "Sum"_("Times"_("L_EXTENDEDPRICE"_, "calc1"_, "calc2"_)), "SUM_DISC"_,
                      "Sum"_("L_DISCOUNT"_), "COUNT_ORDER"_, "Count"_("L_QUANTITY"_))),
            "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "SUM_QTY"_,
                  "SUM_QTY"_, "SUM__BASE_PRICE"_, "SUM__BASE_PRICE"_, "SUM_DISC_PRICE"_,
                  "SUM_DISC_PRICE"_, "SUM_CHARGES"_, "SUM_CHARGES"_, "AVG_QTY"_,
                  "Divide"_("SUM_QTY"_, "COUNT_ORDER"_), "AVG_PRICE"_,
                  "Divide"_("SUM__BASE_PRICE"_, "COUNT_ORDER"_), "AVG_DISC"_,
                  "Divide"_("SUM_DISC"_, "COUNT_ORDER"_), "COUNT_ORDER"_, "COUNT_ORDER"_)),
        /*"Function"_(
            "tuple"_,
            "As"_("RETURNFLAG_AND_LINESTATUS"_, "RETURNFLAG_AND_LINESTATUS"_, "SUM_QTY"_,
                  "Column"_("tuple"_, 2), "SUM__BASE_PRICE"_, "Column"_("tuple"_, 3),
                  "SUM_DISC_PRICE"_, "Column"_("tuple"_, 4), "SUM_CHARGES"_,
                  "Column"_("tuple"_, 5), "AVG_QTY"_,
                  "Divide"_("Column"_("tuple"_, 2), "Column"_("tuple"_, 7)), "AVG_PRICE"_,
                  "Divide"_("Column"_("tuple"_, 3), "Column"_("tuple"_, 7)), "AVG_DISC"_,
                  "Divide"_("Column"_("tuple"_, 6), "Column"_("tuple"_, 7)), "COUNT_ORDER"_,
                  "Column"_("tuple"_, 7)))),*/
        "By"_("RETURNFLAG_AND_LINESTATUS"_));

    INFO(eval(queryQ1));
    CHECK(eval("Group"_(queryQ1, "Count"_)) == "List"_("List"_(4)));

    INFO(eval("Extract"_(queryQ1, 1)));
    INFO(eval("Extract"_(queryQ1, 2)));
    INFO(eval("Extract"_(queryQ1, 3)));
    INFO(eval("Extract"_(queryQ1, 4)));

    CHECK(get<string>(eval("Extract"_("Extract"_(queryQ1, 1), 1))) == "AF");
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ1, 1), 2))) == 37474);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 3))) == 37569624.0_a); // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 4))) == 35676191.5_a); // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 5))) == 37101416.0_a); // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 6))) == 25.354533_a);  // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 7))) == 25419.23_a);   // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ1, 1), 8))) == 0.050866_a);   // NOLINT
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ1, 1), 9))) == 1478);
  }

  SECTION("Q3 Lineitem") {
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                           "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));

    CHECK(eval("Extract"_("Extract"_(lineitemTable, 1), 3)) == eval("DateObject"_("1996-03-13")));

    auto const query = "Project"_(
        "Select"_(lineitemTable, "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1995-03-15")))),
        "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_EXTENDEDPRICE"_,
              "L_EXTENDEDPRICE"_));

    INFO(eval(query));
    CHECK(eval("Group"_(query, "Count"_)) == "List"_("List"_(3252)));

    INFO(eval("Extract"_(query, 1)));
    INFO(eval("Extract"_(query, 2)));
    INFO(eval("Extract"_(query, 3)));
    INFO(eval("Extract"_(query, 4)));
    INFO(eval("Extract"_(query, 5)));

    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 1))) == 1);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(query, 1), 2))) == 0.04_a);     // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(query, 1), 3))) == 17954.55_a); // NOLINT
  }

  SECTION("Q3 Orders") {
    auto const& ordersTable = "Project"_(
        "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_,
                         "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));

    CHECK(eval("Extract"_("Extract"_(ordersTable, 1), 2)) == eval("DateObject"_("1996-01-02")));

    auto const query =
        "Select"_(ordersTable, "Where"_("Greater"_("DateObject"_("1995-03-15"), "O_ORDERDATE"_)));

    INFO(eval(query));
    CHECK(eval("Group"_(query, "Count"_)) == "List"_("List"_(726)));

    INFO(eval("Extract"_(query, 1)));
    INFO(eval("Extract"_(query, 2)));
    INFO(eval("Extract"_(query, 3)));
    INFO(eval("Extract"_(query, 4)));
    INFO(eval("Extract"_(query, 5)));

    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 1))) == 3);
    CHECK(eval("Extract"_("Extract"_(query, 1), 2)) == eval("DateObject"_("1993-10-14")));
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 3))) == 124);
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 4))) == 0);
  }

  SECTION("Q3 First Join") {
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                           "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));
    auto const& ordersTable = "Project"_(
        "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_,
                         "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));
    auto const query = "Project"_(
        "Join"_(
            "Select"_(ordersTable,
                      "Where"_("Greater"_("DateObject"_("1995-03-15"), "O_ORDERDATE"_))),
            "Project"_("Select"_(lineitemTable,
                                 "Where"_("Greater"_("L_SHIPDATE"_, "DateObject"_("1995-03-15")))),
                       "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                             "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
            "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
        "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_EXTENDEDPRICE"_,
              "L_EXTENDEDPRICE"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
              "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));

    INFO(eval(query));
    CHECK(eval("Group"_(query, "Count"_)) == "List"_("List"_(133)));

    INFO(eval("Extract"_(query, 1)));
    INFO(eval("Extract"_(query, 2)));
    INFO(eval("Extract"_(query, 3)));
    INFO(eval("Extract"_(query, 4)));
    INFO(eval("Extract"_(query, 5)));

    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 1))) == 359);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(query, 1), 2))) == 0.10_a);     // NOLINT
    CHECK(get<double_t>(eval("Extract"_("Extract"_(query, 1), 3))) == 37623.42_a); // NOLINT
    CHECK(eval("Extract"_("Extract"_(query, 1), 4)) == eval("DateObject"_("1994-12-19")));
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 5))) == 79);
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 6))) == 0);
  }

  SECTION("Q3 Second Join") {
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                           "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));
    auto const& customerTable = "Project"_(
        "CUSTOMER"_, "As"_("C_CUSTKEY"_, "C_CUSTKEY"_, "C_MKTSEGMENT"_, "C_MKTSEGMENT"_));
    auto const& ordersTable = "Project"_(
        "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_,
                         "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));
    auto const query = "Project"_(
        "Join"_(
            "Project"_(
                "Select"_(customerTable, "Where"_("StringContainsQ"_("C_MKTSEGMENT"_, "BUILDING"))),
                "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
            "Project"_(
                "Join"_("Select"_(ordersTable, "Where"_("Greater"_("DateObject"_("1995-03-15"),
                                                                   "O_ORDERDATE"_))),
                        "Project"_("Select"_(lineitemTable,
                                             "Where"_("Greater"_("L_SHIPDATE"_,
                                                                 "DateObject"_("1995-03-15")))),
                                   "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                                         "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_)),
                        "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                      "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                      "O_CUSTKEY"_, "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
            "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
        "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)), "L_ORDERKEY"_,
              "L_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));

    INFO(eval(query));
    CHECK(eval("Group"_(query, "Count"_)) == "List"_("List"_(14)));

    INFO(eval("Extract"_(query, 1)));
    INFO(eval("Extract"_(query, 2)));
    INFO(eval("Extract"_(query, 3)));
    INFO(eval("Extract"_(query, 4)));
    INFO(eval("Extract"_(query, 5)));

    CHECK(get<double_t>(eval("Extract"_("Extract"_(query, 1), 1))) == 43728.0480_a); // NOLINT
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 2))) == 742);
    CHECK(eval("Extract"_("Extract"_(query, 1), 3)) == eval("DateObject"_("1994-12-23")));
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(query, 1), 4))) == 0);
  }

  SECTION("Q3") {
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                           "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));
    auto const& customerTable = "Project"_(
        "CUSTOMER"_, "As"_("C_CUSTKEY"_, "C_CUSTKEY"_, "C_MKTSEGMENT"_, "C_MKTSEGMENT"_));
    auto const& ordersTable = "Project"_(
        "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_, "O_CUSTKEY"_,
                         "O_CUSTKEY"_, "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_));
    auto const queryQ3 = "Top"_(
        "Group"_(
            "Project"_(
                "Join"_(
                    "Project"_("Select"_(customerTable,
                                         "Where"_("StringContainsQ"_("C_MKTSEGMENT"_, "BUILDING"))),
                               "As"_("C_CUSTKEY"_, "C_CUSTKEY"_)),
                    "Project"_(
                        "Join"_(
                            "Select"_(ordersTable, "Where"_("Greater"_("DateObject"_("1995-03-15"),
                                                                       "O_ORDERDATE"_))),
                            "Project"_("Select"_(lineitemTable,
                                                 "Where"_("Greater"_("L_SHIPDATE"_,
                                                                     "DateObject"_("1995-03-15")))),
                                       "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_,
                                             "L_DISCOUNT"_, "L_EXTENDEDPRICE"_,
                                             "L_EXTENDEDPRICE"_)),
                            "Where"_("Equal"_("O_ORDERKEY"_, "L_ORDERKEY"_))),
                        "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                              "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "O_ORDERDATE"_,
                              "O_ORDERDATE"_, "O_CUSTKEY"_, "O_CUSTKEY"_, "O_SHIPPRIORITY"_,
                              "O_SHIPPRIORITY"_)),
                    "Where"_("Equal"_("C_CUSTKEY"_, "O_CUSTKEY"_))),
                "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "Minus"_(1.0, "L_DISCOUNT"_)),
                      "L_ORDERKEY"_, "L_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_,
                      "O_SHIPPRIORITY"_, "O_SHIPPRIORITY"_)),
            "By"_("L_ORDERKEY"_, "O_ORDERDATE"_, "O_SHIPPRIORITY"_), "Sum"_("revenue"_)),
        "By"_("Minus"_("Sum_revenue"_), "O_ORDERDATE"_), 10);

    INFO(eval(queryQ3));
    CHECK(eval("Group"_(queryQ3, "Count"_)) == "List"_("List"_(8)));

    INFO(eval("Extract"_(queryQ3, 1)));
    INFO(eval("Extract"_(queryQ3, 2)));
    INFO(eval("Extract"_(queryQ3, 3)));
    INFO(eval("Extract"_(queryQ3, 4)));
    INFO(eval("Extract"_(queryQ3, 5)));
    INFO(eval("Extract"_(queryQ3, 6)));
    INFO(eval("Extract"_(queryQ3, 7)));
    INFO(eval("Extract"_(queryQ3, 8)));

    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ3, 1), 1))) == 1637);
    CHECK(eval("Extract"_("Extract"_(queryQ3, 1), 2)) == eval("DateObject"_("1995-02-08")));
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ3, 1), 3))) == 0);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ3, 1), 4))) == 164224.9253_a); // NOLINT
  }

  SECTION("Q6") {
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_QUANTITY"_, "L_QUANTITY"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                           "L_SHIPDATE"_, "L_SHIPDATE"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_));
    auto const queryQ6 = "Group"_(
        "Project"_("Select"_(lineitemTable,
                             "Where"_("And"_(
                                 "Greater"_(24, "L_QUANTITY"_), "Greater"_("L_DISCOUNT"_, 0.0499),
                                 "Greater"_(0.07001, "L_DISCOUNT"_),
                                 "Greater"_("DateObject"_("1995-01-01"), "L_SHIPDATE"_),
                                 "Greater"_("L_SHIPDATE"_, "DateObject"_("1993-12-31"))))),
                   "As"_("revenue"_, "Times"_("L_EXTENDEDPRICE"_, "L_DISCOUNT"_))),
        "Sum"_("revenue"_));

    INFO(eval(queryQ6));
    CHECK(eval("Group"_(queryQ6, "Count"_)) == "List"_("List"_(1)));
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ6, 1), 1))) == 77949.9150_a); // NOLINT
  }

  SECTION("Q9 First Join") {
    auto const& partTable =
        "Project"_("PART"_, "As"_("P_PARTKEY"_, "P_PARTKEY"_, "P_NAME"_, "P_NAME"_));
    auto const& partsuppTable =
        "Project"_("PARTSUPP"_, "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                      "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_));
    auto const query1 = "Project"_(
        "Join"_("Project"_("Select"_(partTable, "Where"_("StringContainsQ"_("P_NAME"_, "green"))),
                           "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                partsuppTable, "Where"_("Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
        "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
              "PS_SUPPLYCOST"_));
    auto output1 = eval(query1);

    INFO(output1);
    CHECK(get<int64_t>(eval("Length"_(output1))) == 36);

    INFO(eval("Extract"_(output1, 1)));
    INFO(eval("Extract"_(output1, 2)));
    INFO(eval("Extract"_(output1, 3)));
    INFO(eval("Extract"_(output1, 4)));
    INFO(eval("Extract"_(output1, 5)));

    CHECK(get<int64_t>(eval("Extract"_("Extract"_(output1, 1), 1))) == 3);
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(output1, 1), 2))) == 4);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(output1, 1), 3))) == 920.92_a); // NOLINT

    SECTION("Q9 Second Join") {
      auto const& lineitemTable = "Project"_(
          "LINEITEM"_, "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_SUPPKEY"_, "L_ORDERKEY"_,
                             "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                             "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_));
      auto const query2 =
          "Project"_("Join"_(output1, lineitemTable,
                             "Where"_("Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                               "List"_("L_PARTKEY"_, "L_SUPPKEY"_)))),
                     "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_SUPPKEY"_, "L_SUPPKEY"_,
                           "L_ORDERKEY"_, "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                           "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_));
      auto output2 = eval(query2);

      INFO(output2);
      CHECK(get<int64_t>(eval("Length"_(output2))) == 493);

      INFO(eval("Extract"_(output2, 1)));
      INFO(eval("Extract"_(output2, 2)));
      INFO(eval("Extract"_(output2, 3)));
      INFO(eval("Extract"_(output2, 4)));
      INFO(eval("Extract"_(output2, 5)));

      CHECK(get<double_t>(eval("Extract"_("Extract"_(output2, 1), 1))) == 498.13_a); // NOLINT
      CHECK(get<int64_t>(eval("Extract"_("Extract"_(output2, 1), 2))) == 6);
      CHECK(get<int64_t>(eval("Extract"_("Extract"_(output2, 1), 3))) == 1);
      CHECK(get<double_t>(eval("Extract"_("Extract"_(output2, 1), 4))) == 25284.0_a); // NOLINT
      CHECK(get<double_t>(eval("Extract"_("Extract"_(output2, 1), 5))) == 0.09_a);    // NOLINT
      CHECK(get<int64_t>(eval("Extract"_("Extract"_(output2, 1), 6))) == 28);

      SECTION("Q9 Third Join") {
        auto const& supplierTable = "Project"_(
            "SUPPLIER"_, "As"_("S_SUPPKEY"_, "S_SUPPKEY"_, "S_NATIONKEY"_, "S_NATIONKEY"_));
        auto const query3 = "Project"_(
            "Join"_(supplierTable, output2, "Where"_("Equal"_("S_SUPPKEY"_, "L_SUPPKEY"_))),
            "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                  "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_,
                  "L_QUANTITY"_, "L_QUANTITY"_, "S_NATIONKEY"_, "S_NATIONKEY"_));
        auto output3 = eval(query3);

        INFO(output3);
        CHECK(get<int64_t>(eval("Length"_(output3))) == 493);

        INFO(eval("Extract"_(output3, 1)));
        INFO(eval("Extract"_(output3, 2)));
        INFO(eval("Extract"_(output3, 3)));
        INFO(eval("Extract"_(output3, 4)));
        INFO(eval("Extract"_(output3, 5)));

        CHECK(get<double_t>(eval("Extract"_("Extract"_(output3, 1), 1))) == 498.13_a); // NOLINT
        CHECK(get<int64_t>(eval("Extract"_("Extract"_(output3, 1), 2))) == 1);
        CHECK(get<double_t>(eval("Extract"_("Extract"_(output3, 1), 3))) == 25284.0_a); // NOLINT
        CHECK(get<double_t>(eval("Extract"_("Extract"_(output3, 1), 4))) == 0.09_a);    // NOLINT
        CHECK(get<int64_t>(eval("Extract"_("Extract"_(output3, 1), 5))) == 28);
        CHECK(get<int64_t>(eval("Extract"_("Extract"_(output3, 1), 6))) == 14);

        SECTION("Q9 Fourth Join") {
          auto const& ordersTable = "Project"_(
              "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_));
          auto const query4 = "Project"_(
              "Join"_(output3, ordersTable, "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
              "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                    "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_, "S_NATIONKEY"_,
                    "S_NATIONKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_));
          auto output4 = eval(query4);

          INFO(output4);
          CHECK(get<int64_t>(eval("Length"_(output4))) == 493);

          INFO(eval("Extract"_(output4, 1)));
          INFO(eval("Extract"_(output4, 2)));
          INFO(eval("Extract"_(output4, 3)));
          INFO(eval("Extract"_(output4, 4)));
          INFO(eval("Extract"_(output4, 5)));

          CHECK(get<double_t>(eval("Extract"_("Extract"_(output4, 1), 1))) == 498.13_a);  // NOLINT
          CHECK(get<double_t>(eval("Extract"_("Extract"_(output4, 1), 2))) == 25284.0_a); // NOLINT
          CHECK(get<double_t>(eval("Extract"_("Extract"_(output4, 1), 3))) == 0.09_a);    // NOLINT
          CHECK(get<int64_t>(eval("Extract"_("Extract"_(output4, 1), 4))) == 28);
          CHECK(get<int64_t>(eval("Extract"_("Extract"_(output4, 1), 5))) == 14);
          CHECK(eval("Extract"_("Extract"_(output4, 1), 6)) == eval("DateObject"_("1996-01-02")));

          SECTION("Q9 Fifth Join") {
            auto const& nationTable =
                "Project"_("NATION"_, "As"_("N_NAME"_, "N_NAME"_, "N_NATIONKEY"_, "N_NATIONKEY"_));
            auto const query5 = "Project"_(
                "Join"_(nationTable, output4, "Where"_("Equal"_("N_NATIONKEY"_, "S_NATIONKEY"_))),
                "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                      "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                               "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_))));
            auto output5 = eval(query5);

            INFO(output5);
            CHECK(get<int64_t>(eval("Length"_(output5))) == 493);

            INFO(eval("Extract"_(output5, 1)));
            INFO(eval("Extract"_(output5, 2)));
            INFO(eval("Extract"_(output5, 3)));
            INFO(eval("Extract"_(output5, 4)));
            INFO(eval("Extract"_(output5, 5)));

            CHECK(get<string>(eval("Extract"_("Extract"_(output5, 1), 1))) == "KENYA");
            CHECK(get<int64_t>(eval("Extract"_("Extract"_(output5, 1), 2))) == 1996);
            CHECK(get<double_t>(eval("Extract"_("Extract"_(output5, 1), 3))) == 9060.8_a); // NOLINT

            std::string groupbyKey = GENERATE("Nation", "Year", "Nation/Year");

            DYNAMIC_SECTION("Q9 Group/Order By " << groupbyKey) {
              auto const query6 =
                  groupbyKey == "Nation"
                      ? "Sort"_("Group"_(output5, "By"_("nation"_), "Sum"_("amount"_)),
                                "By"_("nation"_))
                  : groupbyKey == "Year"
                      ? "Sort"_("Group"_(output5, "By"_("o_year"_), "Sum"_("amount"_)),
                                "By"_("Minus"_("o_year"_)))
                      : "Sort"_("Group"_(output5, "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
                                "By"_("nation"_, "Minus"_("o_year"_)));
              auto output6 = eval(query6);

              auto expectedCount = groupbyKey == "Nation" ? 9 : groupbyKey == "Year" ? 7 : 60;

              INFO(output6);
              auto count = get<int64_t>(eval("Length"_(output6)));
              for(int i = 1; i <= count; ++i) {
                UNSCOPED_INFO(eval("Extract"_(output6, i)));
              }
              CHECK(count == expectedCount);

              if(groupbyKey == "Nation") {
                CHECK(get<std::string>(eval("Extract"_("Extract"_(output6, 1), 1))) == "ARGENTINA");
                CHECK(get<double_t>(eval("Extract"_("Extract"_(output6, 1), 2))) ==
                      121664.3574_a); // NOLINT
              } else if(groupbyKey == "Year") {
                CHECK(get<int64_t>(eval("Extract"_("Extract"_(output6, 1), 1))) == 1998);
                CHECK(get<double_t>(eval("Extract"_("Extract"_(output6, 1), 2))) ==
                      521464.7810_a); // NOLINT
              } else {
                CHECK(get<std::string>(eval("Extract"_("Extract"_(output6, 1), 1))) == "ARGENTINA");
                CHECK(get<int64_t>(eval("Extract"_("Extract"_(output6, 1), 2))) == 1998);
                CHECK(get<double_t>(eval("Extract"_("Extract"_(output6, 1), 3))) ==
                      17779.0697_a); // NOLINT
              }
            }
          }
        }
      }
    }
  }

  SECTION("Q9") {
    auto const& nationTable =
        "Project"_("NATION"_, "As"_("N_NAME"_, "N_NAME"_, "N_NATIONKEY"_, "N_NATIONKEY"_));
    auto const& supplierTable =
        "Project"_("SUPPLIER"_, "As"_("S_SUPPKEY"_, "S_SUPPKEY"_, "S_NATIONKEY"_, "S_NATIONKEY"_));
    auto const& ordersTable =
        "Project"_("ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_ORDERDATE"_));
    auto const& partTable =
        "Project"_("PART"_, "As"_("P_PARTKEY"_, "P_PARTKEY"_, "P_NAME"_, "P_NAME"_));
    auto const& partsuppTable =
        "Project"_("PARTSUPP"_, "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_, "PS_SUPPKEY"_,
                                      "PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_));
    auto const& lineitemTable = "Project"_(
        "LINEITEM"_, "As"_("L_PARTKEY"_, "L_PARTKEY"_, "L_SUPPKEY"_, "L_SUPPKEY"_, "L_ORDERKEY"_,
                           "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                           "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_));
    auto const queryQ9 = "Sort"_(
        "Group"_(
            "Project"_(
                "Join"_(
                    nationTable,
                    "Project"_(
                        "Join"_(
                            "Project"_(
                                "Join"_(
                                    supplierTable,
                                    "Project"_(
                                        "Join"_(
                                            "Project"_(
                                                "Join"_("Project"_(
                                                            "Select"_(partTable,
                                                                      "Where"_("StringContainsQ"_(
                                                                          "P_NAME"_, "green"))),
                                                            "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                                        partsuppTable,
                                                        "Where"_(
                                                            "Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                                "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_,
                                                      "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                                      "PS_SUPPLYCOST"_)),
                                            lineitemTable,
                                            "Where"_(
                                                "Equal"_("List"_("PS_PARTKEY"_, "PS_SUPPKEY"_),
                                                         "List"_("L_PARTKEY"_, "L_SUPPKEY"_)))),
                                        "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_SUPPKEY"_,
                                              "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                              "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                              "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_)),
                                    "Where"_("Equal"_("S_SUPPKEY"_, "L_SUPPKEY"_))),
                                "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_ORDERKEY"_,
                                      "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                      "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_,
                                      "S_NATIONKEY"_, "S_NATIONKEY"_)),
                            ordersTable, "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                        "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_EXTENDEDPRICE"_,
                              "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                              "L_QUANTITY"_, "S_NATIONKEY"_, "S_NATIONKEY"_, "O_ORDERDATE"_,
                              "O_ORDERDATE"_)),
                    "Where"_("Equal"_("N_NATIONKEY"_, "S_NATIONKEY"_))),
                "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                      "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                               "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
            "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
        "By"_("nation"_, "Minus"_("o_year"_)));

    auto const queryQ9b = "Sort"_(
        "Group"_(
            "Project"_(
                "Join"_(
                    nationTable,
                    "Project"_(
                        "Join"_(
                            "Project"_(
                                "Join"_(
                                    supplierTable,
                                    "Project"_(
                                        "Join"_(
                                            "Project"_(
                                                "Join"_("Project"_(
                                                            "Select"_(partTable,
                                                                      "Where"_("StringContainsQ"_(
                                                                          "P_NAME"_, "green"))),
                                                            "As"_("P_PARTKEY"_, "P_PARTKEY"_)),
                                                        partsuppTable,
                                                        "Where"_(
                                                            "Equal"_("P_PARTKEY"_, "PS_PARTKEY"_))),
                                                "As"_("PS_PARTKEY"_, "PS_PARTKEY"_, "PS_SUPPKEY"_,
                                                      "PS_SUPPKEY"_, "PS_SUPPLYCOST"_,
                                                      "PS_SUPPLYCOST"_)),
                                            lineitemTable,
                                            "Where"_(
                                                "And"_("Equal"_("PS_PARTKEY"_, "L_PARTKEY"_),
                                                       "Equal"_("PS_SUPPKEY"_, "L_SUPPKEY"_)))),
                                        "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_SUPPKEY"_,
                                              "L_SUPPKEY"_, "L_ORDERKEY"_, "L_ORDERKEY"_,
                                              "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_, "L_DISCOUNT"_,
                                              "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_)),
                                    "Where"_("Equal"_("S_SUPPKEY"_, "L_SUPPKEY"_))),
                                "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_ORDERKEY"_,
                                      "L_ORDERKEY"_, "L_EXTENDEDPRICE"_, "L_EXTENDEDPRICE"_,
                                      "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_, "L_QUANTITY"_,
                                      "S_NATIONKEY"_, "S_NATIONKEY"_)),
                            ordersTable, "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                        "As"_("PS_SUPPLYCOST"_, "PS_SUPPLYCOST"_, "L_EXTENDEDPRICE"_,
                              "L_EXTENDEDPRICE"_, "L_DISCOUNT"_, "L_DISCOUNT"_, "L_QUANTITY"_,
                              "L_QUANTITY"_, "S_NATIONKEY"_, "S_NATIONKEY"_, "O_ORDERDATE"_,
                              "O_ORDERDATE"_)),
                    "Where"_("Equal"_("N_NATIONKEY"_, "S_NATIONKEY"_))),
                "As"_("nation"_, "N_NAME"_, "o_year"_, "Year"_("O_ORDERDATE"_), "amount"_,
                      "Minus"_("Times"_("L_EXTENDEDPRICE"_, "Minus"_(1, "L_DISCOUNT"_)),
                               "Times"_("PS_SUPPLYCOST"_, "L_QUANTITY"_)))),
            "By"_("nation"_, "o_year"_), "Sum"_("amount"_)),
        "By"_("nation"_, "Minus"_("o_year"_)));

    INFO(eval(queryQ9));
    auto count =
        get<int64_t>(eval("Extract"_("Extract"_(eval("Group"_(queryQ9, "Count"_)), 1), 1)));
    for(int i = 1; i <= count; ++i) {
      UNSCOPED_INFO(eval("Extract"_(queryQ9, i)));
    }
    CHECK(count == 60);

    CHECK(get<string>(eval("Extract"_("Extract"_(queryQ9, 1), 1))) == "ARGENTINA");
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ9, 1), 2))) == 1998);
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ9, 1), 3))) == 17779.0697_a); // NOLINT

    INFO(eval(queryQ9b));
    auto countb =
        get<int64_t>(eval("Extract"_("Extract"_(eval("Group"_(queryQ9b, "Count"_)), 1), 1)));
    for(int i = 1; i <= countb; ++i) {
      UNSCOPED_INFO(eval("Extract"_(queryQ9b, i)));
    }
    CHECK(countb == 60);
  }

  SECTION("Q18") {
    auto const& lineitemTable =
        "Project"_("LINEITEM"_, "As"_("L_ORDERKEY"_, "L_ORDERKEY"_, "L_QUANTITY"_, "L_QUANTITY"_));
    auto const& customerTable =
        "Project"_("CUSTOMER"_, "As"_("C_NAME"_, "C_NAME"_, "C_CUSTKEY"_, "C_CUSTKEY"_));
    auto const& ordersTable = "Project"_(
        "ORDERS"_, "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_, "O_CUSTKEY"_, "O_ORDERDATE"_,
                         "O_ORDERDATE"_, "O_TOTALPRICE"_, "O_TOTALPRICE"_));
    auto const queryQ18 = "Top"_(
        "Group"_(
            "Join"_("Project"_(
                        "Join"_("Join"_("Project"_(
                                            "Select"_("Group"_(lineitemTable, "By"_("L_ORDERKEY"_),
                                                               "Sum"_("L_QUANTITY"_)),
                                                      "Where"_("Greater"_("Sum_L_QUANTITY"_, 100))),
                                            "As"_("INNER_L_ORDERKEY"_, "L_ORDERKEY"_)),
                                        lineitemTable,
                                        "Where"_("Equal"_("INNER_L_ORDERKEY"_, "L_ORDERKEY"_))),
                                ordersTable, "Where"_("Equal"_("L_ORDERKEY"_, "O_ORDERKEY"_))),
                        "As"_("O_ORDERKEY"_, "O_ORDERKEY"_, "O_CUSTKEY"_, "O_CUSTKEY"_,
                              "O_ORDERDATE"_, "O_ORDERDATE"_, "O_TOTALPRICE"_, "O_TOTALPRICE"_,
                              "L_QUANTITY"_, "L_QUANTITY"_)),
                    customerTable, "Where"_("Equal"_("O_CUSTKEY"_, "C_CUSTKEY"_))),
            "By"_("C_NAME"_, "C_CUSTKEY"_, "O_ORDERKEY"_, "O_ORDERDATE"_, "O_TOTALPRICE"_),
            "Sum"_("L_QUANTITY"_)),
        "By"_("Minus"_("O_TOTALPRICE"_), "O_ORDERDATE"_), 100);

    INFO(eval(queryQ18));
    CHECK(eval("Group"_(queryQ18, "Count"_)) == "List"_("List"_(100)));

    INFO(eval("Extract"_(queryQ18, 1)));
    INFO(eval("Extract"_(queryQ18, 2)));
    INFO(eval("Extract"_(queryQ18, 3)));
    INFO(eval("Extract"_(queryQ18, 4)));
    INFO(eval("Extract"_(queryQ18, 5)));
    INFO(eval("Extract"_(queryQ18, 6)));
    INFO(eval("Extract"_(queryQ18, 7)));
    INFO(eval("Extract"_(queryQ18, 8)));
    INFO(eval("Extract"_(queryQ18, 9)));
    INFO(eval("Extract"_(queryQ18, 10)));

    CHECK(get<string>(eval("Extract"_("Extract"_(queryQ18, 1), 1))) == "Customer#000000070");
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ18, 1), 2))) == 70);
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ18, 1), 3))) == 2567);
    CHECK(eval("Extract"_("Extract"_(queryQ18, 1), 4)) == eval("DateObject"_("1998-02-27")));
    CHECK(get<double_t>(eval("Extract"_("Extract"_(queryQ18, 1), 5))) == 263411.28_a); // NOLINT
    CHECK(get<int64_t>(eval("Extract"_("Extract"_(queryQ18, 1), 6))) == 266);
  }
}

int main(int argc, char* argv[]) {
  Catch::Session session;
  session.cli(
      session.cli() | Catch::clara::Opt(librariesToTest, "library")["--library"] |
      Catch::clara::Opt(defaultBatchSize, "batch-size (num rows)")["--batch-size"] |
      Catch::clara::Opt(TpchDatasetSizeMb, "tpch-dataset-size (Mb)")["--tpch-dataset-size"] |
      Catch::clara::Opt(TpchImputedDatasetSizeMb,
                        "tpch-imputed-dataset-size (Mb)")["--tpch-imputed-dataset-size"]);
  int returnCode = session.applyCommandLine(argc, argv);
  if(returnCode != 0) {
    return returnCode;
  }
  return session.run();
}
