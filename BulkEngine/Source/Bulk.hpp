#pragma once

#include "BulkExpression.hpp"

#include <BOSS.hpp>
#include <Engine.hpp>
#include <Expression.hpp>
#include <Utilities.hpp>

#include <algorithm>
#include <deque>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>

using namespace std;

namespace {                        // https://stackoverflow.com/a/52303687
template <typename> struct tag {}; // <== this one IS literal

template <typename T, typename V> struct get_index;

template <typename... Ts> struct get_index<boss::engines::bulk::Expression, std::variant<Ts...>> {
  static constexpr double value = static_cast<double>(variant_size_v<std::variant<Ts...>>);
};

template <typename T, typename... Ts> struct get_index<T, std::variant<Ts...>> {
  static constexpr double value = static_cast<double>(std::variant<tag<Ts>...>(tag<T>()).index());
};

template <typename Head, typename... AcceptedTypes> struct TypeIdentifier {
  static constexpr double value =
      TypeIdentifier<AcceptedTypes...>::value *
          static_cast<double>(variant_size_v<boss::engines::bulk::Expression::SuperType> + 1) +
      static_cast<double>(get_index<Head, boss::engines::bulk::Expression::SuperType>::value + 1);
};

template <typename Head> struct TypeIdentifier<Head> {
  static constexpr double value =
      static_cast<double>(get_index<Head, boss::engines::bulk::Expression::SuperType>::value + 1);
};

} // namespace

#ifdef _WIN32
extern "C" {
__declspec(dllexport) BOSSExpression* evaluate(BOSSExpression* e);
__declspec(dllexport) void reset();
}
#endif // _WIN32

namespace boss::engines::bulk {

typedef std::function<void(Expression const&, bool)> OperatorPushUpFunction;
typedef std::function<void(Expression&&, bool)> OperatorPushUpMoveFunction;

class CallableOperator;

class GenericOperator {
public:
  virtual std::unique_ptr<CallableOperator>
  getCallableInstance(OperatorPushUpFunction const& pushUp,
                      OperatorPushUpMoveFunction const& pushUpMove) const = 0;
  virtual ~GenericOperator() = default;
  GenericOperator(GenericOperator&) = delete;
  GenericOperator& operator=(GenericOperator&) = delete;
  GenericOperator(GenericOperator&&) = default;
  GenericOperator& operator=(GenericOperator&&) = delete;
  GenericOperator() = default;
};

class CallableOperator : public GenericOperator {
public:
  explicit CallableOperator(OperatorPushUpFunction pushUp, OperatorPushUpMoveFunction pushUpMove)
      : pushUpFunc(std::move(pushUp)), pushUpMoveFunc(std::move(pushUpMove)),
        cachedEvaluatorInfoIt(cachedEvaluatorInfo.begin()) {}
  void setPushUpCalls(OperatorPushUpFunction pushUp, OperatorPushUpMoveFunction pushUpMove) {
    pushUpFunc = std::move(pushUp);
    pushUpMoveFunc = std::move(pushUpMove);
  }
  template <typename... Args> bool checkArguments(Args const&... /*args*/) const { return true; }
  virtual bool operator()(ExpressionArguments const& args) = 0;
  virtual void close() {}
  ~CallableOperator() override {
    for(auto& [cachedOpPtr, cachedOpShape] : cachedEvaluatorInfo) {
      if(cachedOpPtr != nullptr) {
        // reset any internal state
        cachedOpPtr->close();
      }
    }
  }
  void rewindCachedInfoPosition() { cachedEvaluatorInfoIt = cachedEvaluatorInfo.begin(); }
  CallableOperator(CallableOperator&) = delete;
  CallableOperator& operator=(CallableOperator&) = delete;
  CallableOperator(CallableOperator&&) = default;
  CallableOperator& operator=(CallableOperator&&) = delete;

protected:
  void pushUp(Expression const& e) {
    pushUpFunc(e, true);
    // consider pushUp as a new round of evaluation
    // so operators like Evaluate can re-use cache during the iterative evaluate/pushs in close()
    this->rewindCachedInfoPosition();
  }
  void pushUp(Expression&& e) {
    pushUpMoveFunc(std::move(e), true);
    // consider pushUp as a new round of evaluation
    // so operators like Evaluate can re-use cache during the iterative evaluate/pushs in close()
    this->rewindCachedInfoPosition();
  }

  template <typename T>
  inline std::enable_if_t<
      std::conjunction_v<
          std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
          std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>,
      Expression>
  evaluateInternal(T&& e);

  template <typename T>
  inline std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>, Expression>
  evaluateInternal(T&& e);

  template <typename T>
  inline std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>,
                          Expression>
  evaluateInternal(T&& e);

  template <typename T, typename Func>
  inline std::enable_if_t<std::conjunction_v<
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
  evaluateInternal(T&& e, Func&& func);

  template <typename T, typename Func>
  inline std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>>
  evaluateInternal(T&& e, Func&& func);

  template <typename T, typename Func>
  inline std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>
  evaluateInternal(T&& e, Func&& func);

private:
  OperatorPushUpFunction pushUpFunc;
  OperatorPushUpMoveFunction pushUpMoveFunc;
  typedef std::list<std::pair<std::unique_ptr<CallableOperator>, double>> CachedEvaluatorInfoList;
  CachedEvaluatorInfoList cachedEvaluatorInfo;
  CachedEvaluatorInfoList::iterator cachedEvaluatorInfoIt;
};

class OperatorDirectoryHash {
public:
  std::size_t operator()(const std::pair<std::string, double>& p) const {
    auto h1 = std::hash<std::string>{}(p.first);
    auto h2 = std::hash<double>{}(p.second);
    return h1 ^ h2;
  }
};

class OperatorDirectory
    : public std::unordered_map<std::pair<std::string, double>, std::unique_ptr<GenericOperator>,
                                OperatorDirectoryHash> {
  template <template <typename...> typename Operator, typename Tuple, std::size_t... I>
  void emplaceOperatorWithSpecificArgumentTypes(std::string const& name,
                                                std::index_sequence<I...> /*unused*/) {
    auto dummyFunc = [](auto&& /*unused*/, bool /*unused*/) {};
    unordered_map::emplace(
        std::pair<std::string, double>{name,
                                       TypeIdentifier<std::tuple_element_t<I, Tuple>...>::value},
        new Operator<std::tuple_element_t<I, Tuple>...>(dummyFunc, dummyFunc));
  }

  template <template <typename...> typename Operator, typename Tuple,
            typename Indices = std::make_index_sequence<std::tuple_size_v<Tuple>>>
  void emplaceOperatorWithSpecificArgumentTypes(std::string const& name) {
    emplaceOperatorWithSpecificArgumentTypes<Operator, Tuple>(name, Indices{});
  }

  template <template <typename...> typename Operator, typename ArgumentTypes, std::size_t... I>
  void emplaceOperatorForEachArgumentTypesVariant(std::string const& name,
                                                  std::index_sequence<I...> /*unused*/) {
    (emplaceOperatorWithSpecificArgumentTypes<Operator,
                                              std::variant_alternative_t<I, ArgumentTypes>>(name),
     ...);
  }

  template <template <typename...> typename Operator, typename ArgumentTypes,
            typename Indices = std::make_index_sequence<std::variant_size_v<ArgumentTypes>>>
  void emplaceOperatorForEachArgumentTypesVariant(std::string const& name) {
    emplaceOperatorForEachArgumentTypesVariant<Operator, ArgumentTypes>(name, Indices{});
  }

public:
  template <template <typename...> typename Operator> auto emplace(std::string const& name) {
    emplaceOperatorForEachArgumentTypesVariant<Operator, typename Operator<>::ArgumentTypesT>(name);
  }
};

class Engine : public boss::Engine {
  static OperatorDirectory& getOperatorDirectory();

public:
  Engine(Engine&) = delete;
  Engine& operator=(Engine&) = delete;
  Engine(Engine&&) = default;
  Engine& operator=(Engine&&) = delete;
  Engine() = default;
  ~Engine() = default;

  template <template <typename...> typename Op> struct Register {
    explicit Register(char const* name) { Engine::getOperatorDirectory().emplace<Op>(name); }
  };

  boss::Expression evaluate(boss::Expression const& e);
  boss::Expression evaluate(boss::ComplexExpression const& e) {
    return evaluate(boss::Expression(e.clone()));
  }

  template <typename Func> static void evaluateInternal(Expression const& e, Func&& func) {
    auto cachedOpPtr = std::unique_ptr<CallableOperator>(nullptr);
    double cachedOpShape = -1.0;
    auto pushUp = [&func](auto&& bulkExpression, bool evaluated) {
      func(std::forward<decltype(bulkExpression)>(bulkExpression), evaluated);
    };
    Engine::evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp, cachedOpPtr,
                            cachedOpShape);
  }

  template <typename T>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>, Expression>
  evaluateInternal(T&& e) {
    return std::visit([](auto&& val) { return evaluateInternal(std::forward<decltype(val)>(val)); },
                      std::forward<decltype(e)>(e));
  }

  template <typename T>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>,
                          Expression>
  evaluateInternal(T&& e) {
    auto cachedOpPtr = std::unique_ptr<CallableOperator>(nullptr);
    double cachedOpShape = -1.0;
    return evaluateInternal(std::forward<decltype(e)>(e), cachedOpPtr, cachedOpShape);
  }

private:
  template <typename Func>
  static ExpressionArguments
  evaluateAndConsumeArguments(Func const& consume, ExpressionArguments&& args, size_t argStartIndex,
                              int iteratingPartitionsArgIndex = -1);

  friend CallableOperator; // for evaluateInternal, evaluateSymbols

  template <typename T,
            typename = std::enable_if_t<std::conjunction_v<
                std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
                std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>>
  static Expression evaluateInternal(T&& e) {
    std::optional<Expression> output;
    auto pushUp = [&output](auto&& bulkExpression, bool /*evaluated*/) {
      if constexpr(std::is_lvalue_reference_v<decltype(bulkExpression)>) {
        output = bulkExpression.clone();
      } else {
        output = std::forward<decltype(bulkExpression)>(bulkExpression);
      }
    };
    Engine::evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp);
    if(!output) {
      return true; // default return value when operators have no output
    }
    return std::move(*output);
  }

  template <typename T, typename = std::enable_if_t<std::disjunction_v<
                            std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>,
                            std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
  static Expression evaluateInternal(T&& e, std::unique_ptr<CallableOperator>& cachedOpPtr,
                                     double& cachedOpShape) {
    std::optional<Expression> output;
    auto pushUp = [&output](auto&& bulkExpression, bool /*evaluated*/) {
      if constexpr(std::is_lvalue_reference_v<decltype(bulkExpression)>) {
        output = bulkExpression.clone();
      } else {
        output = std::forward<decltype(bulkExpression)>(bulkExpression);
      }
    };
    Engine::evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp, cachedOpPtr,
                            cachedOpShape);
    if(!output) {
      return true; // default return value when operators have no output
    }
    return std::move(*output);
  }

  template <typename T, typename Func>
  static std::enable_if_t<std::conjunction_v<
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
  evaluateSymbols(T&& e, Func const& pushUp, int iteratingPartitionsArgIndex = -1) {
    evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp, iteratingPartitionsArgIndex);
  }

  template <typename T, typename Func>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>>
  evaluateSymbols(T&& e, Func const& pushUp, int iteratingPartitionsArgIndex = -1) {
    return std::visit(
        [&pushUp, &iteratingPartitionsArgIndex](auto&& val) {
          evaluateSymbols(std::forward<decltype(val)>(val), pushUp, iteratingPartitionsArgIndex);
        },
        std::forward<decltype(e)>(e));
  }

  template <typename T, typename Func>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>
  evaluateSymbols(T&& e, Func const& pushUp, int iteratingPartitionsArgIndex = -1) {
    auto cachedOpPtr = std::unique_ptr<CallableOperator>(nullptr);
    double cachedOpShape = -1.0;
    evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp, cachedOpPtr, cachedOpShape,
                    iteratingPartitionsArgIndex);
    if(cachedOpPtr != nullptr) {
      // reset any internal state
      cachedOpPtr->close();
    }
  }

  template <typename T>
  static std::enable_if_t<std::conjunction_v<
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
      std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
  evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                  OperatorPushUpMoveFunction const& pushUpMove,
                  int iteratingPartitionsArgIndex = -1);

  template <typename T>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>>
  evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                  OperatorPushUpMoveFunction const& pushUpMove,
                  std::unique_ptr<CallableOperator>& cachedOpPtr, double& cachedOpShape,
                  int iteratingPartitionsArgIndex = -1);

  template <typename T>
  static std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>
  evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                  OperatorPushUpMoveFunction const& pushUpMove,
                  std::unique_ptr<CallableOperator>& cachedOpPtr, double& cachedOpShape,
                  int iteratingPartitionsArgIndex = -1);
};

template <template <typename...> typename Subclass,
          typename... AcceptableTypes> // one per argument
class Operator : public CallableOperator {
public:
  using CallableOperator::CallableOperator;

  bool operator()(ExpressionArguments const& args) override {
    return (*this)(args, std::index_sequence_for<AcceptableTypes...>{});
  }

  template <typename T> T const& get_if_needed(Expression const& expr) {
    if constexpr(std::is_same_v<T, Expression>) {
      return expr;
    } else {
      return std::get<T>(expr);
    }
  }

  template <size_t... I>
  bool operator()(ExpressionArguments const& args, std::index_sequence<I...> /*unused*/) {
    if constexpr(sizeof...(I) > 0) {
      return callOperator(get_if_needed<AcceptableTypes>(args.at(I))...);
    } else {
      throw std::logic_error("not supported to call an operator with 0 arguments");
    }
  };

  std::unique_ptr<CallableOperator>
  getCallableInstance(OperatorPushUpFunction const& pushUp,
                      OperatorPushUpMoveFunction const& pushUpMove) const override {
    return std::make_unique<Subclass<AcceptableTypes...>>(pushUp, pushUpMove);
  }

  ~Operator() override = default;
  Operator(Operator&) = delete;
  Operator& operator=(Operator&) = delete;
  Operator(Operator&&) noexcept = default;
  Operator& operator=(Operator&&) = delete;
  Operator() = default;

private:
  template <typename... Args> bool callOperator(Args const&... args) {
    auto& operatorImpl = (Subclass<AcceptableTypes...>&)*this;
    if(!operatorImpl.checkArguments(args...)) {
      return false;
    }
    operatorImpl(args...);
    operatorImpl.rewindCachedInfoPosition();
    return true;
  }
};

template <typename T>
std::enable_if_t<
    std::conjunction_v<
        std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
        std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>,
    Expression>
CallableOperator::evaluateInternal(T&& e) {
  return Engine::evaluateInternal(std::forward<decltype(e)>(e));
}

template <typename T>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>, Expression>
CallableOperator::evaluateInternal(T&& e) {
  return std::visit(
      [this](auto&& val) { return evaluateInternal(std::forward<decltype(val)>(val)); },
      std::forward<decltype(e)>(e));
}

template <typename T>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>, Expression>
CallableOperator::evaluateInternal(T&& e) {
  if(cachedEvaluatorInfoIt == cachedEvaluatorInfo.end()) {
    cachedEvaluatorInfoIt = cachedEvaluatorInfo.emplace(
        cachedEvaluatorInfoIt, std::make_pair(std::unique_ptr<CallableOperator>(nullptr), -1.0));
  }
  auto& [cachedOpPtr, cachedOpShape] = *cachedEvaluatorInfoIt;
  ++cachedEvaluatorInfoIt;
  return Engine::evaluateInternal(std::forward<decltype(e)>(e), cachedOpPtr, cachedOpShape);
}

template <typename T, typename Func>
std::enable_if_t<std::conjunction_v<
    std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
    std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
CallableOperator::evaluateInternal(T&& e, Func&& func) {
  auto pushUp = [&func](auto&& bulkExpression, bool evaluated) {
    func(std::forward<decltype(bulkExpression)>(bulkExpression), evaluated);
  };
  Engine::evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp);
}

template <typename T, typename Func>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>>
CallableOperator::evaluateInternal(T&& e, Func&& func) {
  std::visit(
      [this, &func](auto&& val) {
        evaluateInternal(std::forward<decltype(val)>(val), std::forward<decltype(func)>(func));
      },
      std::forward<decltype(e)>(e));
}

template <typename T, typename Func>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>
CallableOperator::evaluateInternal(T&& e, Func&& func) {
  auto pushUp = [&func](auto&& bulkExpression, bool evaluated) {
    func(std::forward<decltype(bulkExpression)>(bulkExpression), evaluated);
  };
  if(cachedEvaluatorInfoIt == cachedEvaluatorInfo.end()) {
    cachedEvaluatorInfoIt = cachedEvaluatorInfo.emplace(
        cachedEvaluatorInfoIt, std::make_pair(std::unique_ptr<CallableOperator>(nullptr), -1.0));
  }
  auto& [cachedOpPtr, cachedOpShape] = *cachedEvaluatorInfoIt;
  ++cachedEvaluatorInfoIt;
  Engine::evaluateSymbols(std::forward<decltype(e)>(e), pushUp, pushUp, cachedOpPtr, cachedOpShape);
}

} // namespace boss::engines::bulk
