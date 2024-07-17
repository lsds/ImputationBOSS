#include "Bulk.hpp"
#include "BulkOperators.hpp"
#include "DDLOperators.hpp"
#include "ImputationOperators.hpp"
#include "ParallelOperators.hpp"
#include "RelationalOperators.hpp"
#include "SerializationOperators.hpp"

#include <condition_variable>
#include <mutex>

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

namespace boss::engines::bulk {

OperatorDirectory& Engine::getOperatorDirectory() {
  static OperatorDirectory operatorDirectory;
  return operatorDirectory;
}

static auto& getCurrentStoredSymbol() {
  static auto currentlyStoredSymbol = std::optional<Symbol>();
  return currentlyStoredSymbol;
}

static boss::Expression toBossExpression(Expression&& bulkExpression) {
  static constexpr int MAX_NUM_ROWS_TO_RETURN = 10;
  auto symbolReplacementForLargeRelation = [](auto& partitionsPtr) {
    // create a unique name (+ some useful info)
    static int i = 0;
    auto symbolName = "_table"s + std::to_string(i++);
    auto beginIt = partitionsPtr->begin();
    auto endIt = partitionsPtr->end();
    if(!*beginIt) {
      ++beginIt;
    }
    auto numCols = (*beginIt)->num_fields();
    auto numRows = std::accumulate(beginIt, endIt, 0, [](auto total, auto const& partition) {
      return total + partition->length();
    });
    symbolName += "_cols"s + std::to_string(numCols) + "rows"s + std::to_string(numRows);
    auto relationSymbol = Symbol(symbolName);
    // store it as a symbol, replace any previous stored one
    if(getCurrentStoredSymbol()) {
      DefaultSymbolRegistry::globalInstance().clearSymbol(*getCurrentStoredSymbol());
    }
    getCurrentStoredSymbol() = relationSymbol;
    DefaultSymbolRegistry::globalInstance().registerSymbol(relationSymbol, partitionsPtr);
    return relationSymbol;
  };
  return std::visit(
      boss::utilities::overload(
          [&](ComplexExpression&& e) -> boss::Expression {
            boss::ExpressionArguments bossArgs;
            auto fromArgs = e.getArguments();
            bossArgs.reserve(fromArgs.size());
            std::transform(std::make_move_iterator(fromArgs.begin()),
                           std::make_move_iterator(fromArgs.end()), std::back_inserter(bossArgs),
                           [](auto&& bulkArg) {
                             return toBossExpression(std::forward<decltype(bulkArg)>(bulkArg));
                           });
            return boss::ComplexExpression(e.getHead(), std::move(bossArgs));
          },
          [&symbolReplacementForLargeRelation](
              Table::PartitionPtr const& partition) -> boss::Expression {
#ifdef DEBUG_OUTPUT_OPS
            if(Properties::debugOutputRelationalOps()) {
              std::cerr << "output: ";
              outputDebugPartition(partition);
              std::cerr << std::endl;
              clearDebugPartitions();
              DEBUG_OUTPUT_RELATION_OPS_DEPTH = 0;
            }
#endif // DEBUG_OUTPUT_OPS
            if(!partition) {
              return boss::ComplexExpression("List"_, {});
            }
            if(partition->length() > MAX_NUM_ROWS_TO_RETURN) {
              auto partitionsToStore = std::make_shared<Table::PartitionVector>(1, partition);
              return symbolReplacementForLargeRelation(partitionsToStore);
            }
            return (boss::Expression)*partition; // explicit conversion
          },
          [&symbolReplacementForLargeRelation](
              Table::PartitionVectorPtr const& partitionsPtr) -> boss::Expression {
#ifdef DEBUG_OUTPUT_OPS
            if(Properties::debugOutputRelationalOps()) {
              std::cerr << "output: ";
              outputDebugPartitions(partitionsPtr);
              std::cerr << std::endl;
              clearDebugPartitions();
              DEBUG_OUTPUT_RELATION_OPS_DEPTH = 0;
            }
#endif // DEBUG_OUTPUT_OPS
            boss::ExpressionArguments allRows;
            if(partitionsPtr) {
              auto const& partitions = *partitionsPtr;
              for(auto const& partition : partitions) {
                if(!partition) {
                  continue;
                }
                if(allRows.size() + partition->length() > MAX_NUM_ROWS_TO_RETURN) {
                  return symbolReplacementForLargeRelation(partitionsPtr);
                }
                auto asExpression = (boss::Expression)*partition;
                auto rows = std::get<boss::ComplexExpression>(asExpression).getArguments();
                allRows.insert(allRows.end(), std::make_move_iterator(rows.begin()),
                               std::make_move_iterator(rows.end()));
              }
            }
            return boss::ComplexExpression("List"_, std::move(allRows));
          },
          [](auto const& otherTypes) -> boss::Expression {
            return (boss::Expression)otherTypes; // explicit conversion
          }),
      std::move(bulkExpression));
}

static bool findSymbolInExpression(boss::Expression const& expr, Symbol const& symbol) {
  return std::visit(
      boss::utilities::overload(
          [&symbol](boss::ComplexExpression const& e) {
            auto const& args = e.getDynamicArguments();
            return std::accumulate(std::make_move_iterator(args.begin()),
                                   std::make_move_iterator(args.end()), false,
                                   [&symbol](bool found, auto const& arg) {
                                     return found || findSymbolInExpression(arg, symbol);
                                   });
          },
          [&symbol](Symbol const& s) { return s == symbol; }, [&](auto const& e) { return false; }),
      expr);
}

static bool isAQueryExpression(boss::Expression const& expr) {
  return std::visit(
      boss::utilities::overload(
          [](boss::ComplexExpression const& e) {
            if(e.getHead().getName() == "Select" || e.getHead().getName() == "Join" ||
               e.getHead().getName() == "Project" || e.getHead().getName() == "Group" ||
               e.getHead().getName() == "GroupBy" || e.getHead().getName() == "Sort" ||
               e.getHead().getName() == "SortBy" || e.getHead().getName() == "Order" ||
               e.getHead().getName() == "OrderBy" || e.getHead().getName() == "Top") {
              return true;
            }
            auto const& args = e.getDynamicArguments();
            return std::accumulate(std::make_move_iterator(args.begin()),
                                   std::make_move_iterator(args.end()), false,
                                   [](bool isQuery, auto const& arg) {
                                     return isQuery || isAQueryExpression(arg);
                                   });
          },
          [](auto const& e) { return false; }),
      expr);
}

boss::Expression Engine::evaluate(boss::Expression const& expr) { // NOLINT
  try {
    // free the previously stored symbol if not using it anymore
    if(getCurrentStoredSymbol()) {
      if(isAQueryExpression(expr) && !findSymbolInExpression(expr, *getCurrentStoredSymbol())) {
        DefaultSymbolRegistry::globalInstance().clearSymbol(*getCurrentStoredSymbol());
        getCurrentStoredSymbol().reset();
      }
    }
    // do the evaluation
    std::optional<Expression> output;
    Engine::evaluateSymbols(
        Expression(expr.clone()), [&output](auto&& bulkExpression, bool /*evaluated*/) {
          if(output) {
            // accumulate partitions to the output
            // assuming that operators pushing multiple outputs always return partitions
            if(auto* partitionPtr = std::get_if<Table::PartitionPtr>(&*output)) {
              output = std::make_shared<Table::PartitionVector>(1, *partitionPtr);
            }
            auto partitions = std::get<Table::PartitionVectorPtr>(*output);
            if(auto const* newPartitionPtr = std::get_if<Table::PartitionPtr>(&bulkExpression)) {
              if(*newPartitionPtr) {
                partitions->emplace_back(*newPartitionPtr);
              }
            } else {
              for(auto& partitionPtr : *std::get<Table::PartitionVectorPtr>(bulkExpression)) {
                if(partitionPtr) {
                  partitions->emplace_back(partitionPtr);
                }
              }
            }
            return;
          }
          if constexpr(std::is_lvalue_reference_v<decltype(bulkExpression)>) {
            output = bulkExpression.clone();
          } else {
            output = std::forward<decltype(bulkExpression)>(bulkExpression);
          }
        });
    if(!output) {
      return true; // default return value when operators have no output
    }
    return toBossExpression(std::move(*output));
  } catch(std::exception const& e) {
    boss::ExpressionArguments args;
    args.reserve(2);
    args.emplace_back(expr.clone());
    args.emplace_back(std::string{e.what()});
    return boss::ComplexExpression{"ErrorWhenEvaluatingExpression"_, std::move(args)};
  }
}

template <typename Func>
ExpressionArguments
Engine::evaluateAndConsumeArguments(Func const& consume, ExpressionArguments&& args,
                                    size_t argStartIndex, int iteratingPartitionsArgIndex) {
  // 1. try if the operator can be called without further evaluation
  bool evaluated = consume(args);
  if(evaluated) {
    return std::move(args);
  }
  // 2. for multi-partition args, try to do separate operator calls per partition
  // (and recursively for each argument)
  for(auto i = args.size() - 1; i < args.size(); --i) {
    if(auto const* maybePartitions = std::get_if<Table::PartitionVectorPtr>(&args[i])) {
      auto partitionsPtr = *maybePartitions;
      auto const& partitions = *partitionsPtr;
      // don't iterate every combination of partitions.
      // instead, iterate all the 1st arg (with an empty 2nd arg),
      //             then all the 2nd arg (with an empty 1st arg)
      // for that, push an empty partition first for any partition-type argument
      args[i] = Table::PartitionPtr();
      args = Engine::evaluateAndConsumeArguments(consume, std::move(args), argStartIndex);
      if(iteratingPartitionsArgIndex >= 0 /* && iteratingPartitionsArgIndex != i*/) {
        // don't iterate on these partitions if we are already iterating another partition vector
        continue;
      }
      // all for each partition
      for(auto const& partition : partitions) {
        if(!partition) {
          continue;
        }
        args[i] = partition;
        args = Engine::evaluateAndConsumeArguments(consume, std::move(args), argStartIndex, i);
        // clear the partition argument once it gets evaluated
        // this way, if we have a 2nd partition argument calling this evaluation recursively
        // only the 1st time, the 1st partition argument will be non-empty
        // (without this code, it would instead repeat the last evaluated partition)
        args[i] = Table::PartitionPtr();
      }
      return std::move(args);
    }
  }
  // 3. finally, try to evaluate each argument (which haven't been evaluated yet)
  for(auto i = argStartIndex; i < args.size(); --i) {
    bool recursivelyCalled = false;
    Engine::evaluateSymbols(
        std::move(args[i]),
        [&](auto&& e, bool evaluated) {
          bool isPartition = std::holds_alternative<Table::PartitionPtr>(e) ||
                             std::holds_alternative<Table::PartitionVectorPtr>(e);
          if constexpr(std::is_lvalue_reference_v<decltype(e)>) {
            args[i] = e.clone();
          } else {
            args[i] = std::forward<decltype(e)>(e);
          }
          if(!evaluated) {
            return;
          }
          recursivelyCalled = true;
          if(!isPartition) {
            args = Engine::evaluateAndConsumeArguments(consume, std::move(args), i - 1,
                                                       iteratingPartitionsArgIndex);
          } else {
            auto const* maybePartitions = std::get_if<Table::PartitionPtr>(&args[i]);
            if(maybePartitions != nullptr && *maybePartitions) {
              iteratingPartitionsArgIndex = i;
            }
            args = Engine::evaluateAndConsumeArguments(consume, std::move(args), i - 1,
                                                       iteratingPartitionsArgIndex);
            // special case to clear the partition argument once it gets evaluated
            // this way, if we have a 2nd partition argument calling this evaluation recursively
            // only the 1st time, the 1st partition argument will be non-empty
            // (without this code, it would instead repeat the last evaluated partition)
            args[i] = Table::PartitionPtr();
          }
        },
        iteratingPartitionsArgIndex);
    if(recursivelyCalled) {
      return std::move(args);
    }
  }
  return std::move(args);
}

template <typename T>
std::enable_if_t<std::conjunction_v<
    std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>,
    std::negation<std::is_same<std::remove_cv_t<std::decay_t<T>>, Expression>>>>
Engine::evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                        OperatorPushUpMoveFunction const& pushUpMove,
                        int /*iteratingPartitionsArgIndex*/) {
  if constexpr(std::is_same_v<std::decay_t<T>, Symbol>) {
    auto* table = TableSymbolRegistry::globalInstance().findSymbol(e);
    if(table != nullptr) {
#ifdef DEBUG_OUTPUT_OPS
      if(Properties::debugOutputRelationalOps()) {
        for(int i = 0; i < DEBUG_OUTPUT_RELATION_OPS_DEPTH; ++i) {
          std::cerr << "  ";
        }
        std::cerr << "scan " << e.getName() << std::endl;
      }
      ++DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      pushUpMove(table->finaliseAndGetPartitions(true), true);
#ifdef DEBUG_OUTPUT_OPS
      --DEBUG_OUTPUT_RELATION_OPS_DEPTH;
#endif // DEBUG_OUTPUT_OPS
      return;
    }
    auto* knownSymbol = DefaultSymbolRegistry::instance().findSymbol(e);
    if(knownSymbol != nullptr) {
      pushUp(*knownSymbol, true);
      return;
    }
    auto* knownGlobalSymbol = DefaultSymbolRegistry::globalInstance().findSymbol(e);
    if(knownGlobalSymbol != nullptr) {
      pushUp(*knownGlobalSymbol, true);
      return;
    }
    pushUp(e, false);
  } else {
    // default: nothing to do
    pushUp(e, false);
  }
}

template <typename T>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, Expression>>
Engine::evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                        OperatorPushUpMoveFunction const& pushUpMove,
                        std::unique_ptr<CallableOperator>& cachedOpPtr, double& cachedOpShape,
                        int iteratingPartitionsArgIndex) {
  std::visit(boss::utilities::overload(
                 [&pushUp, &pushUpMove, &iteratingPartitionsArgIndex](auto&& val) {
                   evaluateSymbols(std::forward<decltype(val)>(val), pushUp, pushUpMove,
                                   iteratingPartitionsArgIndex);
                 },
                 [&pushUp, &pushUpMove, &cachedOpPtr, &cachedOpShape,
                  &iteratingPartitionsArgIndex](ComplexExpression&& cexpr) {
                   evaluateSymbols(std::move(cexpr), pushUp, pushUpMove, cachedOpPtr, cachedOpShape,
                                   iteratingPartitionsArgIndex);
                 },
                 [&pushUp, &pushUpMove, &cachedOpPtr, &cachedOpShape,
                  &iteratingPartitionsArgIndex](ComplexExpression const& cexpr) {
                   evaluateSymbols(cexpr, pushUp, pushUpMove, cachedOpPtr, cachedOpShape,
                                   iteratingPartitionsArgIndex);
                 }),
             std::forward<decltype(e)>(e));
}

template <typename T>
std::enable_if_t<std::is_same_v<std::remove_cv_t<std::decay_t<T>>, ComplexExpression>>
Engine::evaluateSymbols(T&& e, OperatorPushUpFunction const& pushUp,
                        OperatorPushUpMoveFunction const& pushUpMove,
                        std::unique_ptr<CallableOperator>& cachedOpPtr, double& cachedOpShape,
                        int iteratingPartitionsArgIndex) {
  auto&& head = [&e]() -> decltype(auto) {
    if constexpr(std::is_const_v<decltype(e)>) {
      return e.getHead();
    } else {
      return std::move(e.getHead());
    }
  }();
  auto const& name = head.getName();
  auto evaluate = [&pushUp, &pushUpMove, &cachedOpPtr, &cachedOpShape,
                   &name](auto&& args, bool keepExpressionArgs = false) {
    auto typeID = std::accumulate(
        args.rbegin(), args.rend(), 0., [&keepExpressionArgs](double id, auto const& arg) {
          return id * static_cast<double>(variant_size_v<Expression::SuperType> + 1) +
                 static_cast<double>(keepExpressionArgs
                                         ? (variant_size_v<Expression::SuperType> + 1)
                                         : (arg.index() + 1));
        });
    if(cachedOpPtr == nullptr) {
      auto foundIt = getOperatorDirectory().find({name, typeID});
      if(foundIt != getOperatorDirectory().end()) {
        cachedOpPtr = foundIt->second->getCallableInstance(pushUp, pushUpMove);
        cachedOpShape = typeID;
      }
    } else if(cachedOpShape != typeID) { // the operator arguments are supposed
                                         //  to be same types in successive calls
      auto foundIt = getOperatorDirectory().find({name, typeID});
      if(foundIt == getOperatorDirectory().end()) {
        // ok, this is the case where the arguments need to be evaluated further
        return false;
      }
      // now it can happen when calling from evaluateInternal()
      // throw std::logic_error("operator '" + name +
      //                       "' got different argument types in successive calls");
      cachedOpPtr = foundIt->second->getCallableInstance(pushUp, pushUpMove);
      cachedOpShape = typeID;
    } else {
      // reset the psuhUp functions in case they have changed
      cachedOpPtr->setPushUpCalls(pushUp, pushUpMove);
    }
    bool evaluated = false;
    if(cachedOpPtr != nullptr) {
      evaluated = (*cachedOpPtr)(args);
      if(!evaluated) {
        // handle the case where an operator return itself when failing to evaluate
        // e.g. when a complex expression other that "List" is passed to Length
        // ideally, we want to manage that prior to the operator call though...
        cachedOpPtr = nullptr;
      }
    }
    return evaluated;
  };
  ExpressionArguments args = e.getArguments();
  // try without resolving the types of the arguments
  bool evaluated = evaluate(args, true);
  if(!evaluated) {
    // otherwise try to evaluate the arguments first
    auto argStartIndex = args.size() - 1;
    args = Engine::evaluateAndConsumeArguments(evaluate, std::move(args), argStartIndex,
                                               iteratingPartitionsArgIndex);
  }
  if(cachedOpPtr != nullptr) {
    return;
  }
  // push at least the unevaluated argument
  pushUpMove(ComplexExpression(std::move(head), std::move(args)), false);
}

} // namespace boss::engines::bulk

static auto& enginePtr(bool initialise = true) {
  static auto engine = std::unique_ptr<boss::engines::bulk::Engine>();
  if(!engine && initialise) {
    engine.reset(new boss::engines::bulk::Engine());
  }
  return engine;
}

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  static std::mutex m;
  std::lock_guard lock(m);
  auto* r = new BOSSExpression{enginePtr()->evaluate(e->delegate.clone())};
  return r;
};

extern "C" void reset() { enginePtr(false).reset(nullptr); }
