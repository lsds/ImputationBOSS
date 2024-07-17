#pragma once
#include "BulkExpression.hpp"
#include <arrow/table.h>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace boss::engines::bulk {

/** Keep any type of expression stored and mapped to a symbol. */
template <typename StoredType, typename... DependentRegistries> class SymbolRegistry {
private:
  SymbolRegistry() : threadId(std::this_thread::get_id()) {
    (DependentRegistries::instance().setCallback(this, [this]() { this->clear(); }), ...);
  }

public:
  using StoredTypePtr = std::unique_ptr<StoredType>;

  ~SymbolRegistry() {
    // clear other registries depending on this SymbolRegistry first
    for(auto& [owner, func] : destructionCallbacks) {
      func();
    }
    // release all the symbols before deallocating the map
    // (otherwise table destructor might crash while accessing the map)
    for(auto& [key, valuePtr] : symbolMap) {
      valuePtr.reset(nullptr);
    }
    // then unregister callback if this SymbolRegistry has dependent registries
    (DependentRegistries::instance().clearCallback(this), ...);
  }
  SymbolRegistry(SymbolRegistry const& other) = delete;
  SymbolRegistry(SymbolRegistry&& other) = delete;
  SymbolRegistry& operator=(SymbolRegistry const& other) = delete;
  SymbolRegistry& operator=(SymbolRegistry&& other) = delete;

  static SymbolRegistry& instance() { return getInstance<false>(); }
  static SymbolRegistry& globalInstance() { return getInstance<true>(); }

  template <bool global> static SymbolRegistry& getInstance() {
    static std::thread::id mainThreadId = std::this_thread::get_id();
    thread_local bool isMainThread = std::this_thread::get_id() == mainThreadId;
    if constexpr(!global) {
      if(!isMainThread) {
        thread_local SymbolRegistry instance;
        return instance;
      }
    }
    static SymbolRegistry instance;
    return instance;
  }

  std::thread::id ownerThreadId() const { return threadId; }

  template <typename Func> void setCallback(void* owner, Func&& func) {
    destructionCallbacks[owner] = std::forward<decltype(func)>(func);
  }
  void clearCallback(void* owner) { destructionCallbacks.erase(owner); }

  void registerSymbol(Symbol const& symbol, StoredTypePtr&& value) {
    symbolMap[symbol.getName()] = std::move(value);
  }

  template <typename... Args> StoredType& registerSymbol(Symbol const& symbol, Args&&... args) {
    auto const& foundIt = symbolMap.find(symbol.getName());
    if(foundIt == symbolMap.end()) {
      auto storedPtr = std::make_unique<StoredType>(std::move(args)...);
      auto* storedRef = storedPtr.get();
      symbolMap[symbol.getName()] = std::move(storedPtr);
      return *storedRef;
    }
    auto& ptr = foundIt->second;
    if constexpr(sizeof...(args) == 1 && std::is_assignable_v<StoredType, decltype(args)...>) {
      *(ptr.get()) = (std::move(args), ...);
    } else {
      ptr.reset(new StoredType(std::move(args)...));
    }
    return *(ptr.get());
  }

  /* replace the stored symbol and return the pointer to the old value */
  template <typename... Args> StoredTypePtr swapSymbol(Symbol const& symbol, Args&&... args) {
    auto& storedSymbol = symbolMap[symbol.getName()];
    auto oldSymbol = std::move(storedSymbol);
    storedSymbol = std::make_unique<StoredType>(std::move(args)...);
    return oldSymbol;
  }

  void clearSymbol(Symbol const& symbol) { symbolMap[symbol.getName()].reset(); }

  StoredType* findSymbol(Symbol const& symbol) {
    auto const& foundIt = symbolMap.find(symbol.getName());
    return foundIt != symbolMap.end() ? foundIt->second.get() : nullptr;
  }

  StoredTypePtr& findOrCreateSymbolReference(Symbol const& symbol) {
    auto const& foundIt = symbolMap.find(symbol.getName());
    if(foundIt != symbolMap.end()) {
      return foundIt->second;
    }
    return symbolMap[symbol.getName()];
  }

  void clear() { symbolMap.clear(); }

private:
  using SymbolMapping = std::unordered_map<std::string, StoredTypePtr>;
  SymbolMapping symbolMap;

  std::map<void*, std::function<void(void)>> destructionCallbacks;

  std::thread::id threadId;
};

using DefaultSymbolRegistry = SymbolRegistry<Expression>;

} // namespace boss::engines::bulk
