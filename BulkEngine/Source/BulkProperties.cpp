#include "BulkProperties.hpp"
#include "SymbolRegistry.hpp"

namespace boss::engines::bulk {

template <typename T>
T const& Properties::initialiseOrGetProperty(Symbol const& name, T&& defaultValue) {
  auto const& ptr = [&name, &defaultValue]() -> std::unique_ptr<Expression> const& {
    auto& symbolRegistry = DefaultSymbolRegistry::globalInstance();
    auto& ptr = symbolRegistry.findOrCreateSymbolReference(name);
    if(!ptr) {
      ptr.reset(new Expression(defaultValue));
    }
    return ptr;
  }();
  return std::get<T>(*ptr);
}

template bool const& Properties::initialiseOrGetProperty<bool>(Symbol const& name,
                                                               bool&& defaultValue);
template int64_t const& Properties::initialiseOrGetProperty<int64_t>(Symbol const& name,
                                                                     int64_t&& defaultValue);
template double_t const& Properties::initialiseOrGetProperty<double_t>(Symbol const& name,
                                                                       double_t&& defaultValue);
template std::string const&
Properties::initialiseOrGetProperty<std::string>(Symbol const& name, std::string&& defaultValue);

} // namespace boss::engines::bulk