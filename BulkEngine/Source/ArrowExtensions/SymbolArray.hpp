#pragma once

#include <Expression.hpp>

#include <arrow/array/array_binary.h>
#include <arrow/array/array_dict.h>
#include <arrow/array/builder_dict.h>
#include <arrow/extension_type.h>

#include <string>

namespace boss::engines::bulk {

/** We use SymbolArray to store symbols.
 * There is no particular feature we implement on the top of the string array.
 * The main purpose is to be able to distingish a string array from a symbol array.
 * But we could possibly make it more specific in the feature (such as using dictionary).*/
class SymbolArray : public arrow::StringArray {
private:
  std::shared_ptr<arrow::ArrayData>
  static stringTypeToSymbolType(std::shared_ptr<arrow::ArrayData> const& data) {
    auto adjustedData = data->Copy();
    adjustedData->type = std::make_shared<SymbolType>();
    return adjustedData;
  }
  std::shared_ptr<arrow::ArrayData>
  static symbolTypeToStringType(std::shared_ptr<arrow::ArrayData> const& data) {
    auto adjustedData = data->Copy();
    adjustedData->type = std::make_shared<arrow::StringType>();
    return adjustedData;
  }

public:
  explicit SymbolArray(const std::shared_ptr<arrow::ArrayData>& data)
      : arrow::StringArray(data->type->id() == arrow::Type::STRING ? data
                                                                   : symbolTypeToStringType(data)) {
    if(data->type->id() == arrow::Type::STRING) {
      // make sure to set back the extension type after the end of call from base array class
      SetData(stringTypeToSymbolType(data));
    }
  }

  Symbol Value(size_t index) { return Symbol(std::string(GetView(index))); }

  /** Custom type to implement an Arrow array for symbols.
   * This is mostly boilerplate code to be compliant with Arrow. */
  class SymbolType : public arrow::ExtensionType {
  public:
    explicit SymbolType() : ExtensionType(arrow::utf8()) {}

    std::shared_ptr<arrow::Array> MakeArray(std::shared_ptr<arrow::ArrayData> data) const override {
      // temporarly change to the underline type for the construction
      // it will be reverted in the SymbolArray constructor
      auto adjustedData = data->Copy();
      adjustedData->type = arrow::utf8();
      return std::make_shared<SymbolArray>(adjustedData);
    }

    ///////////////////////////////////////////////////////////////////////
    // code required by Arrow to implement an extension type
    std::string extension_name() const override { return "symbol-type"; }
    bool ExtensionEquals(const ExtensionType& other) const override {
      const auto& other_ext = static_cast<const ExtensionType&>(other);
      return other_ext.extension_name() == this->extension_name();
    }
    arrow::Result<std::shared_ptr<DataType>>
    Deserialize(std::shared_ptr<DataType> /*storage_type*/,
                const std::string& /*serialized*/) const override {
      return std::make_shared<SymbolType>();
    }
    std::string Serialize() const override { return std::string(); }
    ///////////////////////////////////////////////////////////////////////
  };
};

} // namespace boss::engines::bulk
