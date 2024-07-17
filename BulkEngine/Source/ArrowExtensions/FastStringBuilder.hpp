#pragma once
#include <arrow/builder.h>

namespace boss::engines::bulk {
/**
 * an extension to arrow::StringBuilder
 * which allow faster concatenation of string data
 */
class FastStringBuilder : public arrow::StringBuilder {
public:
  using arrow::StringBuilder::StringBuilder;

  // append string data but without adding a new offset
  // (just concatenating to the previous string)
  void UnsafeConcatenate(arrow::util::string_view const& view) {
    UnsafeConcatenate(view.data(), static_cast<offset_type>(view.size()));
  }
  void UnsafeConcatenate(const char* value, offset_type length) {
    UnsafeConcatenate(reinterpret_cast<const uint8_t*>(value), length);
  }
  void UnsafeConcatenate(const uint8_t* value, offset_type length) {
    unsafeAppendImpl(value, length);
  }

  // exposing arrow functions which already handle checking the status
  // to avoid this boilerplate in the code
  using arrow::StringBuilder::Reserve;
  void Reserve(int64_t elements) {
    auto status = arrow::StringBuilder::Reserve(elements);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  using arrow::StringBuilder::ReserveData;
  void ReserveData(int64_t elements) {
    auto status = arrow::StringBuilder::ReserveData(elements);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  using arrow::StringBuilder::Append;
  void Append(const uint8_t* value, offset_type length) {
    auto status = arrow::StringBuilder::Append(value, length);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }

  // re-implement UnsafeAppend in a safer way
  void UnsafeAppend(uint8_t const* value, offset_type length) {
    UnsafeAppendNextOffset();
    unsafeAppendImpl(value, length);
    UnsafeAppendToBitmap(true);
  }
  void UnsafeAppend(const char* value, offset_type length) {
    UnsafeAppend(reinterpret_cast<const uint8_t*>(value), length);
  }
  void UnsafeAppend(arrow::util::string_view value) {
    UnsafeAppendNextOffset();
    while(value.size() > std::numeric_limits<typename arrow::StringBuilder::offset_type>::max()) {
      // internally, UnsafeAppend does a cast to offset_type
      // split the string to avoid crash
      auto subvalue =
          value.substr(0, std::numeric_limits<typename arrow::StringBuilder::offset_type>::max());
      unsafeAppendImpl(reinterpret_cast<const uint8_t*>(subvalue.data()),
                       static_cast<offset_type>(subvalue.size()));
      value = value.substr(std::numeric_limits<typename arrow::StringBuilder::offset_type>::max());
    }
    if(value.size() > 0) {
      unsafeAppendImpl(reinterpret_cast<const uint8_t*>(value.data()),
                       static_cast<offset_type>(value.size()));
    }
    UnsafeAppendToBitmap(true);
  }

private:
  void unsafeAppendImpl(uint8_t const* value, offset_type length) {
    // use memmove if there is overlap (https://stackoverflow.com/a/25630217)
    // (or avoid completely copy if dest is same as source)
    auto const* srcDataStart = value;
    auto const* srcDataEnd = srcDataStart + length;
    auto* destDataStart = value_data_builder_.bytes_builder()->mutable_data() +
                          value_data_builder_.bytes_builder()->length();
    auto* destDataEnd = destDataStart + length;
    if(((void*)destDataEnd >= (void*)srcDataStart && (void*)destDataEnd < (void*)srcDataEnd) ||
       ((void*)srcDataEnd >= (void*)destDataStart && (void*)srcDataEnd < (void*)destDataEnd)) {
      if((void*)srcDataStart != (void*)destDataStart) {
        memmove(destDataStart, srcDataStart, static_cast<size_t>(length));
      }
    } else {
      memcpy(destDataStart, srcDataStart, static_cast<size_t>(length));
    }
    value_data_builder_.bytes_builder()->UnsafeAdvance(length);
  }
};

} // namespace boss::engines::bulk
