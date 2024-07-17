#pragma once

#include <arrow/util/string_view.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif // __AVX2__ || __AVX512F__

namespace boss::engines::bulk {

class Utilities {
public:
  /**
   * a RangeGenerator which expose the same interface as an arrow::Array
   * it can be used to generate an array of constant values which fits more efficiently in memory
   */
  template <typename ValueType,
            typename StoredType = std::conditional_t<std::is_same_v<ValueType, std::string> ||
                                                         std::is_same_v<ValueType, Symbol>,
                                                     arrow::util::string_view, ValueType>>
  class RangeGenerator {
  public:
    using ElementType = ValueType;

    // generate an array for a single repeating value
    template <typename T> RangeGenerator(int64_t size, T const& value) : size(size), value(value) {}
    RangeGenerator(int64_t size, Symbol const& value) : size(size), value(value.getName()) {}

    // arrow::Array interface
    auto const& Value(int64_t /*i*/) const { return value; }
    auto const& operator[](int64_t i) const { return Value(i); }
    int64_t length() const { return size; }
    auto const* raw_values() const { return &value; }

    // create 8 repeating 32bits elements
    // (expect only source elements which are 32bits)
    auto rawSIMD32(int64_t /*offset*/) const {
#ifdef __AVX2__
      if constexpr(std::is_same_v<ElementType, float_t>) {
        return _mm256_set1_ps(value);
      } else if constexpr(std::is_same_v<ElementType, int32_t>) {
        return _mm256_set1_epi32(value);
      } else
#endif //__AVX2__
      {
        return 0;
      }
    }

    // extract 4 64bits elements
    // (convert to 64bits if the source elements are 32bits)
    auto rawSIMD64(int64_t /*offset*/) const {
#ifdef __AVX2__
      if constexpr(std::is_same_v<ElementType, double_t>) {
        // return 64bits floats
        return _mm256_set1_pd(value);
      } else if constexpr(std::is_same_v<ElementType, float_t>) {
        // convert to 64bits floats
        auto packed32 = _mm_set1_ps(value);
        return _mm256_cvtps_pd(packed32);
      } else if constexpr(std::is_same_v<ElementType, int64_t>) {
        // return 64bits ints
        return _mm256_set1_epi64x(value);
      } else if constexpr(std::is_same_v<ElementType, int32_t>) {
        // convert to 64bits ints
        auto packed32 = _mm_set1_epi32(value);
        return _mm256_cvtepi32_epi64(packed32);
      } else
#endif //__AVX2__
      {
        return 0;
      }
    }

    // only for arrow::StringArray
    template <typename T = ElementType>
    std::enable_if_t<std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>, int64_t>
    total_values_length() const {
      return value.length();
    }

  private:
    int64_t size;
    StoredType value;
  };
};

} // namespace boss::engines::bulk
