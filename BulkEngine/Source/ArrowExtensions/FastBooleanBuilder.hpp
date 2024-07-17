#pragma once
#include <arrow/builder.h>

namespace boss::engines::bulk {
/**
 * an extension to arrow::BooleanBuilder
 * which exposes functions to work directly
 * on the underlined bit pack representation of both BooleanBuilder and BooleanArray
 * for faster read/write access of booleans during bulk computations
 */
class FastBooleanBuilder : public arrow::BooleanBuilder {
public:
  using arrow::BooleanBuilder::BooleanBuilder;

  // to be used with any operator returning a bool type (e.g. comparisons etc)
  template <typename Func, typename... ArrayType>
  void compute(Func&& func, ArrayType const&... inputArrays) {
    auto* output = data_builder_.mutable_data();
    constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
    int64_t outputIndex = 0;
    auto const outputLength = length_ / bitsPerPack;
    for(; outputIndex < outputLength; ++outputIndex) {
      auto inputIndex = outputIndex * bitsPerPack;
      output[outputIndex] = 0;
      for(auto i = 0U; i < bitsPerPack; ++i) {
        output[outputIndex] |= (unsigned int)func(inputArrays[inputIndex + i]...) << i;
      }
    }
    auto inputIndex = outputIndex * bitsPerPack;
    if(length_ > inputIndex) {
      auto remaining = length_ - inputIndex;
      output[outputIndex] = 0;
      for(auto i = 0U; i < remaining; ++i) {
        output[outputIndex] |= (unsigned int)func(inputArrays[inputIndex + i]...) << i;
      }
    }
  }

  // to be used with any bitwise operator
  // which is even faster since the operator can be applied directly on the packed bits
  template <typename Func, typename... ArrayType>
  void computeBitwise(Func&& bitwiseFunc, ArrayType const&... inputArrays) {
    [this, bitwiseFunc](auto const*... rawInputArrays) {
      auto* output = data_builder_.mutable_data();
      constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
      auto const packedLength = length_ / bitsPerPack;
      for(int64_t index = 0; index < packedLength; ++index) {
        output[index] = bitwiseFunc(rawInputArrays[index]...);
      }
      if(length_ > packedLength * bitsPerPack) {
        output[packedLength] = bitwiseFunc(rawInputArrays[packedLength]...);
      }
    }(inputArrays.values()->data()...);
  }

  // to be used for re-ordering an input array (e.g. sorts and joins)
  // assuming that the destination array has already been resized and initialised to 0
  template <typename IndexArrayType>
  void appendInIndexedOrder(arrow::BooleanArray const& inputArray, IndexArrayType const& rowIndices,
                            int64_t offset = 0) {
    auto* output = data_builder_.mutable_data();
    constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
    auto const outputLength = (length_ - offset) / bitsPerPack;
    int64_t outputIndex = offset / bitsPerPack;
    auto packIndex = outputIndex * bitsPerPack;
    if(offset > packIndex) {
      auto startingIndices = offset - packIndex;
      for(auto i = 0U; i < startingIndices; ++i) {
        output[outputIndex] |= (unsigned int)(inputArray.Value(rowIndices[i]))
                               << (unsigned int)(i + bitsPerPack - startingIndices);
      }
      ++outputIndex;
    }
    for(; outputIndex < outputLength; ++outputIndex) {
      auto inputIndex = outputIndex * bitsPerPack - offset;
      output[outputIndex] = 0;
      for(auto i = 0U; i < bitsPerPack; ++i) {
        output[outputIndex] |= (unsigned int)(inputArray.Value(rowIndices[inputIndex + i])) << i;
      }
    }
    auto inputIndex = outputIndex * bitsPerPack;
    if(length_ > inputIndex) {
      auto remaining = length_ - inputIndex;
      output[outputIndex] = 0;
      for(auto i = 0U; i < remaining; ++i) {
        output[outputIndex] |= (unsigned int)(inputArray.Value(rowIndices[inputIndex + i])) << i;
      }
    }
  }

  // to be used for filtering an input array (e.g. selection)
  // assuming that the destination array has already been resized and initialised to 0
  void appendWithCondition(arrow::BooleanArray const& inputArray,
                           arrow::BooleanArray const& conditionArray, int64_t offset = 0) {
    auto* output = data_builder_.mutable_data();
    auto const* input = inputArray.values()->data();
    auto const* condition = conditionArray.values()->data();
    constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
    int64_t outputIndex = offset / bitsPerPack;
    auto outputBitOffset = 0U;
    auto packIndex = outputIndex * bitsPerPack;
    if(offset > packIndex) {
      outputBitOffset = offset - packIndex;
    }
    int64_t inputIndex = 0;
    auto inputLength = inputArray.length();
    auto const packLength = inputLength / bitsPerPack;
    for(; inputIndex < packLength; ++inputIndex) {
      if(condition[inputIndex] != 0U) {
        for(auto i = 0U; i < bitsPerPack; ++i) {
          if((condition[inputIndex] & (1U << i)) != 0U) {
            output[outputIndex] |= (input[inputIndex] & (1U << i)) << outputBitOffset;
            outputIndex += (outputBitOffset + 1) >> 3U;   // +1 if about to wrap     // NOLINT
            outputBitOffset = (outputBitOffset + 1) & 7U; // wrap at 8               // NOLINT
          }
        }
      }
    }
    auto inputBitIndex = inputIndex * bitsPerPack;
    if(inputLength > inputBitIndex) {
      auto remaining = inputLength - inputBitIndex;
      if(condition[inputIndex] != 0U) {
        for(auto i = 0U; i < remaining; ++i) {
          if((condition[inputIndex] & (1U << i)) != 0U) {
            output[outputIndex] |= (input[inputIndex] & (1U << i)) << outputBitOffset;
            outputIndex += (outputBitOffset + 1) >> 3U;   // +1 if about to wrap     // NOLINT
            outputBitOffset = (outputBitOffset + 1) & 7U; // wrap at 8               // NOLINT
          }
        }
      }
    }
  }
};

} // namespace boss::engines::bulk
