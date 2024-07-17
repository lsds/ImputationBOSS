#pragma once
#include "ArrowExtensions/ArrayBuilderConversionBuffer.hpp"
#include "ArrowExtensions/FastBooleanBuilder.hpp"
#include "ArrowExtensions/FastStringBuilder.hpp"
#include "ArrowExtensions/SymbolArray.hpp"
#include "ArrowExtensions/SymbolBuilder.hpp"
#include "BulkProperties.hpp"
#include <ExpressionUtilities.hpp>
#include <arrow/array.h>
#include <arrow/builder.h>
using boss::utilities::operator""_;

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif // __AVX2__ || __AVX512F__

namespace boss::engines::bulk {
/**
 * Definition of the arrow arrays/builders we are using for each scalar type
 * Only ValueArray<T>, ValueArrayPtr<T>, ValueBuilder<T> should be used in the rest of the code
 */

template <typename T> struct ValueArrayType;
template <> struct ValueArrayType<bool> {
  using ArrowArray = arrow::BooleanArray;
  using ArrowBuilder = FastBooleanBuilder;
};
template <> struct ValueArrayType<int32_t> {
  using ArrowArray = arrow::Int32Array;
  using ArrowBuilder = arrow::Int32Builder;
};
template <> struct ValueArrayType<int64_t> {
  using ArrowArray = arrow::Int64Array;
  using ArrowBuilder = arrow::Int64Builder;
};
template <> struct ValueArrayType<float_t> {
  using ArrowArray = arrow::FloatArray;
  using ArrowBuilder = arrow::FloatBuilder;
};
template <> struct ValueArrayType<double_t> {
  using ArrowArray = arrow::DoubleArray;
  using ArrowBuilder = arrow::DoubleBuilder;
};
template <> struct ValueArrayType<std::string> {
  using ArrowArray = arrow::StringArray;
  using ArrowBuilder = FastStringBuilder;
};
template <> struct ValueArrayType<Symbol> {
  using ArrowArray = SymbolArray;
  using ArrowBuilder = SymbolBuilder;
};

template <typename T> class ValueArray : public ValueArrayType<T>::ArrowArray {
public:
  explicit ValueArray(arrow::Array const& base) : ValueArrayType<T>::ArrowArray(base.data()) {}
  explicit ValueArray(std::shared_ptr<arrow::ArrayData> const& baseData)
      : ValueArrayType<T>::ArrowArray(baseData) {}

  using ElementType = T;
  using RawType = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;

  // access raw data (for bulk inserts)
  RawType const* rawData() const {
    return reinterpret_cast<RawType const*>(this->raw_values_) +
           this->data_->offset * sizeof(RawType) / sizeof(T);
  }
  auto rawLength() const {
    if constexpr(std::is_same_v<T, bool>) {
      return this->data_->length / (sizeof(RawType) * CHAR_BIT);
    } else {
      return this->data_->length;
    }
  }

  // make a consistent function call for both numerical and string arrays
  auto Value(int64_t i) const {
    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
      return ValueArrayType<T>::ArrowArray::GetView(i);
    } else {
      return ValueArrayType<T>::ArrowArray::Value(i);
    }
  }

  auto operator[](int64_t i) const { return Value(i); }

  // extract 8 32bits elements
  // (expect only source elements which are 32bits)
  auto rawSIMD32(int64_t offset) const {
#ifdef __AVX2__
    if constexpr(std::is_same_v<ElementType, float_t>) {
      return _mm256_loadu_ps(ValueArrayType<T>::ArrowArray::raw_values() + offset);
    } else if constexpr(std::is_same_v<ElementType, int32_t>) {
      return _mm256_loadu_si256((__m256i*)(ValueArrayType<T>::ArrowArray::raw_values() + offset));
    } else
#endif //__AVX2__
    {
      return 0;
    }
  }

  // extract 4 64bits elements
  // (convert to 64bits if the source elements are 32bits)
  auto rawSIMD64(int64_t offset) const {
#ifdef __AVX2__
    if constexpr(std::is_same_v<ElementType, double_t>) {
      // return 64bits floats
      return _mm256_loadu_pd(ValueArrayType<T>::ArrowArray::raw_values() + offset);
    } else if constexpr(std::is_same_v<ElementType, float_t>) {
      // convert to 64bits floats
      auto packed32 = _mm_loadu_ps(ValueArrayType<T>::ArrowArray::raw_values() + offset);
      return _mm256_cvtps_pd(packed32);
    } else if constexpr(std::is_same_v<ElementType, int64_t>) {
      // return 64bits ints
      return _mm256_loadu_si256((__m256i*)(ValueArrayType<T>::ArrowArray::raw_values() + offset));
    } else if constexpr(std::is_same_v<ElementType, int32_t>) {
      // convert to 64bits ints
      auto packed32 =
          _mm_loadu_si128((__m128i*)(ValueArrayType<T>::ArrowArray::raw_values() + offset));
      return _mm256_cvtepi32_epi64(packed32);
    } else
#endif //__AVX2__
    {
      return 0;
    }
  }
};

template <typename T> class ValueArrayPtr : public std::shared_ptr<ValueArray<T>> {
public:
  ValueArrayPtr() : std::shared_ptr<ValueArray<T>>() {}

  template <typename CompatibleArray>
  explicit ValueArrayPtr(std::shared_ptr<CompatibleArray> const& basePtr)
      : std::shared_ptr<ValueArray<T>>(std::make_shared<ValueArray<T>>([&basePtr]() {
          if(basePtr->type_id() == ValueArray<T>::TypeClass::type_id) {
            return basePtr->data();
          } else {
            auto adjustedData = basePtr->data()->Copy();
            adjustedData->type = std::make_shared<typename ValueArray<T>::TypeClass>();
            return adjustedData;
          }
        }())) {}

  // conversion from ValueArrayPtr<T> to a "List" expression
  // which is the only representation which is exposed outside of the bulk backend
  explicit operator boss::Expression() const {
    auto const& arrowArray = **this;
    auto size = arrowArray.length();
    auto args = boss::ExpressionArguments();
    args.reserve(size);
    for(auto index = 0L; index < size; ++index) {
      if constexpr(std::is_same_v<T, std::string>) {
        args.emplace_back(arrowArray.GetString(index));
      } else if constexpr(std::is_same_v<T, Symbol>) {
        args.emplace_back(Symbol(arrowArray.GetString(index)));
      } else {
        args.emplace_back(arrowArray.Value(index));
      }
    }
    return boss::ComplexExpression("List"_, std::move(args));
  }
};

template <typename T> class ValueBuilder : public ValueArrayType<T>::ArrowBuilder {
public:
  explicit ValueBuilder(int64_t size, arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ValueBuilder(size, size, pool) {}

  ValueBuilder(int64_t capacity, int64_t size,
               arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ValueArrayType<T>::ArrowBuilder(pool) {
    auto resizeStatus = ValueArrayType<T>::ArrowBuilder::Resize(
        capacity >= 0 ? capacity : bulk::Properties::getMicroBatchesMaxSize());
    if(!resizeStatus.ok()) {
      throw std::runtime_error(resizeStatus.ToString());
    }
    auto status = ValueArrayType<T>::ArrowBuilder::AppendEmptyValues(size);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }

  template <typename BufferValueType>
  using ConversionBuffer = ArrayBuilderConversionBuffer<BufferValueType>;

  // convert back an array to a builder (e.g. for inserting new values)
  explicit ValueBuilder(std::shared_ptr<arrow::Array> const& fromArray, int64_t minSize = 0,
                        arrow::MemoryPool* pool = arrow::default_memory_pool())
      : ValueArrayType<T>::ArrowBuilder(pool) {
    // Common
    // we do not copy the null bitmap since we don't handle missing values this way
    // (and it could expose source arrow arrays' null bitmaps for write access
    // when loading them without copy)
    auto const& data = *fromArray->data();
    this->length_ = data.length;
    this->capacity_ = data.length;

    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
      // String, Symbol builder
      auto const& offsets_buffer = *data.buffers[1];
      auto const& value_buffer = *data.buffers[2];
      std::destroy_at(&this->offsets_builder_);
      new(&this->offsets_builder_) decltype(this->offsets_builder_)(
          arrow::BufferBuilder(std::make_shared<ConversionBuffer<int32_t>>(
                                   offsets_buffer, data.offset,
                                   data.length, // without the final offset since we are
                                                // converting it back to a builder
                                   fromArray),
                               this->pool_));
      auto newValueBufferSize =
          offsets_buffer.size() >= (data.offset + data.length) * sizeof(int32_t)
              ? this->offsets_builder_.data()[data.length] // supposed to be safe since arrays
                                                           // should have a final offset
              : value_buffer.size(); // when loading from cvs, Arrow forgets(?) the final offset
                                     // TODO: check if still needed with newer Arrow version
      std::destroy_at(&this->value_data_builder_);
      new(&this->value_data_builder_) decltype(this->value_data_builder_)(
          arrow::BufferBuilder(std::make_shared<ConversionBuffer<uint8_t>>(
                                   value_buffer, 0, newValueBufferSize, fromArray),
                               this->pool_));
    } else {
      // Boolean, Int, Float Builders
      auto const& data_buffer = *data.buffers[1];
      std::destroy_at(&this->data_builder_);
      new(&this->data_builder_) decltype(this->data_builder_)(arrow::BufferBuilder(
          std::make_shared<ConversionBuffer<T>>(data_buffer, data.offset, data.length, fromArray),
          this->pool_));
    }

    if(this->capacity_ < minSize) {
      this->Reserve(minSize - this->capacity_);
    }
  }

  // conversion from ValueBuilder<T> to ValueArrayPtr<T>
  // to avoid this boilerplate in every code (e.g. in the bulk operators)
  // which create a ValueBuilder<T> and return a ValueArrayPtr<T> as a bulk Expression
  explicit operator ValueArrayPtr<T>() {
    std::shared_ptr<arrow::Array> outputArray;
    auto status = this->Finish(&outputArray);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    return ValueArrayPtr<T>(outputArray);
  }

  // exposing arrow functions which already handle checking the status
  // to avoid this boilerplate in the code
  template <typename U> void Append(U const& element) {
    auto status = ValueArrayType<T>::ArrowBuilder::Append(element);
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  template <typename U> void AppendValues(U const* elements, int64_t length) {
    if constexpr(sizeof(T) == sizeof(U)) {
      // use memmove if there is overlap (https://stackoverflow.com/a/25630217)
      // (or avoid completely copy if dest is same as source)
      auto const* srcDataStart = elements;
      auto const* srcDataEnd = srcDataStart + length;
      auto* destDataStart = (T*)this->data_builder_.mutable_data() + this->data_builder_.length();
      auto* destDataEnd = destDataStart + length;
      if(((void*)destDataEnd >= (void*)srcDataStart && (void*)destDataEnd < (void*)srcDataEnd) ||
         ((void*)srcDataEnd >= (void*)destDataStart && (void*)srcDataEnd < (void*)destDataEnd)) {
        auto status = ValueArrayType<T>::ArrowBuilder::AppendEmptyValues(length);
        if(!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
        if((void*)srcDataStart != (void*)destDataStart) {
          memmove(destDataStart, srcDataStart, static_cast<size_t>(length));
        }
        return;
      }
      auto status = ValueArrayType<T>::ArrowBuilder::AppendValues(elements, length);
      if(!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    } else {
      throw std::runtime_error(std::string("AppendValues with incompatible type T=") +
                               typeid(T).name() + " U=" + typeid(T).name());
    }
  }

  arrow::Status SetLength(int64_t length) {
    if(ARROW_PREDICT_FALSE(length < 0)) {
      return arrow::Status::Invalid("The length must be positive (requested: ", length, ")");
    }

    if(ARROW_PREDICT_FALSE(length > this->length_)) {
      return arrow::Status::Invalid(
          "The length must be less or equal to the maximum length (requested: ", length,
          ", current length: ", this->length_, ")");
    }

    this->length_ = length;
    return arrow::Status::OK();
  }

  // to be used with any SIMD operators
  // so far, supporting an operator taking 8x 32 bits ints/floats
  // and returning 32-bits packed ints/floats values
  template <typename Func, typename... ArrayType>
  void computeSIMD32(Func&& func, ArrayType const&... inputArrays) {
    auto* output = this->data_builder_.mutable_data();
    static auto const packSize = 8; // 8x 32bits
    int64_t index = 0;
    for(; index < this->length_; index += packSize) {
      func(&output[index], inputArrays.rawSIMD32(index)...);
    }
    if(this->length_ > index) {
      func(&output[index], inputArrays.rawSIMD32(index)...);
    }
  }

  // to be used with any SIMD operators
  // so far, supporting an operator taking 4x 64 bits ints/floats
  // and returning 64-bits packed ints/floats values
  template <typename Func, typename... ArrayType>
  void computeSIMD64(Func&& func, ArrayType const&... inputArrays) {
    auto* output = this->data_builder_.mutable_data();
    static auto const packSize = 4; // 4x 64bits
    int64_t index = 0;
    for(; index < this->length_; index += packSize) {
      func(&output[index], inputArrays.rawSIMD64(index)...);
    }
    if(this->length_ > index) {
      func(&output[index], inputArrays.rawSIMD64(index)...);
    }
  }

private:
  // re-implement FinishInternal for performance: do not shrink_to_fit the buffers
  arrow::Status FinishInternal(std::shared_ptr<arrow::ArrayData>* out) override {
    if constexpr(std::is_same_v<T, std::string> || std::is_same_v<T, Symbol>) {
      if(this->offsets_builder_.length() <
         (this->length_ + 1) * sizeof(int32_t)) // fix from original Arrow code: check if
                                                // the buffer already has the final offset
                                                // (it happens when re-using buffer)
      {
        auto status = this->AppendNextOffset();
        if(!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
      }

      auto offsets = std::shared_ptr<arrow::Buffer>();
      auto value_data = std::shared_ptr<arrow::Buffer>();
      auto null_bitmap = std::shared_ptr<arrow::Buffer>();
      auto offsets_size =
          this->offsets_builder_.length() * sizeof(typename arrow::StringBuilder::offset_type);
      auto values_size = this->value_data_builder_.length();
      auto nulls_size = this->null_bitmap_builder_.length();
      this->offsets_builder_.bytes_builder()->Rewind(0); // to avoid to be zero-ed during Finish
      this->value_data_builder_.bytes_builder()->Rewind(0);
      this->null_bitmap_builder_.bytes_builder()->Rewind(0);
      ARROW_RETURN_NOT_OK(this->offsets_builder_.Finish(&offsets, false));
      ARROW_RETURN_NOT_OK(this->value_data_builder_.Finish(&value_data, false));
      ARROW_RETURN_NOT_OK(this->null_bitmap_builder_.Finish(&null_bitmap, false));
      // then put back the right size
      ARROW_RETURN_NOT_OK(
          std::static_pointer_cast<arrow::ResizableBuffer>(offsets)->Resize(offsets_size, false));
      ARROW_RETURN_NOT_OK(
          std::static_pointer_cast<arrow::ResizableBuffer>(value_data)
                              ->Resize(values_size, false));
      ARROW_RETURN_NOT_OK(
          std::static_pointer_cast<arrow::ResizableBuffer>(null_bitmap)
                              ->Resize(nulls_size, false));

      *out = arrow::ArrayData::Make(this->type(), this->length_, {null_bitmap, offsets, value_data},
                                    this->null_count_, 0);
      this->Reset();
      return arrow::Status::OK();
    } else {
      auto data = std::shared_ptr<arrow::Buffer>();
      auto null_bitmap = std::shared_ptr<arrow::Buffer>();
      auto data_size = this->data_builder_.length() * sizeof(T);
      auto nulls_size = this->null_bitmap_builder_.length();
      this->data_builder_.bytes_builder()->Rewind(0); // to avoid to be zero-ed during Finish
      this->null_bitmap_builder_.bytes_builder()->Rewind(0);
      ARROW_RETURN_NOT_OK(this->null_bitmap_builder_.Finish(&null_bitmap, false));
      ARROW_RETURN_NOT_OK(this->data_builder_.Finish(&data, false));
      *out = arrow::ArrayData::Make(this->type(), this->length_, {null_bitmap, data},
                                    this->null_count_);
      // then put back the right size
      ARROW_RETURN_NOT_OK(
          std::static_pointer_cast<arrow::ResizableBuffer>(data)->Resize(data_size, false));
      ARROW_RETURN_NOT_OK(
          std::static_pointer_cast<arrow::ResizableBuffer>(null_bitmap)
                              ->Resize(nulls_size, false));
    }
    this->capacity_ = 0;
    this->length_ = 0;
    this->null_count_ = 0;
    return arrow::Status::OK();
  }
};

// specialization for bool
// to be used with any SIMD operators
// so far, supporting an operator taking 8x 32 bits ints/floats
// and returning 8-bits packed bool values
template <>
template <typename Func, typename... ArrayType>
void ValueBuilder<bool>::computeSIMD32(Func&& func, ArrayType const&... inputArrays) {
  auto* output = data_builder_.mutable_data();
  constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
  int64_t outputIndex = 0;
  auto const outputLength = length_ / bitsPerPack;
  for(; outputIndex < outputLength; ++outputIndex) {
    auto inputIndex = outputIndex * bitsPerPack;
    output[outputIndex] = func(inputArrays.rawSIMD32(inputIndex)...);
  }
  auto inputIndex = outputIndex * bitsPerPack;
  if(length_ > inputIndex) {
    output[outputIndex] = func(inputArrays.rawSIMD32(inputIndex)...);
  }
}

// specialization for bool
// to be used with any SIMD operators
// so far, supporting an operator taking 8x 64 bits ints/floats
// and returning 8-bits packed bool values
template <>
template <typename Func, typename... ArrayType>
void ValueBuilder<bool>::computeSIMD64(Func&& func, ArrayType const&... inputArrays) {
  auto* output = data_builder_.mutable_data();
  constexpr auto bitsPerPack = (int)sizeof(*output) * CHAR_BIT;
  int64_t outputIndex = 0;
  auto const outputLength = length_ / bitsPerPack;
  for(; outputIndex < outputLength; ++outputIndex) {
    auto inputIndex = outputIndex * bitsPerPack;
    output[outputIndex] =
        func(inputArrays.rawSIMD64(inputIndex)..., inputArrays.rawSIMD64(inputIndex + 4)...);
  }
  auto inputIndex = outputIndex * bitsPerPack;
  if(length_ > inputIndex) {
    output[outputIndex] =
        func(inputArrays.rawSIMD64(inputIndex)..., inputArrays.rawSIMD64(inputIndex + 4)...);
  }
}

} // namespace boss::engines::bulk
