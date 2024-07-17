#pragma once

#include <arrow/buffer.h>
#include <arrow/buffer_builder.h>
#include <arrow/util/bit_util.h>

#include <memory>
#include <variant>

namespace boss::engines::bulk {

/// this class only exist for the workaround to extract a sliced buffer data from a builder
/// to implement Slice for builders
template <typename T> class ArrayBuilderConversionBuffer : public arrow::ResizableBuffer {
public:
  using ParentArrayPtr = std::shared_ptr<arrow::Array const>;

  ArrayBuilderConversionBuffer(arrow::Buffer const& buffer, int64_t offset, int64_t size,
                               ParentArrayPtr const& parent,
                               arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::ResizableBuffer(const_cast<uint8_t*>(buffer.data()) + offset * sizeof(T), // NOLINT
                               size * sizeof(T)),
        parent(parent), pool_(pool) {
    capacity_ = buffer.capacity() - offset * sizeof(T);
  }

  ~ArrayBuilderConversionBuffer() override {
    if(!parent && data_) {
      pool_->Free(const_cast<uint8_t*>(data_), capacity_); // NOLINT
    }
  }

  ArrayBuilderConversionBuffer(ArrayBuilderConversionBuffer&) = delete;
  ArrayBuilderConversionBuffer& operator=(ArrayBuilderConversionBuffer&) = delete;
  ArrayBuilderConversionBuffer(ArrayBuilderConversionBuffer&&) = delete;
  ArrayBuilderConversionBuffer& operator=(ArrayBuilderConversionBuffer&&) = delete;

  arrow::Status Resize(const int64_t new_size, bool /*shrink_to_fit*/ = true) override {
    if(new_size < 0) {
      return arrow::Status::Invalid("Negative buffer resize: ", new_size);
    }
    // ignore shrinking, just set the size
    if(new_size > size_) {
      auto status = Reserve(new_size);
      if(!status.ok()) {
        return status;
      }
    }
    size_ = new_size;
    return arrow::Status::OK();
  }

  arrow::Status Reserve(const int64_t capacity) override {
    if(capacity < 0) {
      return arrow::Status::Invalid("Negative buffer capacity: ", capacity);
    }
    if(data_ == nullptr || capacity > capacity_) {
      auto new_capacity = arrow::BitUtil::RoundUpToMultipleOf64(capacity);
      if(data_ && !parent) {
        auto status =
            pool_->Reallocate(capacity_, new_capacity, const_cast<uint8_t**>(&data_)); // NOLINT
        if(!status.ok()) {
          return status;
        }
      } else {
        uint8_t* new_data = nullptr;
        auto status = pool_->Allocate(new_capacity, &new_data);
        if(!status.ok()) {
          return status;
        }
        if(new_capacity > capacity_) {
          memcpy(new_data, data_, static_cast<size_t>(capacity_));
        } else {
          memcpy(new_data, data_, static_cast<size_t>(new_capacity));
        }
        data_ = new_data;
        parent = nullptr;
      }
      capacity_ = new_capacity;
    }
    return arrow::Status::OK();
  }

private:
  ParentArrayPtr parent;
  arrow::MemoryPool* pool_;
};

template <>
inline ArrayBuilderConversionBuffer<bool>::ArrayBuilderConversionBuffer(
    arrow::Buffer const& buffer, int64_t offset, int64_t size, ParentArrayPtr const& parent,
    arrow::MemoryPool* pool)
    : arrow::ResizableBuffer(const_cast<uint8_t*>(buffer.data()) + // NOLINT
                                 arrow::BitUtil::BytesForBits(offset),
                             arrow::BitUtil::BytesForBits(size)),
      parent(parent), pool_(pool) {}

} // namespace boss::engines::bulk
