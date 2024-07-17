#pragma once

#include "Bulk.hpp"
#include "BulkArrays.hpp"
#include "BulkExpression.hpp"
#include "BulkUtilities.hpp"
#include "OperatorUtilities.hpp"
#include "SymbolRegistry.hpp"
#include "Table.hpp"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif // __AVX2__ || __AVX512F__

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

namespace boss::engines::bulk {

template <typename... ArgumentTypes>
class Greater : public boss::engines::bulk::Operator<Greater, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant_merge_t<AnyCombinationOfArgumentTypes_t<2, tuple<int32_t, ValueArrayPtr<int32_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<int64_t, ValueArrayPtr<int64_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<float_t, ValueArrayPtr<float_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<double_t, ValueArrayPtr<double_t>>>>;
  using Operator<Greater, ArgumentTypes...>::Operator;

  template <typename T1, typename T2> void operator()(T1 t1, T2 t2) { this->pushUp(t1 > t2); }

  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, int32_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD32(
        [](__m256i lhs, __m256i rhs) {
          auto comp = _mm256_cmpgt_epi32(lhs, rhs);
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs > rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, int64_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD64(
        [](__m256i lhs1, __m256i rhs1, __m256i lhs2, __m256i rhs2) {
          // comp for each pack of 4 64bits
          auto comp1 = _mm256_cmpgt_epi64(lhs1, rhs1);
          auto comp2 = _mm256_cmpgt_epi64(lhs2, rhs2);
          // combine the two comps into one pack of 8 32bits
          auto combined = _mm256_shuffle_ps(_mm256_castsi256_ps(comp1), _mm256_castsi256_ps(comp2),
                                            0x88); // _MM_SHUFFLE(2,0,2,0)
          auto ordered =
              _mm256_permute4x64_pd(_mm256_castps_pd(combined), 0xd8); // _MM_SHUFFLE(3, 1, 2, 0)
          auto comp = _mm256_castpd_si256(ordered);
          // convert 32bits to 8bits (like 32bits version)
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs > rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, float_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD32(
        [](__m256 lhs, __m256 rhs) {
          auto comp = _mm256_castps_si256(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ));
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs > rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, double_t>> bulkOp(T1 const& lhsArray,
                                                                              T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD64(
        [](__m256d lhs1, __m256d rhs1, __m256d lhs2, __m256d rhs2) {
          // comp for each pack of 4 64bits
          auto comp1 = _mm256_cmp_pd(lhs1, rhs1, _CMP_GT_OQ);
          auto comp2 = _mm256_cmp_pd(lhs2, rhs2, _CMP_GT_OQ);
          // combine the two comps into one pack of 8 32bits
          auto combined = _mm256_shuffle_ps(_mm256_castpd_ps(comp1), _mm256_castpd_ps(comp2),
                                            0x88); // _MM_SHUFFLE(2,0,2,0)
          auto ordered =
              _mm256_permute4x64_pd(_mm256_castps_pd(combined), 0xd8); // _MM_SHUFFLE(3, 1, 2, 0)
          auto comp = _mm256_castpd_si256(ordered);
          // convert 32bits to 8bits (like 32bits version)
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs > rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<Greater> const r("Greater"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class Equal : public boss::engines::bulk::Operator<Equal, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant_merge_t<AnyCombinationOfArgumentTypes_t<2, tuple<int32_t, ValueArrayPtr<int32_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<int64_t, ValueArrayPtr<int64_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<float_t, ValueArrayPtr<float_t>>>,
                      AnyCombinationOfArgumentTypes_t<2, tuple<double_t, ValueArrayPtr<double_t>>>>;
  using Operator<Equal, ArgumentTypes...>::Operator;

  template <typename T1, typename T2> void operator()(T1 t1, T2 t2) { this->pushUp(t1 == t2); }

  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, int32_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD32(
        [](__m256i lhs, __m256i rhs) {
          auto comp = _mm256_cmpeq_epi32(lhs, rhs);
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs == rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, int64_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD64(
        [](__m256i lhs1, __m256i rhs1, __m256i lhs2, __m256i rhs2) {
          // comp for each pack of 4 64bits
          auto comp1 = _mm256_cmpeq_epi64(lhs1, rhs1);
          auto comp2 = _mm256_cmpeq_epi64(lhs2, rhs2);
          // combine the two comps into one pack of 8 32bits
          auto combined = _mm256_shuffle_ps(_mm256_castsi256_ps(comp1), _mm256_castsi256_ps(comp2),
                                            0x88); // _MM_SHUFFLE(2,0,2,0)
          auto ordered =
              _mm256_permute4x64_pd(_mm256_castps_pd(combined), 0xd8); // _MM_SHUFFLE(3, 1, 2, 0)
          auto comp = _mm256_castpd_si256(ordered);
          // convert 32bits to 8bits (like 32bits version)
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs == rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, float_t>> bulkOp(T1 const& lhsArray,
                                                                             T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD32(
        [](__m256 lhs, __m256 rhs) {
          auto comp = _mm256_castps_si256(_mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ));
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs == rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  template <typename T1, typename T2>
  std::enable_if_t<std::is_same_v<typename T1::ElementType, double_t>> bulkOp(T1 const& lhsArray,
                                                                              T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
#ifdef __AVX2__
    outputBuilder.computeSIMD64(
        [](__m256d lhs1, __m256d rhs1, __m256d lhs2, __m256d rhs2) {
          // comp for each pack of 4 64bits
          auto comp1 = _mm256_cmp_pd(lhs1, rhs1, _CMP_EQ_OQ);
          auto comp2 = _mm256_cmp_pd(lhs2, rhs2, _CMP_EQ_OQ);
          // combine the two comps into one pack of 8 32bits
          auto combined = _mm256_shuffle_ps(_mm256_castpd_ps(comp1), _mm256_castpd_ps(comp2),
                                            0x88); // _MM_SHUFFLE(2,0,2,0)
          auto ordered =
              _mm256_permute4x64_pd(_mm256_castps_pd(combined), 0xd8); // _MM_SHUFFLE(3, 1, 2, 0)
          auto comp = _mm256_castpd_si256(ordered);
          // convert 32bits to 8bits (like 32bits version)
          auto permComp = _mm256_permute2f128_si256(comp, comp, 1);
          auto packed16 = _mm256_packs_epi32(comp, permComp);
          auto packed8 = _mm256_packs_epi16(packed16, packed16);
          return _mm256_movemask_epi8(packed8);
        },
        lhsArray, rhsArray);
#else  // !__AVX2__
    outputBuilder.compute([](auto lhs, auto rhs) { return lhs == rhs; }, lhsArray, rhsArray);
#endif // __AVX2__
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<Equal> const rb("Equal"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class And : public boss::engines::bulk::Operator<And, ArgumentTypes...> {
public:
  static constexpr int MAX_NUM_ARGS = 10;
  using ArgumentTypesT =
      variant_merge_t<variant<tuple<bool, ValueArrayPtr<bool>>, tuple<ValueArrayPtr<bool>, bool>>,
                      RepeatedArgumentTypeOfAnySize_t<1, MAX_NUM_ARGS, bool>,
                      RepeatedArgumentTypeOfAnySize_t<1, MAX_NUM_ARGS, ValueArrayPtr<bool>>>;
  using Operator<And, ArgumentTypes...>::Operator;

  template <typename... T> void operator()(T... values) { this->pushUp((values && ...)); }

  template <typename T1, typename... Ts>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<Ts> const&... rhsArrayPtrs) {
    auto const& lhsArray = *lhsArrayPtr;
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
    outputBuilder.computeBitwise(
        [](auto... values) {
          static_assert(std::conjunction_v<std::is_unsigned<std::decay_t<decltype(values)>>...>);
          return ((unsigned int)values & ...);
        },
        lhsArray, *rhsArrayPtrs...);
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }

  void operator()(bool lhsConstantValue, ValueArrayPtr<bool> const& rhsArrayPtr) {
    operator()(rhsArrayPtr, lhsConstantValue);
  }

  void operator()(ValueArrayPtr<bool>&& lhsArrayPtr, bool rhsConstantValue) {
    if(rhsConstantValue) {
      // no changes to the left side array
      this->pushUp(std::move(lhsArrayPtr));
      return;
    }
    // create an array with only false values
    this->pushUp((ValueArrayPtr<bool>)ValueBuilder<bool>(lhsArrayPtr->length()));
  }
};
namespace {
boss::engines::bulk::Engine::Register<And> const r0("And"); // NOLINT
}

template <typename... ArgumentTypes>
class Not : public boss::engines::bulk::Operator<Not, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<bool>, tuple<ValueArrayPtr<bool>>>;
  using Operator<Not, ArgumentTypes...>::Operator;

  void operator()(bool value) { this->pushUp(value); }

  void operator()(ValueArrayPtr<bool> const& arrayPtr) {
    auto const& boolArray = *arrayPtr;
    auto outputBuilder = ValueBuilder<bool>(boolArray.length());
    outputBuilder.computeBitwise(
        [](auto value) {
          static_assert(std::is_unsigned_v<std::decay_t<decltype(value)>>);
          return ~(unsigned int)value;
        },
        boolArray);
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<Not> const r01("Not"); // NOLINT
}

template <typename... ArgumentTypes>
class Plus : public boss::engines::bulk::Operator<Plus, ArgumentTypes...> {
public:
  static constexpr int MAX_NUMBER_OF_VARIOUS_ARGS = 5;
  static constexpr int MAX_NUMBER_OF_SIMPLE_ARGS = 20;
  using ArgumentTypesT = variant_merge_t<
      AnyCombinationOfArgumentTypes_t<
          2, tuple<int32_t, int64_t, float_t, double_t, ValueArrayPtr<int32_t>,
                   ValueArrayPtr<int64_t>, ValueArrayPtr<float_t>, ValueArrayPtr<double_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<int32_t, ValueArrayPtr<int32_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<int64_t, ValueArrayPtr<int64_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<float_t, ValueArrayPtr<float_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<double_t, ValueArrayPtr<double_t>>>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      int32_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      int64_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      float_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      double_t>>;
  using Operator<Plus, ArgumentTypes...>::Operator;
  template <typename... Args> void operator()(Args const&... args) {
    this->pushUp(calculate(args...));
  }

private:
  template <typename T1, typename... Ts>
  auto calculate(T1 const& lshValue, Ts const&... otherValues) {
    return calculate(lshValue, calculate(otherValues...));
  }
  template <typename T1, typename T2>
  auto calculate(T1 const& lhsConstantValue, T2 const& rhsConstantValue) {
    return lhsConstantValue + rhsConstantValue;
  }
  template <typename T1, typename T2>
  auto calculate(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  auto calculate(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue),
                  *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  auto calculate(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    return bulkOp(*lhsArrayPtr,
                  Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }
  template <typename T1, typename T2> auto bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    using OutputT = decltype(lhsArray.Value(0) + rhsArray.Value(0));
    auto outputBuilder = ValueBuilder<OutputT>(lhsArray.length());
#ifdef __AVX2__
    // for now, we support SIMD only for binary operation on the two same types
    if constexpr(std::is_same_v<typename T1::ElementType, int32_t> &&
                 std::is_same_v<typename T2::ElementType, int32_t>) {
      outputBuilder.computeSIMD32(
          [](int32_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_add_epi32(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, int64_t> &&
                        std::is_same_v<typename T2::ElementType, int64_t>) {
      outputBuilder.computeSIMD64(
          [](int64_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_add_epi64(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, float_t> &&
                        std::is_same_v<typename T2::ElementType, float_t>) {
      outputBuilder.computeSIMD32(
          [](float_t* memAddr, __m256 lhs, __m256 rhs) {
            auto output = _mm256_add_ps(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_ps(memAddr, output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, double_t> &&
                        std::is_same_v<typename T2::ElementType, double_t>) {
      outputBuilder.computeSIMD64(
          [](double_t* memAddr, __m256d lhs, __m256d rhs) {
            auto output = _mm256_add_pd(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_pd(memAddr, output);
          },
          lhsArray, rhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < lhsArray.length(); ++index) {
        outputBuilder[index] = lhsArray.Value(index) + rhsArray.Value(index);
      }
    }
    return (ValueArrayPtr<OutputT>)outputBuilder;
  }
};
namespace {
boss::engines::bulk::Engine::Register<Plus> const r1("Plus"); // NOLINT
}

template <typename... ArgumentTypes>
class Minus : public boss::engines::bulk::Operator<Minus, ArgumentTypes...> {
public:
  using ArgumentTypesT = AnyCombinationOfArgumentTypesOfAnySize_t<
      1, 2,
      tuple<int32_t, int64_t, float_t, double_t, ValueArrayPtr<int32_t>, ValueArrayPtr<int64_t>,
            ValueArrayPtr<float_t>, ValueArrayPtr<double_t>>>;
  using Operator<Minus, ArgumentTypes...>::Operator;

  template <typename T1> void operator()(T1 rhsConstantValue) { this->pushUp(-rhsConstantValue); }
  template <typename T1, typename T2> void operator()(T1 lhsConstantValue, T2 rhsConstantValue) {
    this->pushUp(lhsConstantValue - rhsConstantValue);
  }
  template <typename T1> void operator()(ValueArrayPtr<T1> const& rhsArrayPtr) {
    bulkOp(*rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1> void bulkOp(T1 const& rhsArray) {
    using OutputT = decltype(-rhsArray.Value(0));
    auto outputBuilder = ValueBuilder<OutputT>(rhsArray.length());
#ifdef __AVX2__
    if constexpr(std::is_same_v<typename T1::ElementType, int32_t>) {
      outputBuilder.computeSIMD32(
          [](int32_t* memAddr, __m256i rhs) {
            auto zero = _mm256_setzero_si256();
            auto output = _mm256_sub_epi32(zero, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, int64_t>) {
      outputBuilder.computeSIMD64(
          [](int64_t* memAddr, __m256i rhs) {
            auto zero = _mm256_setzero_si256();
            auto output = _mm256_sub_epi64(zero, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, float_t>) {
      outputBuilder.computeSIMD32(
          [](float_t* memAddr, __m256 rhs) {
            auto zero = _mm256_setzero_ps();
            auto output = _mm256_sub_ps(zero, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_ps(memAddr, output);
          },
          rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, double_t>) {
      outputBuilder.computeSIMD64(
          [](double_t* memAddr, __m256d rhs) {
            auto zero = _mm256_setzero_pd();
            auto output = _mm256_sub_pd(zero, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_pd(memAddr, output);
          },
          rhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < rhsArray.length(); ++index) {
        outputBuilder[index] = -rhsArray.Value(index);
      }
    }
    this->pushUp((ValueArrayPtr<OutputT>)outputBuilder);
  }
  template <typename T1, typename T2> void bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    using OutputT = decltype(lhsArray.Value(0) - rhsArray.Value(0));
    auto outputBuilder = ValueBuilder<OutputT>(lhsArray.length());
#ifdef __AVX2__
    // for now, we support SIMD only for binary operation on the two same types
    if constexpr(std::is_same_v<typename T1::ElementType, int32_t> &&
                 std::is_same_v<typename T2::ElementType, int32_t>) {
      outputBuilder.computeSIMD32(
          [](int32_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_sub_epi32(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, int64_t> &&
                        std::is_same_v<typename T2::ElementType, int64_t>) {
      outputBuilder.computeSIMD64(
          [](int64_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_sub_epi64(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, float_t> &&
                        std::is_same_v<typename T2::ElementType, float_t>) {
      outputBuilder.computeSIMD32(
          [](float_t* memAddr, __m256 lhs, __m256 rhs) {
            auto output = _mm256_sub_ps(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_ps(memAddr, output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, double_t> &&
                        std::is_same_v<typename T2::ElementType, double_t>) {
      outputBuilder.computeSIMD64(
          [](double_t* memAddr, __m256d lhs, __m256d rhs) {
            auto output = _mm256_sub_pd(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_pd(memAddr, output);
          },
          lhsArray, rhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < lhsArray.length(); ++index) {
        outputBuilder[index] = lhsArray.Value(index) - rhsArray.Value(index);
      }
    }
    this->pushUp((ValueArrayPtr<OutputT>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<Minus> const r1b("Minus"); // NOLINT
}

template <typename... ArgumentTypes>
class Times : public boss::engines::bulk::Operator<Times, ArgumentTypes...> {
public:
  static constexpr int MAX_NUMBER_OF_VARIOUS_ARGS = 5;
  static constexpr int MAX_NUMBER_OF_SIMPLE_ARGS = 20;
  using ArgumentTypesT = variant_merge_t<
      AnyCombinationOfArgumentTypes_t<
          2, tuple<int32_t, int64_t, float_t, double_t, ValueArrayPtr<int32_t>,
                   ValueArrayPtr<int64_t>, ValueArrayPtr<float_t>, ValueArrayPtr<double_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<int32_t, ValueArrayPtr<int32_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<int64_t, ValueArrayPtr<int64_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<float_t, ValueArrayPtr<float_t>>>,
      AnyCombinationOfArgumentTypesOfAnySize_t<3, MAX_NUMBER_OF_VARIOUS_ARGS,
                                               tuple<double_t, ValueArrayPtr<double_t>>>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      int32_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      int64_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      float_t>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      double_t>>;
  using Operator<Times, ArgumentTypes...>::Operator;
  template <typename... Args> void operator()(Args const&... args) {
    this->pushUp(calculate(args...));
  }

private:
  template <typename T1, typename... Ts>
  auto calculate(T1 const& lshValue, Ts const&... otherValues) {
    return calculate(lshValue, calculate(otherValues...));
  }
  template <typename T1, typename T2>
  auto calculate(T1 const& lhsConstantValue, T2 const& rhsConstantValue) {
    return lhsConstantValue * rhsConstantValue;
  }
  template <typename T1, typename T2>
  auto calculate(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  auto calculate(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue),
                  *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  auto calculate(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    return bulkOp(*lhsArrayPtr,
                  Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }
  template <typename T1, typename T2> auto bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    using OutputT = decltype(lhsArray.Value(0) * rhsArray.Value(0));
    auto outputBuilder = ValueBuilder<OutputT>(lhsArray.length());
#ifdef __AVX2__
    // for now, we support SIMD only for binary operation on the two same types
    if constexpr(std::is_same_v<typename T1::ElementType, int32_t> &&
                 std::is_same_v<typename T2::ElementType, int32_t>) {
      outputBuilder.computeSIMD32(
          [](int32_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_mullo_epi32(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, int64_t> &&
                        std::is_same_v<typename T2::ElementType, int64_t>) {
      outputBuilder.computeSIMD64(
          [](int64_t* memAddr, __m256i lhs, __m256i rhs) {
            auto output = _mm256_mul_epi32(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, float_t> &&
                        std::is_same_v<typename T2::ElementType, float_t>) {
      outputBuilder.computeSIMD32(
          [](float_t* memAddr, __m256 lhs, __m256 rhs) {
            auto output = _mm256_mul_ps(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_ps(memAddr, output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, double_t> &&
                        std::is_same_v<typename T2::ElementType, double_t>) {
      outputBuilder.computeSIMD64(
          [](double_t* memAddr, __m256d lhs, __m256d rhs) {
            auto output = _mm256_mul_pd(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_pd(memAddr, output);
          },
          lhsArray, rhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < lhsArray.length(); ++index) {
        outputBuilder[index] = lhsArray.Value(index) * rhsArray.Value(index);
      }
    }
    return (ValueArrayPtr<OutputT>)outputBuilder;
  }
};
namespace {
boss::engines::bulk::Engine::Register<Times> const r1c("Times"); // NOLINT
}

template <typename... ArgumentTypes>
class Divide : public boss::engines::bulk::Operator<Divide, ArgumentTypes...> {
public:
  using ArgumentTypesT = AnyCombinationOfArgumentTypes_t<
      2, tuple<int32_t, int64_t, float_t, double_t, ValueArrayPtr<int32_t>, ValueArrayPtr<int64_t>,
               ValueArrayPtr<float_t>, ValueArrayPtr<double_t>>>;
  using Operator<Divide, ArgumentTypes...>::Operator;

  template <typename T1, typename T2> void operator()(T1 lhsConstantValue, T2 rhsConstantValue) {
    this->pushUp(lhsConstantValue / rhsConstantValue);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2> void bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    using OperationT = decltype(lhsArray.Value(0) / rhsArray.Value(0));
    using OutputT = std::conditional_t<std::is_integral_v<OperationT>, double_t, OperationT>;
    auto outputBuilder = ValueBuilder<OutputT>(lhsArray.length());
#ifdef __AVX2__
    // for now, we support SIMD only for binary operation on the two same types
    // and noly for floating point (since only ps and pd are supported with AVX2)
    if constexpr(std::is_same_v<typename T1::ElementType, float_t> &&
                 std::is_same_v<typename T2::ElementType, float_t>) {
      outputBuilder.computeSIMD32(
          [](float_t* memAddr, __m256 lhs, __m256 rhs) {
            auto output = _mm256_div_ps(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_ps(memAddr, output);
          },
          lhsArray, rhsArray);
    } else if constexpr(std::is_same_v<typename T1::ElementType, double_t> &&
                        std::is_same_v<typename T2::ElementType, double_t>) {
      outputBuilder.computeSIMD64(
          [](double_t* memAddr, __m256d lhs, __m256d rhs) {
            auto output = _mm256_div_pd(lhs, rhs); // NOLINT(portability-simd-intrinsics)
            _mm256_store_pd(memAddr, output);
          },
          lhsArray, rhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < lhsArray.length(); ++index) {
        outputBuilder[index] = (OutputT)lhsArray.Value(index) / rhsArray.Value(index);
      }
    }
    this->pushUp((ValueArrayPtr<OutputT>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<Divide> const r1d("Divide"); // NOLINT
}

template <typename... ArgumentTypes>
class ConvertToInt32 : public boss::engines::bulk::Operator<ConvertToInt32, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<int64_t>, tuple<float_t>, tuple<double_t>, tuple<ValueArrayPtr<int64_t>>,
              tuple<ValueArrayPtr<float_t>>, tuple<ValueArrayPtr<double_t>>>;
  using Operator<ConvertToInt32, ArgumentTypes...>::Operator;

  template <typename T1> void operator()(T1 rhsConstantValue) {
    this->pushUp((int32_t)rhsConstantValue);
  }
  template <typename T1> void operator()(ValueArrayPtr<T1> const& rhsArrayPtr) {
    bulkOp(*rhsArrayPtr);
  }

private:
  template <typename T1> void bulkOp(T1 const& rhsArray) {
    auto outputBuilder = ValueBuilder<int32_t>(rhsArray.length());
    for(int64_t index = 0; index < rhsArray.length(); ++index) {
      outputBuilder[index] = (int32_t)rhsArray.Value(index);
    }
    this->pushUp((ValueArrayPtr<int32_t>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<ConvertToInt32> const r1e("Int32"); // NOLINT
}

template <typename... ArgumentTypes>
class ConvertToInt64 : public boss::engines::bulk::Operator<ConvertToInt64, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<int32_t>, tuple<float_t>, tuple<double_t>, tuple<ValueArrayPtr<int32_t>>,
              tuple<ValueArrayPtr<float_t>>, tuple<ValueArrayPtr<double_t>>>;
  using Operator<ConvertToInt64, ArgumentTypes...>::Operator;

  template <typename T1> void operator()(T1 rhsConstantValue) {
    this->pushUp((int64_t)rhsConstantValue);
  }
  template <typename T1> void operator()(ValueArrayPtr<T1> const& rhsArrayPtr) {
    bulkOp(*rhsArrayPtr);
  }

private:
  template <typename T1> void bulkOp(T1 const& rhsArray) {
    auto outputBuilder = ValueBuilder<int64_t>(rhsArray.length());
    for(int64_t index = 0; index < rhsArray.length(); ++index) {
      outputBuilder[index] = (int64_t)rhsArray.Value(index);
    }
    this->pushUp((ValueArrayPtr<int64_t>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<ConvertToInt64> const r1f("Int64"); // NOLINT
}

template <typename... ArgumentTypes>
class ConvertToFloat : public boss::engines::bulk::Operator<ConvertToFloat, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<int32_t>, tuple<int64_t>, tuple<double_t>, tuple<ValueArrayPtr<int32_t>>,
              tuple<ValueArrayPtr<int64_t>>, tuple<ValueArrayPtr<double_t>>>;
  using Operator<ConvertToFloat, ArgumentTypes...>::Operator;

  template <typename T1> void operator()(T1 rhsConstantValue) {
    this->pushUp((float_t)rhsConstantValue);
  }
  template <typename T1> void operator()(ValueArrayPtr<T1> const& rhsArrayPtr) {
    bulkOp(*rhsArrayPtr);
  }

private:
  template <typename T1> void bulkOp(T1 const& rhsArray) {
    auto outputBuilder = ValueBuilder<float_t>(rhsArray.length());
    for(int64_t index = 0; index < rhsArray.length(); ++index) {
      outputBuilder[index] = (float_t)rhsArray.Value(index);
    }
    this->pushUp((ValueArrayPtr<float_t>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<ConvertToFloat> const r1g("Float"); // NOLINT
}

template <typename... ArgumentTypes>
class ConvertToDouble : public boss::engines::bulk::Operator<ConvertToDouble, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<int32_t>, tuple<int64_t>, tuple<float_t>, tuple<ValueArrayPtr<int32_t>>,
              tuple<ValueArrayPtr<int64_t>>, tuple<ValueArrayPtr<float_t>>>;
  using Operator<ConvertToDouble, ArgumentTypes...>::Operator;

  template <typename T1> void operator()(T1 rhsConstantValue) {
    this->pushUp((double_t)rhsConstantValue);
  }
  template <typename T1> void operator()(ValueArrayPtr<T1> const& rhsArrayPtr) {
    bulkOp(*rhsArrayPtr);
  }

private:
  template <typename T1> void bulkOp(T1 const& rhsArray) {
    auto outputBuilder = ValueBuilder<double_t>(rhsArray.length());
    for(int64_t index = 0; index < rhsArray.length(); ++index) {
      outputBuilder[index] = (double_t)rhsArray.Value(index);
    }
    this->pushUp((ValueArrayPtr<double_t>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<ConvertToDouble> const r1h("Double"); // NOLINT
}

template <typename... ArgumentTypes>
class StringJoin : public boss::engines::bulk::Operator<StringJoin, ArgumentTypes...> {
public:
  static constexpr int MAX_NUMBER_OF_VARIOUS_ARGS = 2;
  static constexpr int MAX_NUMBER_OF_SIMPLE_ARGS = 10;
  using ArgumentTypesT = variant_merge_t<
      AnyCombinationOfArgumentTypes_t<MAX_NUMBER_OF_VARIOUS_ARGS,
                                      tuple<std::string, ValueArrayPtr<std::string>>>,
      RepeatedArgumentTypeOfAnySize_t<MAX_NUMBER_OF_VARIOUS_ARGS + 1, MAX_NUMBER_OF_SIMPLE_ARGS,
                                      std::string>>;
  using Operator<StringJoin, ArgumentTypes...>::Operator;

  template <typename... Args> void operator()(Args... args) { this->pushUp((args + ...)); }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2> void bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<std::string>(0);
    auto length = lhsArray.length();
    auto totalDataSize = lhsArray.total_values_length() + rhsArray.total_values_length();
    outputBuilder.Reserve(length);
    outputBuilder.ReserveData(totalDataSize);
    for(int64_t index = 0; index < length; ++index) {
      outputBuilder.UnsafeAppend(lhsArray.Value(index));
      outputBuilder.UnsafeConcatenate(rhsArray.Value(index));
    }
    this->pushUp((ValueArrayPtr<std::string>)outputBuilder);
  }
};
namespace {
static boss::engines::bulk::Engine::Register<StringJoin> const r2("StringJoin"); // NOLINT
}

template <typename... ArgumentTypes>
class StringContainsQ : public boss::engines::bulk::Operator<StringContainsQ, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      AnyCombinationOfArgumentTypes_t<2, tuple<std::string, ValueArrayPtr<std::string>>>;
  using Operator<StringContainsQ, ArgumentTypes...>::Operator;

  template <typename T1, typename T2> void operator()(T1 const& t1, T2 const& t2) {
    this->pushUp(t1.find(t2) != string::npos);
  }

  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue), *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  void operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    bulkOp(*lhsArrayPtr, Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2> void bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    auto outputBuilder = ValueBuilder<bool>(lhsArray.length());
    outputBuilder.compute(
        [](auto const& lhs, auto const& rhs) { return lhs.find(rhs) != string::npos; }, lhsArray,
        rhsArray);
    this->pushUp((ValueArrayPtr<bool>)outputBuilder);
  }
};
namespace {
boss::engines::bulk::Engine::Register<StringContainsQ> const r3("StringContainsQ"); // NOLINT
}

template <typename... ArgumentTypes>
class StringContainsQPL
    : public boss::engines::bulk::Operator<StringContainsQPL, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      AnyCombinationOfArgumentTypes_t<2, tuple<std::string, ValueArrayPtr<std::string>>>;
  using Operator<StringContainsQPL, ArgumentTypes...>::Operator;

  template <typename T1, typename T2> bool operator()(T1 const& t1, T2 const& t2) {
    return t1.find(t2) != string::npos;
  }

  template <typename T1, typename T2>
  bool operator()(ValueArrayPtr<T1> const& lhsArrayPtr, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(*lhsArrayPtr, *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  bool operator()(T1 const& lhsConstantValue, ValueArrayPtr<T2> const& rhsArrayPtr) {
    return bulkOp(Utilities::RangeGenerator<T1>(rhsArrayPtr->length(), lhsConstantValue),
                  *rhsArrayPtr);
  }
  template <typename T1, typename T2>
  bool operator()(ValueArrayPtr<T1> const& lhsArrayPtr, T2 const& rhsConstantValue) {
    return bulkOp(*lhsArrayPtr,
                  Utilities::RangeGenerator<T2>(lhsArrayPtr->length(), rhsConstantValue));
  }

private:
  template <typename T1, typename T2> bool bulkOp(T1 const& lhsArray, T2 const& rhsArray) {
    auto outputBuilder =
        ValueBuilder<int64_t>(lhsArray.length()); // maybe allocate less than the full buffer size
    auto comparator = [](auto const& lhs, auto const& rhs) {
      return lhs.find(rhs) != string::npos;
    };
    auto destIndex = 0L;
    for(auto index = 0L; index < lhsArray.length(); ++index) {
      // Increase the buffer size with every insertion
      // if (comparator(lhsArray.Value(index), rhsArray.Value(index))) {
      //  outputBuilder.Reserve(1); // NOLINT
      //  outputBuilder.UnsafeAppend((int) index);
      //}
      outputBuilder[destIndex] = (int64_t)index;
      destIndex += comparator(lhsArray.Value(index), rhsArray.Value(index));
    }
    auto status = outputBuilder.SetLength(destIndex); // manually set the result of the index
    if(!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    this->pushUp((ValueArrayPtr<int64_t>)outputBuilder);
    return true;
  }
};
namespace {
boss::engines::bulk::Engine::Register<StringContainsQPL> const r3b("StringContainsQPL"); // NOLINT
}

template <typename... ArgumentTypes>
class Sym : public boss::engines::bulk::Operator<Sym, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<string>>;
  using Operator<Sym, ArgumentTypes...>::Operator;
  void operator()(string const& name) { this->pushUp(boss::Symbol(name)); }
};
namespace {
boss::engines::bulk::Engine::Register<Sym> const r4("Symbol"); // NOLINT
}

template <typename... ArgumentTypes>
class Extract : public boss::engines::bulk::Operator<Extract, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<ComplexExpression, int64_t>, tuple<ValueArrayPtr<bool>, int64_t>,
              tuple<ValueArrayPtr<int32_t>, int64_t>, tuple<ValueArrayPtr<int64_t>, int64_t>,
              tuple<ValueArrayPtr<float_t>, int64_t>, tuple<ValueArrayPtr<double_t>, int64_t>,
              tuple<ValueArrayPtr<std::string>, int64_t>, tuple<Table::PartitionPtr, int64_t>,
              tuple<Table::PartitionVectorPtr, int64_t>>;
  using Operator<Extract, ArgumentTypes...>::Operator;
  template <typename... Args> bool checkArguments(Args const&... args) {
    return CallableOperator::checkArguments(args...);
  }

  bool checkArguments(ComplexExpression const& expr, int64_t /*index*/) {
    return expr.getHead().getName() == "List";
  }

  void operator()(ComplexExpression const& expr, int64_t index) {
    if(index <= 0) {
      throw std::underflow_error("tried to call Extract with index " + std::to_string(index));
    }
    if(index <= offset) {
      return; // already passed
    }
    index -= offset;
    auto size = expr.getArguments().size();
    offset += size;
    if(index <= size) {
      this->pushUp(expr.cloneArgument(index - 1));
      found = true;
    }
  }
  template <typename T> void operator()(ValueArrayPtr<T> const& arrayPtr, int64_t index) {
    if(index <= 0) {
      throw std::underflow_error("tried to call Extract with index " + std::to_string(index));
    }
    if(index <= offset) {
      return; // already passed
    }
    index -= offset;
    offset += arrayPtr->length();
    if(index <= arrayPtr->length()) {
      this->pushUp((T)arrayPtr->Value(index - 1));
      found = true;
    }
  }
  void operator()(Table::PartitionPtr const& partition, int64_t index) {
    if(!partition) {
      return;
    }
    if(index <= 0) {
      throw std::underflow_error("tried to call Extract with index " + std::to_string(index));
    }
    if(index <= offset) {
      return; // already passed
    }
    index -= offset;
    offset += partition->length();
    if(index <= partition->length()) {
      this->pushUp(partition->row(index - 1));
      found = true;
    }
  }
  void operator()(Table::PartitionVectorPtr const& partitions, int64_t index) {
    if(!partitions) {
      return;
    }
    if(index <= 0) {
      throw std::underflow_error("tried to call Extract with index " + std::to_string(index));
    }
    if(index <= offset) {
      return; // already passed
    }
    for(auto const& partition : *partitions) {
      if(!partition) {
        continue;
      }
      index -= offset;
      offset += partition->length();
      if(index <= partition->length()) {
        this->pushUp(partition->row(index - 1));
        found = true;
        break;
      }
    }
  }

  void close() override {
    auto failed = !found;
    auto maxIndex = offset;
    offset = 0;
    found = false;
    if(failed) {
      throw std::overflow_error("tried to call Extract with index > " + std::to_string(maxIndex));
    }
  }

private:
  int64_t offset = 0;
  bool found = false;
};
namespace {
boss::engines::bulk::Engine::Register<Extract> const r5("Extract"); // NOLINT
}

template <typename... ArgumentTypes>
class GetIndexOf : public boss::engines::bulk::Operator<GetIndexOf, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<ValueArrayPtr<bool>, bool>, tuple<ValueArrayPtr<int32_t>, int32_t>,
              tuple<ValueArrayPtr<int64_t>, int64_t>, tuple<ValueArrayPtr<float_t>, float_t>,
              tuple<ValueArrayPtr<double_t>, double_t>,
              tuple<ValueArrayPtr<std::string>, std::string>, tuple<ValueArrayPtr<Symbol>, Symbol>>;
  using Operator<GetIndexOf, ArgumentTypes...>::Operator;

  template <typename T> void operator()(ValueArrayPtr<T> const& arrayPtr, T const& value) {
    auto const& arrowArray = *arrayPtr;
    for(int64_t index = 0; index < arrowArray.length(); ++index) {
      if(arrowArray.Value(index) == value) {
        this->pushUp(index + 1);
        return;
      }
    }
    if constexpr(std::is_same_v<T, std::string>) {
      throw std::logic_error("IndexOf: value not found: \"" + value + "\"");
    } else {
      throw std::logic_error("IndexOf: value not found: " + std::to_string(value));
    }
  }
  void operator()(ValueArrayPtr<Symbol> const& arrayPtr, Symbol const& value) {
    auto const& arrowArray = *arrayPtr;
    for(int64_t index = 0; index < arrowArray.length(); ++index) {
      if(arrowArray.Value(index) == value.getName()) {
        this->pushUp(index + 1);
        return;
      }
    }
    throw std::logic_error("IndexOf: value not found: \"" + value.getName() + "\"_");
  }
};
namespace {
boss::engines::bulk::Engine::Register<GetIndexOf> const r6("IndexOf"); // NOLINT
}

template <typename... ArgumentTypes>
class ExtractColumn : public boss::engines::bulk::Operator<ExtractColumn, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Table::PartitionPtr, int64_t>>;
  using Operator<ExtractColumn, ArgumentTypes...>::Operator;
  void operator()(Table::PartitionPtr const& partition, int64_t index) {
    if(partition) {
      this->pushUp(partition->column(index - 1));
    }
  }
};
namespace {
boss::engines::bulk::Engine::Register<ExtractColumn> const r8("Column"); // NOLINT
}

template <typename... ArgumentTypes>
class ExtractPartition : public boss::engines::bulk::Operator<ExtractPartition, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol, int64_t>>;
  using Operator<ExtractPartition, ArgumentTypes...>::Operator;
  void operator()(Symbol const& table, int64_t index) {
    this->pushUp(TableSymbolRegistry::globalInstance().findSymbol(table)->finaliseAndGetPartition(
        index - 1));
  }
};
namespace {
boss::engines::bulk::Engine::Register<ExtractPartition> const r9("Partition"); // NOLINT
}

template <typename... ArgumentTypes>
class FunctionEvaluation
    : public boss::engines::bulk::Operator<FunctionEvaluation, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<bool, ComplexExpression>, tuple<int32_t, ComplexExpression>,
                                 tuple<int64_t, ComplexExpression>,
                                 tuple<Symbol, ComplexExpression, ComplexExpression>,
                                 tuple<Symbol, Symbol, ComplexExpression>,
                                 tuple<ComplexExpression, ComplexExpression, ComplexExpression>,
                                 tuple<ComplexExpression, Symbol, ComplexExpression>>;
  using Operator<FunctionEvaluation, ArgumentTypes...>::Operator;

  template <typename T> void operator()(T body, ComplexExpression const& /*arguments*/) {
    // constant body, nothing to do
    this->pushUp(body);
  }
  template <typename BodyType>
  void operator()(Symbol const& parameter, BodyType const& body,
                  ComplexExpression const& arguments) {
    // single parameter
    operator()("List"_(parameter), body, arguments);
  }
  void operator()(ComplexExpression const& parameterList, Symbol const& body,
                  ComplexExpression const& arguments) {
    operator()(parameterList, Expression(body), arguments);
  }
  template <typename ExpressionType>
  void operator()(ComplexExpression const& parameterList, ExpressionType const& body,
                  ComplexExpression const& arguments) {
    ExpressionArguments parametersArgs = parameterList.getArguments();
    ExpressionArguments argumentsArgs = arguments.getArguments();
    Expression output = body.clone();
    // going through each pair of parameter / argument
    // and set the parameter symbol to be the argument
    auto paramsIt = std::make_move_iterator(parametersArgs.begin());
    auto paramsItEnd = std::make_move_iterator(parametersArgs.end());
    auto argsIt = std::make_move_iterator(argumentsArgs.begin());
    auto argsItEnd = std::make_move_iterator(argumentsArgs.end());
    for(; paramsIt != paramsItEnd && argsIt != argsItEnd; ++paramsIt, ++argsIt) {
      // this is a nested let for each parameter
      auto args = ExpressionArguments();
      args.reserve(3);
      args.emplace_back(std::move(*paramsIt));
      args.emplace_back(std::move(*argsIt));
      args.emplace_back(std::move(output));
      output = ComplexExpression("Let"_, std::move(args));
    }
    this->pushUp(this->evaluateInternal(std::move(output)));
  }
};
namespace {
boss::engines::bulk::Engine::Register<FunctionEvaluation> const r10("Function"); // NOLINT
}

template <typename... ArgumentTypes>
class Let : public boss::engines::bulk::Operator<Let, ArgumentTypes...> {
public:
  using ArgumentTypesT =
      variant<tuple<Symbol, int32_t, ComplexExpression>, tuple<Symbol, int64_t, ComplexExpression>,
              tuple<Symbol, float_t, ComplexExpression>, tuple<Symbol, double_t, ComplexExpression>,
              tuple<Symbol, std::string, ComplexExpression>,
              tuple<Symbol, Symbol, ComplexExpression>,
              tuple<Symbol, ComplexExpression, ComplexExpression>,
              tuple<Symbol, Table::PartitionPtr, ComplexExpression>>;
  using Operator<Let, ArgumentTypes...>::Operator;

  void operator()(Symbol const& symbol, ComplexExpression const& expressionValue,
                  ComplexExpression const& expressionBody) {
    // evaluate the symbol value to set
    auto evaluatedValue = this->evaluateInternal(expressionValue);
    operator()(symbol, std::move(evaluatedValue), expressionBody);
  }

  template <typename SymbolValueType>
  void operator()(Symbol const& symbol, SymbolValueType symbolValue,
                  ComplexExpression const& expressionBody) {
    // temporarly set the symbol
    auto& symbolRegistry = DefaultSymbolRegistry::instance();
    auto oldSymbol = symbolRegistry.swapSymbol(symbol, symbolValue);
    // evaluate and push the expression
    this->pushUp(this->evaluateInternal(expressionBody));
    // reset the symbol to the old value
    symbolRegistry.registerSymbol(symbol, std::move(oldSymbol));
  }
};
namespace {
boss::engines::bulk::Engine::Register<Let> const r11("Let"); // NOLINT
}

template <typename... ArgumentTypes>
class DateObject : public boss::engines::bulk::Operator<DateObject, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<std::string>, tuple<ValueArrayPtr<std::string>>>;
  using Operator<DateObject, ArgumentTypes...>::Operator;

  void operator()(std::string const& str) { this->pushUp(toUnixTime(str)); }
  void operator()(ValueArrayPtr<std::string> const& strArrayPtr) {
    auto const& strArray = *strArrayPtr;
    auto length = strArray.length();
    auto outputBuilder = ValueBuilder<int32_t>(length);
    for(int64_t index = 0; index < length; ++index) {
      outputBuilder[index] = toUnixTime(strArray.Value(index));
    }
    this->pushUp((ValueArrayPtr<int32_t>)outputBuilder);
  }

private:
  template <typename StringType> static int32_t toUnixTime(StringType const& str) {
    std::istringstream iss;
    iss.str(std::string(str));
    struct std::tm tm = {};
    iss >> std::get_time(&tm, "%Y-%m-%d");
    auto t = std::mktime(&tm);
    static auto const hoursInADay = 24;
    return (int32_t)(std::chrono::duration_cast<std::chrono::hours>(
                         std::chrono::system_clock::from_time_t(t).time_since_epoch())
                         .count() /
                     hoursInADay);
  }
};
namespace {
boss::engines::bulk::Engine::Register<DateObject> const r12("DateObject"); // NOLINT
}

template <typename... ArgumentTypes>
class YearOperator : public boss::engines::bulk::Operator<YearOperator, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<int32_t>>, tuple<ValueArrayPtr<int64_t>>,
                                 tuple<int32_t>, tuple<int64_t>>;
  using Operator<YearOperator, ArgumentTypes...>::Operator;

  template <typename T> void operator()(T const& value) { this->pushUp(toYear(value)); }
  template <typename T> void operator()(ValueArrayPtr<T> const& arrayPtr) {
    this->pushUp(bulkOp(*arrayPtr));
  }

private:
  // from https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_16
  static double_t constexpr coeff = 1.0 / (31536000.0                                // NOLINT
                                           + 86400.0 / 4.0                           // NOLINT
                                           - 86400.0 / 100.0                         // NOLINT
                                           + 86400.0 / 400.0);                       // NOLINT
  static double_t constexpr offset = 1900.002 + coeff * (70.0 * 31536000.0           // NOLINT
                                                         + 69.0 * 86400.0 / 4.0      // NOLINT
                                                         - 86400.0 / 100.0           // NOLINT
                                                         - 299.0 * 86400.0 / 400.0); // NOLINT
  static double_t constexpr multiplier = coeff * 24.0 * 60.0 * 60.0;                 // NOLINT
  template <typename T> static T toYear(T const& epochInDays) {
    return epochInDays * multiplier + offset;
  }

  template <typename T> auto bulkOp(ValueArray<T> const& lhsArray) {
    auto outputBuilder = ValueBuilder<T>(lhsArray.length());
#ifdef __AVX2__
    if constexpr(std::is_same_v<T, int32_t>) {
      outputBuilder.computeSIMD32(
          [](int32_t* memAddr, __m256i in) {
            auto mul = _mm256_set1_ps(multiplier);        // NOLINT
            auto off = _mm256_set1_ps(offset);            // NOLINT
            auto inps = _mm256_cvtepi32_ps(in);           // NOLINT
            auto outps = _mm256_fmadd_ps(inps, mul, off); // NOLINT
            auto output = _mm256_cvtps_epi32(outps);      // NOLINT
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray);
    } else if constexpr(std::is_same_v<T, int64_t>) {
      outputBuilder.computeSIMD64(
          [](int64_t* memAddr, __m256i in) {
            auto mul = _mm256_set1_pd(multiplier);        // NOLINT
            auto off = _mm256_set1_pd(offset);            // NOLINT
            auto inpd = _mm256_cvtepi64_pd(in);           // NOLINT
            auto outpd = _mm256_fmadd_pd(inpd, mul, off); // NOLINT
            auto output = _mm256_cvtpd_epi64(outpd);      // NOLINT
            _mm256_store_si256(reinterpret_cast<__m256i*>(memAddr), output);
          },
          lhsArray);
    } else
#endif // __AVX2__
    {
      for(int64_t index = 0; index < lhsArray.length(); ++index) {
        outputBuilder[index] = toYear(lhsArray.Value(index));
      }
    }
    return (ValueArrayPtr<T>)outputBuilder;
  }
};
namespace {
boss::engines::bulk::Engine::Register<YearOperator> const r12b("Year"); // NOLINT
}

template <typename... ArgumentTypes>
class Length : public boss::engines::bulk::Operator<Length, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<ValueArrayPtr<bool>>, tuple<ValueArrayPtr<int32_t>>,
                                 tuple<ValueArrayPtr<int64_t>>, tuple<ValueArrayPtr<float_t>>,
                                 tuple<ValueArrayPtr<double_t>>, tuple<ValueArrayPtr<std::string>>,
                                 tuple<Table::PartitionPtr>, tuple<Table::PartitionVectorPtr>,
                                 tuple<ComplexExpression>>;
  using Operator<Length, ArgumentTypes...>::Operator;
  template <typename... Args> bool checkArguments(Args const&... args) {
    return CallableOperator::checkArguments(args...);
  }

  bool checkArguments(ComplexExpression const& expr) { return expr.getHead().getName() == "List"; }

  void operator()(ComplexExpression const& expr) { totalLength += expr.getArguments().size(); }
  template <typename ArrayPtr> void operator()(ArrayPtr const& arrayPtr) {
    if(arrayPtr) {
      totalLength += arrayPtr->length();
    }
  }
  void operator()(Table::PartitionVectorPtr const& partitions) {
    totalLength += std::accumulate(partitions->begin(), partitions->end(), (int64_t)0,
                                   [](int64_t total, auto const& partition) {
                                     return partition ? total + partition->length() : total;
                                   });
  }

  void close() override {
    this->pushUp(totalLength);
    totalLength = 0;
  }

private:
  int64_t totalLength = 0;
};
namespace {
boss::engines::bulk::Engine::Register<Length> const r13("Length"); // NOLINT
}

template <typename... ArgumentTypes>
class Unevaluated : public boss::engines::bulk::Operator<Unevaluated, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Expression>>;
  using Operator<Unevaluated, ArgumentTypes...>::Operator;
  void operator()(Expression const& expr) { this->pushUp(expr); }
};
namespace {
boss::engines::bulk::Engine::Register<Unevaluated> const r14("Unevaluated"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class Evaluate : public boss::engines::bulk::Operator<Evaluate, ArgumentTypes...> {
public:
  static constexpr int MAX_COLUMNS = 5;
  using ColumnSymbolsT = RepeatedArgumentTypeOfAnySize_t<1, MAX_COLUMNS, Symbol>;
  using ArgumentTypesT =
      variant_merge_t<variant<tuple<Table::PartitionPtr>>,
                      ArgumentTypeCombine_t<variant<tuple<Table::PartitionPtr>>, ColumnSymbolsT>>;
  using Operator<Evaluate, ArgumentTypes...>::Operator;

  template <typename... Symbols>
  void operator()(Table::PartitionPtr const& partition, Symbols const&... symbols) {
    if(!partition || partition->length() == 0) {
      this->pushUp(partition);
      return;
    }
    storeAsCleanOrDirty(partition, symbols...);
  }

  void close() override {
    // clean dirty partitions
    auto& cleanPartitions = *cleanPartitionsPtr;
    auto& dirtyPartitions = *dirtyPartitionsPtr;
    if(cleanPartitions.empty()) { // can do the evaluation only if we have at least one clean row
      this->pushUp(dirtyPartitionsPtr);
      return;
    }
    for(int64_t partitionIndex = 0; partitionIndex < dirtyPartitions.size(); ++partitionIndex) {
      cleanAndPushDirtyPartition(dirtyPartitions[partitionIndex],
                                 dirtyPartitionsColumns[partitionIndex], true, true);
    }
    // release data
    cleanPartitions.clear();
    dirtyPartitions.clear();
    dirtyPartitionsColumns.clear();
  }

private:
  Table::PartitionVectorPtr cleanPartitionsPtr = std::make_unique<Table::PartitionVector>();
  Table::PartitionVectorPtr dirtyPartitionsPtr = std::make_unique<Table::PartitionVector>();
  std::vector<ExpressionArguments> dirtyPartitionsColumns;

  static constexpr int NUM_FIXED_ARGUMENTS_TO_ADD =
      3; // clean partitions, dirty partitions, column index

  template <typename... Symbols>
  void storeAsCleanOrDirty(Table::PartitionPtr const& partition, Symbols const&... symbolFilter) {
    bool cleanPartition = true;
    ExpressionArguments columns;
    columns.reserve(partition->num_fields() + NUM_FIXED_ARGUMENTS_TO_ADD);
    for(int64_t columnIndex = 0; columnIndex < partition->num_fields(); ++columnIndex) {
      auto column = partition->column(columnIndex);
      auto const& columnName = partition->getSchema()[columnIndex]->name();

      bool isCleanColumn = !std::holds_alternative<std::shared_ptr<ComplexExpressionArray>>(column);
      if(isCleanColumn) {
        if(!cleanPartition) {
          // still store it for partition evaluation (but this column won't be evaluated)
          columns.emplace_back(std::move(column));
        }
        continue;
      }
      if constexpr(sizeof...(Symbols) > 0) { // not filtering out if no symbols
        bool columnNeedEvaluation = ((symbolFilter.getName() == columnName) || ...);
        if(!columnNeedEvaluation) {
          if(!cleanPartition) {
            // still store it for partition evaluation (but this column won't be evaluated)
            columns.emplace_back(std::move(column));
          }
          continue;
        }
      }
      if(cleanPartition) {
        cleanPartition = false;
        // from now on, we store the columns for partition evaluation
        for(int64_t i = 0; i < columnIndex; ++i) {
          columns.emplace_back(partition->column(i));
        }
      }
      // not a clean partition AND we want to evaluate it
      // convert into a ComplexExpression to mark it for evaluation
      auto const& complexArrayPtr = get<std::shared_ptr<ComplexExpressionArray>>(column);
      columns.emplace_back(fullyConvertToComplexExpression(*complexArrayPtr));
    }
    if(cleanPartition) {
      cleanPartitionsPtr->push_back(partition);
      this->pushUp(partition); // already push clean partitions (no need to be pipeline breaker)
      return;
    }
    // try to evaluate the dirty partitions without clean partitions
    if(!cleanAndPushDirtyPartition(partition, columns, false, false)) {
      // if not, we need to evaluate them again after getting all the clean partitions
      dirtyPartitionsPtr->push_back(partition);
      dirtyPartitionsColumns.emplace_back(std::move(columns));
    }
  }

  ComplexExpression fullyConvertToComplexExpression(ComplexExpressionArray const& complexArray) {
    ExpressionArguments args;
    args.reserve(complexArray.num_fields());
    for(int64_t columnIndex = 0; columnIndex < complexArray.num_fields(); ++columnIndex) {
      auto expr = complexArray.column(columnIndex);
      if(auto* maybeComplexArray = std::get_if<std::shared_ptr<ComplexExpressionArray>>(&expr)) {
        expr = fullyConvertToComplexExpression(**maybeComplexArray);
      }
      args.emplace_back(std::move(expr));
    }
    return ComplexExpression(complexArray.getHead(), std::move(args));
  }

  bool cleanAndPushDirtyPartition(Table::PartitionPtr const& dirtyPartitionPtr,
                                  ExpressionArguments& columns, bool passAdditionalArguments,
                                  bool alwaysPush) {
    bool fullyEvaluated = true;
    for(int64_t columnIndex = 0; columnIndex < columns.size(); ++columnIndex) {
      auto& column = columns[columnIndex];
      auto onEval = boss::utilities::overload(
          [&column, &fullyEvaluated](Expression const& evaluatedColumn, bool evaluated) {
            if(evaluated) {
              column = evaluatedColumn.clone();
            } else {
              fullyEvaluated = false;
            }
          },
          [&column, &fullyEvaluated](Expression&& evaluatedColumn, bool evaluated) {
            if(evaluated) {
              column = std::move(evaluatedColumn);
            } else {
              fullyEvaluated = false;
            }
          });
      if(auto const* maybeDirtyColumn = std::get_if<ComplexExpression>(&column)) {
        // this is a column to evaluate
        if(!passAdditionalArguments) {
          // try to evaluate without adding the additional arguments
          this->evaluateInternal(maybeDirtyColumn->clone(), onEval);
          continue;
        }
        // re-write it and send for evaluation
        ExpressionArguments args = maybeDirtyColumn->getArguments();
        args.reserve(args.size() + 3);
        args.emplace_back(cleanPartitionsPtr);
        args.emplace_back(dirtyPartitionPtr);
        args.emplace_back(columnIndex + 1);
        this->evaluateInternal(ComplexExpression(maybeDirtyColumn->getHead(), std::move(args)),
                               onEval);
      }
    }
    if(alwaysPush || fullyEvaluated) {
      // build a relation with all the projected columns
      auto builder = ComplexExpressionArrayBuilder("List"_, dirtyPartitionPtr->getSchema(),
                                                   dirtyPartitionPtr->length());
      builder.appendExpression(ComplexExpression("List"_, std::move(columns)));
      auto output = (Table::PartitionPtr)builder;
      output->setGlobalIndexes(dirtyPartitionPtr->getGlobalIndexes());
      output->setTablePartitionIndexes(*dirtyPartitionPtr);
      this->pushUp(std::move(output));
      return true;
    }
    return false;
  }
};
namespace {
boss::engines::bulk::Engine::Register<Evaluate> const r15("Evaluate"); // NOLINT
} // namespace

template <typename... ArgumentTypes>
class SetSymbol : public boss::engines::bulk::Operator<SetSymbol, ArgumentTypes...> {
public:
  using ArgumentTypesT = variant<tuple<Symbol, bool>, tuple<Symbol, int64_t>,
                                 tuple<Symbol, double_t>, tuple<Symbol, std::string>>;
  using Operator<SetSymbol, ArgumentTypes...>::Operator;
  template <typename T> void operator()(Symbol const& symbol, T const& value) {
    auto& symbolRegistry = DefaultSymbolRegistry::globalInstance();
    symbolRegistry.registerSymbol(symbol, value);
  }
};
namespace {
boss::engines::bulk::Engine::Register<SetSymbol> const r200("Set"); // NOLINT
}

template <typename... ArgumentTypes>
class Random : public boss::engines::bulk::Operator<Random, ArgumentTypes...> {
public:
  static constexpr int MAX_EXPRESSIONS = 64;
  using ArgumentTypesT = RepeatedArgumentTypeOfAnySize_t<1, MAX_EXPRESSIONS, Expression>;
  using Operator<Random, ArgumentTypes...>::Operator;
  template <typename... E> void operator()(E const&... expr) {
    std::array<Expression const*, sizeof...(E)> list = {&expr...};
    static std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, sizeof...(E) - 1);
    this->pushUp(*list[distr(gen)]); // NOLINT
  }
};
namespace {
boss::engines::bulk::Engine::Register<Random> const r201("Random"); // NOLINT
}

} // namespace boss::engines::bulk
