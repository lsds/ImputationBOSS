#pragma once

namespace boss::engines::bulk {

/* variant_merge */
template <typename... Args> struct variant_merge;
template <typename... Args0> struct variant_merge<std::variant<Args0...>> {
  using type = std::variant<Args0...>;
};
template <typename... Args0, typename... Args1>
struct variant_merge<std::variant<Args0...>, std::variant<Args1...>> {
  using type = std::variant<Args0..., Args1...>;
};
template <typename... Args0, typename... Args1, typename... Others>
struct variant_merge<std::variant<Args0...>, std::variant<Args1...>, Others...> {
  using type =
      typename variant_merge<std::variant<Args0...>,
                             typename variant_merge<std::variant<Args1...>, Others...>::type>::type;
};
template <typename... Args> using variant_merge_t = typename variant_merge<Args...>::type;

/* tuple_merge */
template <typename... Args> struct tuple_merge;
template <typename... Args0> struct tuple_merge<std::tuple<Args0...>> {
  using type = std::tuple<Args0...>;
};
template <typename... Args0, typename... Args1>
struct tuple_merge<std::tuple<Args0...>, std::tuple<Args1...>> {
  using type = std::tuple<Args0..., Args1...>;
};
template <typename... Args0, typename... Args1, typename... Others>
struct tuple_merge<std::tuple<Args0...>, std::tuple<Args1...>, Others...> {
  using type =
      typename tuple_merge<std::tuple<Args0...>,
                           typename tuple_merge<std::tuple<Args1...>, Others...>::type>::type;
};
template <typename... Args> using tuple_merge_t = typename tuple_merge<Args...>::type;

/* variant_tuple_merge */
template <typename... Args> struct variant_tuple_merge;
template <typename... Args0> struct variant_tuple_merge<std::variant<std::tuple<Args0...>>> {
  using type = std::variant<std::tuple<Args0...>>;
};
template <typename... Args0, typename... Args1>
struct variant_tuple_merge<std::variant<std::tuple<Args0...>>, std::variant<std::tuple<Args1...>>> {
  using type = std::variant<std::tuple<Args0..., Args1...>>;
};
template <typename... Args0, typename... Args1, typename... Others>
struct variant_tuple_merge<std::variant<std::tuple<Args0...>>, std::variant<std::tuple<Args1...>>,
                           Others...> {
  using type = typename variant_tuple_merge<
      std::variant<std::tuple<Args0...>>,
      typename variant_tuple_merge<std::variant<std::tuple<Args1...>>, Others...>::type>::type;
};
template <typename... Args>
using variant_tuple_merge_t = typename variant_tuple_merge<Args...>::type;

/* tuple_amend */
template <typename T, typename... Args> struct tuple_amend;
template <typename... Args0, typename... Args1> struct tuple_amend<std::tuple<Args0...>, Args1...> {
  using type = std::tuple<Args0..., Args1...>;
};
template <typename... Args> using tuple_amend_t = typename tuple_amend<Args...>::type;

/* variant_to_tuple */
template <typename T> struct variant_to_tuple;
template <typename... Args0> struct variant_to_tuple<std::variant<Args0...>> {
  using type = std::tuple<Args0...>;
};
template <typename Args> using variant_to_tuple_t = typename variant_to_tuple<Args>::type;

/* tuple_to_variant */
template <typename T> struct tuple_to_variant;
template <typename... Args0> struct tuple_to_variant<std::tuple<Args0...>> {
  using type = std::variant<Args0...>;
};
template <typename Args> using tuple_to_variant_t = typename tuple_to_variant<Args>::type;

/* ArgumentTypeReverseCombine */
template <typename T, typename U> struct ArgumentTypeReverseCombine;
template <typename Tuples0, typename... Tuples1>
struct ArgumentTypeReverseCombine<std::variant<Tuples0>, std::variant<Tuples1...>> {
  using type = std::variant<tuple_merge_t<Tuples1, Tuples0>...>;
};
template <typename... Args>
using ArgumentTypeReverseCombine_t = typename ArgumentTypeReverseCombine<Args...>::type;

/* ArgumentTypeCombine */
template <typename T, typename U> struct ArgumentTypeCombine;
template <typename Tuples0, typename... Tuples1>
struct ArgumentTypeCombine<std::variant<Tuples0>, std::variant<Tuples1...>> {
  using type = std::variant<tuple_merge_t<Tuples0, Tuples1>...>;
};
template <typename Tuples0, typename... Tuples1, typename VariantToMerge>
struct ArgumentTypeCombine<std::variant<Tuples0, Tuples1...>, VariantToMerge> {
  using type = typename ArgumentTypeCombine<
      std::variant<Tuples1...>,
      ArgumentTypeReverseCombine_t<std::variant<Tuples0>, VariantToMerge>>::type;
};
template <typename... Args>
using ArgumentTypeCombine_t = typename ArgumentTypeCombine<Args...>::type;

/* RepeatedArgumentTypeOfAnySize */
template <int minSize, int maxSize, typename Type, typename dummy = void,
          typename InitialTupleType = std::tuple<>, typename... TupleTypes>
class RepeatedArgumentTypeOfAnySize;
template <int minSize, int maxSize, typename Type, typename InitialTupleType>
class RepeatedArgumentTypeOfAnySize<minSize, maxSize, Type, std::enable_if_t<(minSize > 1)>,
                                    InitialTupleType> {
public:
  using type = typename RepeatedArgumentTypeOfAnySize<minSize - 1, maxSize - 1, Type, void,
                                                      tuple_amend_t<InitialTupleType, Type>>::type;
};
template <int maxSize, typename Type, typename InitialTupleType, typename... TupleTypes>
class RepeatedArgumentTypeOfAnySize<1, maxSize, Type, std::enable_if_t<(maxSize > 1)>,
                                    InitialTupleType, TupleTypes...> {
public:
  using type = typename RepeatedArgumentTypeOfAnySize<1, maxSize - 1, Type, void, InitialTupleType,
                                                      tuple_amend_t<InitialTupleType, Type>,
                                                      tuple_amend_t<TupleTypes, Type>...>::type;
};
template <typename Type, typename InitialTupleType, typename... TupleTypes>
class RepeatedArgumentTypeOfAnySize<1, 1, Type, void, InitialTupleType, TupleTypes...> {
public:
  using type = variant<tuple_amend_t<InitialTupleType, Type>, tuple_amend_t<TupleTypes, Type>...>;
};
template <int minSize, int maxSize, typename Type>
using RepeatedArgumentTypeOfAnySize_t =
    typename RepeatedArgumentTypeOfAnySize<minSize, maxSize, Type>::type;

/* AnyCombinationOfArgumentTypes */
template <int maxSize, typename InputTupleType> class AnyCombinationOfArgumentTypes;
template <typename... InputTypes>
class AnyCombinationOfArgumentTypes<2, std::tuple<InputTypes...>> {
public:
  template <typename Input, typename... OutputTuples> struct CombinationalMerge {
    using variant_type = std::variant<tuple_amend_t<OutputTuples, Input>...>;
  };
  using type = variant_merge_t<
      typename CombinationalMerge<InputTypes, std::tuple<InputTypes>...>::variant_type...>;
};
template <typename... InputTypes>
class AnyCombinationOfArgumentTypes<1, std::tuple<InputTypes...>> {
public:
  using type = std::variant<std::tuple<InputTypes>...>;
};
template <int maxSize, typename... InputTypes>
class AnyCombinationOfArgumentTypes<maxSize, std::tuple<InputTypes...>> {
public:
  template <typename Arg0, typename arg1> struct CombinationalMerge;
  template <typename... Tuples0, typename... Tuples1>
  struct CombinationalMerge<std::variant<Tuples0...>, std::variant<Tuples1...>> {
    using variant_type = variant_merge_t<
        typename CombinationalMerge<Tuples0, std::variant<Tuples1...>>::variant_type...>;
  };
  template <typename Tuple0, typename... Tuples1>
  struct CombinationalMerge<Tuple0, std::variant<Tuples1...>> {
    using variant_type = std::variant<tuple_merge_t<Tuple0, Tuples1>...>;
  };
  static constexpr int firstHalf = maxSize / 2;
  static constexpr int secondHalf = (maxSize % 2 == 0) ? maxSize / 2 : maxSize / 2 + 1;
  using type = typename CombinationalMerge<
      typename AnyCombinationOfArgumentTypes<firstHalf, std::tuple<InputTypes...>>::type,
      typename AnyCombinationOfArgumentTypes<secondHalf,
                                             std::tuple<InputTypes...>>::type>::variant_type;
};
template <int fixedSize, typename InputTupleType>
using AnyCombinationOfArgumentTypes_t =
    typename AnyCombinationOfArgumentTypes<fixedSize, InputTupleType>::type;

/* AnyCombinationOfArgumentTypesOfAnySize */
template <int minSize, int maxSize, typename InputTupleType, typename dummy = void>
class AnyCombinationOfArgumentTypesOfAnySize;
template <int minSize, int maxSize, typename InputTupleType>
class AnyCombinationOfArgumentTypesOfAnySize<minSize, maxSize, InputTupleType,
                                             std::enable_if_t<(minSize < maxSize)>> {
public:
  using type = variant_merge_t<
      AnyCombinationOfArgumentTypes_t<minSize, InputTupleType>,
      typename AnyCombinationOfArgumentTypesOfAnySize<minSize + 1, maxSize, InputTupleType>::type>;
};
template <int minSize, int maxSize, typename InputTupleType>
class AnyCombinationOfArgumentTypesOfAnySize<minSize, maxSize, InputTupleType,
                                             std::enable_if_t<(minSize >= maxSize)>> {
public:
  using type = AnyCombinationOfArgumentTypes_t<maxSize, InputTupleType>;
};
template <int minSize, int maxSize, typename InputTupleType>
using AnyCombinationOfArgumentTypesOfAnySize_t =
    typename AnyCombinationOfArgumentTypesOfAnySize<minSize, maxSize, InputTupleType>::type;

} // namespace boss::engines::bulk
