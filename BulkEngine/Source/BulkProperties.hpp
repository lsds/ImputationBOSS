#pragma once
#include <ExpressionUtilities.hpp>
using boss::utilities::operator""_;

namespace boss::engines::bulk {

class Properties {
public:
  static int64_t getMicroBatchesMaxSize() {
    static auto const& val =
        initialiseOrGetProperty("MicroBatchesMaxSize"_, (int64_t)1000); // NOLINT
    return val;
  }

  static bool getUseMemoryMappedFiles() {
    static auto const& val = initialiseOrGetProperty("UseMemoryMappedFiles"_, true);
    return val;
  }

  static bool getForceToPreserveInsertionOrder() {
    static auto const& val = initialiseOrGetProperty("ForceToPreserveInsertionOrder"_, false);
    return val;
  }

  static bool getEnableOrderPreservationCache() {
    static auto const& val = initialiseOrGetProperty("EnableOrderPreservationCache"_, false);
    return val;
  }

  static bool getEnableOrderPreservationCacheForJoins() {
    static auto const& val = initialiseOrGetProperty("EnableOrderPreservationCacheForJoins"_, false);
    return val;
  }

  static bool debugOutputRelationalOps() {
    static auto const& val = initialiseOrGetProperty("DebugOutputRelationalOps"_, false);
    return val;
  }

  static bool getForceNoOpForAtoms() {
    static auto const& val = initialiseOrGetProperty("ForceNoOpForAtoms"_, false);
    return val;
  }

  static bool getDisableExpressionPartitioning() {
    static auto const& val = initialiseOrGetProperty("DisableExpressionPartitioning"_, false);
    return val;
  }

  static bool getDisableExpressionDecomposition() {
    static auto const& val = initialiseOrGetProperty("DisableExpressionDecomposition"_, false);
    return val;
  }

private:
  template <typename T>
  static T const& initialiseOrGetProperty(Symbol const& name, T&& defaultValue);
};

} // namespace boss::engines::bulk