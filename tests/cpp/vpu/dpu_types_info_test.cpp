// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dpu_types_info.h"

#include <gtest/gtest.h>

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// Test fixture for free functions declared in dpu_types_info.h
class TestDpuTypesInfo : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Verifies that dCIM_32x128 is recognised as a dCIM execution mode.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_dCIM_32x128_IsTrue) {
    EXPECT_TRUE(is_dcim_execution_mode(ExecutionMode::dCIM_32x128));
}

/// Verifies that dCIM_64x128 is recognised as a dCIM execution mode.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_dCIM_64x128_IsTrue) {
    EXPECT_TRUE(is_dcim_execution_mode(ExecutionMode::dCIM_64x128));
}

/// Verifies that CUBOID_16x16 is NOT a dCIM execution mode.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_CUBOID_16x16_IsFalse) {
    EXPECT_FALSE(is_dcim_execution_mode(ExecutionMode::CUBOID_16x16));
}

/// Verifies that CUBOID_8x16 is NOT a dCIM execution mode.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_CUBOID_8x16_IsFalse) {
    EXPECT_FALSE(is_dcim_execution_mode(ExecutionMode::CUBOID_8x16));
}

/// Verifies that CUBOID_4x16 is NOT a dCIM execution mode.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_CUBOID_4x16_IsFalse) {
    EXPECT_FALSE(is_dcim_execution_mode(ExecutionMode::CUBOID_4x16));
}

/// Verifies the function is a true constexpr: all checks are evaluated at compile time.
/// If this test compiles, every static_assert passed at zero runtime cost.
TEST_F(TestDpuTypesInfo, IsDcimExecutionMode_IsConstexprEvaluable) {
    static_assert(is_dcim_execution_mode(ExecutionMode::dCIM_32x128),   "dCIM_32x128 must be constexpr-true");
    static_assert(is_dcim_execution_mode(ExecutionMode::dCIM_64x128),   "dCIM_64x128 must be constexpr-true");
    static_assert(!is_dcim_execution_mode(ExecutionMode::CUBOID_16x16), "CUBOID_16x16 must be constexpr-false");
    static_assert(!is_dcim_execution_mode(ExecutionMode::CUBOID_8x16),  "CUBOID_8x16 must be constexpr-false");
    static_assert(!is_dcim_execution_mode(ExecutionMode::CUBOID_4x16),  "CUBOID_4x16 must be constexpr-false");
    SUCCEED();  // reaching here means all compile-time assertions passed
}

}  // namespace VPUNN_unit_tests
