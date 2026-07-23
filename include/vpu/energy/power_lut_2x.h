// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_ENERGY_POWER_LUT_2X_H
#define VPUNN_VPU_ENERGY_POWER_LUT_2X_H

#include "power.h"

namespace VPUNN {
class PowerVPU2x {
public:
    constexpr static float getFP_overI8_maxPower_ratio() {
        return 0.87f;  ///< this implies INT is more power hungry (=> power virus int  is the max!)
    }
    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt =
                details::PowerFactor{1.0f,  ///< adjustor
                                     {      ///< values_lut
                                      {VPUNN::Operation::CONVOLUTION,
                                       {{4, 0.87f}, {5, 0.92f}, {6, 1.0f}, {7, 0.95f}, {8, 0.86f}, {9, 0.87f}}},
                                      {VPUNN::Operation::DW_CONVOLUTION, {{6, 5.84f}}},
                                      {VPUNN::Operation::AVEPOOL, {{6, 32.60f}}},
                                      {VPUNN::Operation::MAXPOOL, {{6, 5.29f}}},
                                      {VPUNN::Operation::ELTWISE, {{7, 232.71f}}}}};

        const float fp_ratio = getFP_overI8_maxPower_ratio();
        auto fp16_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION,
                                                  {{4, 0.87f * fp_ratio},
                                                   {5, 0.92f * fp_ratio},
                                                   {6, 1.0f * fp_ratio},
                                                   {7, 0.95f * fp_ratio},
                                                   {8, 0.86f * fp_ratio},
                                                   {9, 0.87f * fp_ratio}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 5.84f * fp_ratio}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 32.60f * fp_ratio}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 5.29f * fp_ratio}}},
                                                 {VPUNN::Operation::ELTWISE, {{7, 232.71f * fp_ratio}}}}};
        // VPU2.0 values (Op type: {log2(input_channels): power_factor}))
        return details::DevicePowerLUT{
                {VPUNN::VPUDevice::VPU_2_0, VPUNN::VPUDevice::VPU_2_1},  ///< devices sharing VPU 2.x power LUT
                0.87f,                                                   ///< maxvirus
                {{VPUNN::MPEEngine::SCL,
                  {///< factors
                   {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                   {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};

class PowerVPU27 {
public:
    constexpr static float getFP_overI8_maxPower_ratio() {
        return 1.3f;  // float more hungry
    }

    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION, {{6, 1.0f}}},
                                                 {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 21.0f}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 21.0f}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 11.0f}}},
                                                 {VPUNN::Operation::ELTWISE, {{8, 5.0f}}}}};

        const float fp_ratio = getFP_overI8_maxPower_ratio();
        auto fp16_scl_pt = details::PowerFactor{1.0f,  ///< adjustor
                                                {      ///< values_lut
                                                 {VPUNN::Operation::CONVOLUTION, {{6, 1.0f * fp_ratio}}},
                                                 {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f * fp_ratio}}},
                                                 {VPUNN::Operation::DW_CONVOLUTION, {{6, 21.0f * fp_ratio}}},
                                                 {VPUNN::Operation::AVEPOOL, {{6, 21.0f * fp_ratio}}},
                                                 {VPUNN::Operation::MAXPOOL, {{6, 11.0f * fp_ratio}}},
                                                 {VPUNN::Operation::ELTWISE, {{8, 5.0f * fp_ratio}}}}};

        return details::DevicePowerLUT{{VPUNN::VPUDevice::VPU_2_7},  ///< devices
                                       1.3f,                         ///< maxvirus
                                       {{VPUNN::MPEEngine::SCL,
                                         {///< factors
                                          {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                                          {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_ENERGY_POWER_LUT_2X_H
