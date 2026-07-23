// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_ENERGY_POWER_LUT_5X_H
#define VPUNN_VPU_ENERGY_POWER_LUT_5X_H

#include "power.h"

namespace VPUNN {

class PowerVPU50 {
public:
    static details::DevicePowerLUT make_lut() {
        auto int8_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                                {
                                                        ///< values_lut
                                                        {VPUNN::Operation::CONVOLUTION, {{6, 0.61f}}},
                                                        {VPUNN::Operation::CM_CONVOLUTION, {{6, 0.61f}}},
                                                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 9.63f}}},
                                                        {VPUNN::Operation::AVEPOOL, {{6, 3.57f}}},
                                                        {VPUNN::Operation::MAXPOOL, {{6, 3.23f}}},
                                                        {VPUNN::Operation::ELTWISE, {{8, 29.44f}}},
                                                        {VPUNN::Operation::ELTWISE_MUL, {{8, 29.87f}}},
                                                        {
                                                                VPUNN::Operation::LAYER_NORM,
                                                                {{8, 5.0f}}  // unknown
                                                        },
                                                }};

        auto fp8_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                               {
                                                       ///< values_lut
                                                       {VPUNN::Operation::CONVOLUTION, {{6, 0.72f}}},
                                                       {VPUNN::Operation::CM_CONVOLUTION, {{6, 0.72f}}},
                                                       {VPUNN::Operation::DW_CONVOLUTION, {{6, 11.51f}}},
                                                       {VPUNN::Operation::AVEPOOL, {{6, 3.88f}}},
                                                       {VPUNN::Operation::MAXPOOL, {{6, 2.85f}}},
                                                       {VPUNN::Operation::ELTWISE, {{8, 39.87f}}},
                                                       {VPUNN::Operation::ELTWISE_MUL, {{8, 33.30f}}},
                                                       {
                                                               VPUNN::Operation::LAYER_NORM,
                                                               {{8, 5.0f}}  // unknown
                                                       },
                                               }};

        auto fp16_scl_pt = details::PowerFactor{0.88f,  ///< adjustor
                                                {
                                                        ///< values_lut
                                                        {VPUNN::Operation::CONVOLUTION, {{6, 1.0f}}},
                                                        {VPUNN::Operation::CM_CONVOLUTION, {{6, 1.0f}}},
                                                        {VPUNN::Operation::DW_CONVOLUTION, {{6, 13.82f}}},
                                                        {VPUNN::Operation::AVEPOOL, {{6, 3.88f}}},
                                                        {VPUNN::Operation::MAXPOOL, {{6, 2.91f}}},
                                                        {VPUNN::Operation::ELTWISE, {{8, 29.12f}}},
                                                        {VPUNN::Operation::ELTWISE_MUL, {{8, 26.62f}}},
                                                        {VPUNN::Operation::LAYER_NORM, {{8, 5.0f}}},
                                                }};

        return details::DevicePowerLUT{
                {VPUNN::VPUDevice::NPU_5_0, VPUNN::VPUDevice::NPU_5_0_W},  ///< devices sharing NPU 5.x power LUT
                1.0f,                                                      ///< maxvirus
                {{VPUNN::MPEEngine::SCL,
                  {///< factors
                   {details::ComputePowerTypeClass::INT8, std::move(int8_scl_pt)},
                   {details::ComputePowerTypeClass::FP8, std::move(fp8_scl_pt)},
                   {details::ComputePowerTypeClass::FP16, std::move(fp16_scl_pt)}}}}};
    }
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_ENERGY_POWER_LUT_5X_H
