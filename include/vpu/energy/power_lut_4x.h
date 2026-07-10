// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_ENERGY_POWER_LUT_4X_H
#define VPUNN_VPU_ENERGY_POWER_LUT_4X_H

#include "power.h"
#include "power_lut_2x.h"

namespace VPUNN {

class PowerVPU40 : protected PowerVPU27 {
public:
    static details::DevicePowerLUT make_lut() {
        details::DevicePowerLUT this_device = PowerVPU27::make_lut();
        this_device.devices = {VPUNN::VPUDevice::VPU_4_0};
        return this_device;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_ENERGY_POWER_LUT_4X_H
