// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PREPROCESSING_ADAPTERS_17_H
#define VPUNN_PREPROCESSING_ADAPTERS_17_H

#include <core/logger.h>
#include <vpu/types.h>

#include <sstream>  // for error formatting
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "preprocessing_adapters_bundle.h"  // inside here are all previous adapters
#include "preprocessing_adapters_utils.h"

namespace VPUNN {
///  AVEPOOL -> DW_CONV and CONV <16ch + autopad -> CM_CONV retained from NN6X.
///  NEW: LAYER_NORM and REDUCE_SUMSQUARES are explicitly unsupported and will throw.
class NN7XInputAdapter : public NN6XInputAdapter {
public:
    /// @brief Maps operations to the ones supported by the NPU_RESERVED_1 descriptor interface.
    ///
    /// AVEPOOL is remapped to DW_CONVOLUTION.
    /// CONVOLUTION with <16 input channels and input_autopad is remapped to CM_CONVOLUTION.
    /// LAYER_NORM and REDUCE_SUMSQUARES are explicitly unsupported and will throw std::out_of_range.
    ///
    /// @throws std::out_of_range if the operation is LAYER_NORM or REDUCE_SUMSQUARES.
    static Operation mock_replace_operations(const Operation in_operation, const DPUWorkload& wl) {
        if (in_operation == Operation::LAYER_NORM) {
            throw std::out_of_range(
                    "[ERROR] NN7XInputAdapter: Operation LAYER_NORM is not supported by the NPU_RESERVED_1 NN descriptor "
                    "(interface 17). Remove or remap this operation before inference.");
        }
        if (in_operation == Operation::REDUCE_SUMSQUARES) {
            throw std::out_of_range(
                    "[ERROR] NN7XInputAdapter: Operation REDUCE_SUMSQUARES is not supported by the NPU_RESERVED_1 NN descriptor "
                    "(interface 17). This operation has been defeatured.");
        }
        // Delegate all other operations to NN6X logic (AVEPOOL -> DW_CONV, CONV autopad -> CM_CONV).
        return NN6XInputAdapter::mock_replace_operations(in_operation, wl);
    }

    using NN6XInputAdapter::avoid_untrained_space;
};
}  // namespace VPUNN

#endif  // VPUNN_PREPROCESSING_ADAPTERS_17_H
