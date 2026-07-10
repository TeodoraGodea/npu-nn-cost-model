// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_NN_DESCRIPTOR_CAPABILITIES_H
#define VPUNN_NN_DESCRIPTOR_CAPABILITIES_H

#include <set>
#include <string>

namespace VPUNN {

/// @brief Capabilities provider that expresses no special NN feature support.
///
/// Used in tests and as an explicit "no capabilities" argument when constructing
/// a preprocessing archetype without a device-specific capabilities set.
/// Must always be passed explicitly.
struct DefaultCapabilities {
    static inline const std::set<std::string> capabilities = {};
    static const std::set<std::string>& get() {
        return capabilities;
    }
};

}  // namespace VPUNN

#endif  // VPUNN_NN_DESCRIPTOR_CAPABILITIES_H
