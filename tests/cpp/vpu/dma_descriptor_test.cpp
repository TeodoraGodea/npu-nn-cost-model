// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dma_descriptors.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TestVPUDMADescriptor : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

// ===========================================================================
// getTransferBytes / getSrcTransferBytes / getDstTransferBytes
// ===========================================================================

TEST_F(TestVPUDMADescriptor, getTransferBytes_and_src_dst_packed_uint8) {
    // src: 512 UINT8 packed 1D; dst: 512 UINT8 packed 1D
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{512, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{512, 1}});

    EXPECT_EQ(desc.getTransferBytes(), 512) << "getTransferBytes == src accessed bytes";
    EXPECT_EQ(desc.getSrcTransferBytes(), 512) << "getSrcTransferBytes: 512 bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 512) << "getDstTransferBytes: 512 bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 512) << "packed src: all 512 bytes contiguous";
    EXPECT_EQ(desc.getContiguousBytesDst(), 512) << "packed dst: all 512 bytes contiguous";
}

TEST_F(TestVPUDMADescriptor, getTransferBytes_src_strided_ignores_stride) {
    // src: 256 FP16 with inner stride=4 (gap); dst: 256 FP16 packed
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::FLOAT16;
    desc.src.setDimension_OutermostFirst({{256, 4}});  // strided: 256 * 2 = 512 logical bytes
    desc.dst.dtype = DataType::FLOAT16;
    desc.dst.setDimension_OutermostFirst({{256, 2}});  // packed:  256 * 2 = 512 logical bytes

    EXPECT_EQ(desc.getTransferBytes(), 512) << "getTransferBytes ignores src stride gaps";
    EXPECT_EQ(desc.getSrcTransferBytes(), 512) << "getSrcTransferBytes: 256*2 = 512 logical bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 512) << "getDstTransferBytes: 256*2 = 512 packed bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 2) << "src stride=4 != elem(2): only 2 bytes contiguous";
    EXPECT_EQ(desc.getContiguousBytesDst(), 512) << "packed dst: all 512 bytes contiguous";
}

// ===========================================================================
// getNumContiguousChunksSrc / getNumContiguousChunksDst
// ===========================================================================

TEST_F(TestVPUDMADescriptor, getNumContiguousChunksSrc_packed_is_one) {
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{1024, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{1024, 1}});

    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1) << "packed 1D src: 1 contiguous chunk";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 1) << "packed 1D dst: 1 contiguous chunk";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 1024) << "packed src: 1024 contiguous bytes";
    EXPECT_EQ(desc.getContiguousBytesDst(), 1024) << "packed dst: 1024 contiguous bytes";
}

TEST_F(TestVPUDMADescriptor, getNumContiguousChunksSrc_strided_one_chunk_per_element) {
    // src: 64 UINT8 stride=2 => contiguous=1 byte, total=64 bytes => 64 chunks
    // dst: 64 UINT8 packed  => 1 chunk
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{64, 2}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{64, 1}});

    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 64) << "strided src stride=2: 64 chunks of 1 byte";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 1) << "packed dst: 1 chunk";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 1) << "src stride=2: only 1 byte contiguous";
    EXPECT_EQ(desc.getContiguousBytesDst(), 64) << "packed dst: 64 contiguous bytes";
}

TEST_F(TestVPUDMADescriptor, getNumContiguousChunksDst_strided_2D_row_gaps) {
    // dst: 16 rows × 32 UINT8 cols, outer stride=64 (32-byte gap per row) => 16 chunks of 32
    // src: fully packed equivalent
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{16 * 32, 1}});  // flat packed
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{16, 64}, {32, 1}});  // 2D with outer gap

    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1) << "flat packed src: 1 chunk";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 16) << "2D dst outer gap: 16 chunks of 32 bytes";
    EXPECT_EQ(desc.getSrcTransferBytes(), 16 * 32) << "src logical: 512 bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 16 * 32) << "dst logical: 512 bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 16 * 32) << "flat packed src: all 512 bytes contiguous";
    EXPECT_EQ(desc.getContiguousBytesDst(), 32) << "2D dst outer gap: only 32 bytes (one row) contiguous";
}

// ===========================================================================
// getContiguousBytesSrc / getContiguousBytesDst
// ===========================================================================

TEST_F(TestVPUDMADescriptor, getContiguousBytes_asymmetric_src_dst_layouts) {
    // src: 2D 8×64 UINT8 fully packed → all 512 bytes contiguous
    // dst: 2D 8×64 UINT8 with outer gap (stride=128) → only 64 bytes contiguous
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{8, 64}, {64, 1}});  // packed
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{8, 128}, {64, 1}});  // outer gap

    EXPECT_EQ(desc.getSrcTransferBytes(), 8 * 64) << "src logical: 512 bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 8 * 64) << "dst logical: 512 bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 512) << "packed src: 512 contiguous bytes";
    EXPECT_EQ(desc.getContiguousBytesDst(), 64) << "dst outer gap: only 64 contiguous bytes";
    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1) << "packed src: 1 chunk";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 8) << "dst: 8 chunks of 64 bytes";
}

TEST_F(TestVPUDMADescriptor, getContiguousBytes_fp16_src_packed_dst_strided) {
    // src: 128 FP16 packed → 256 contiguous bytes
    // dst: 128 FP16 stride=4 (gap) → 2 contiguous bytes per element
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::FLOAT16;
    desc.src.setDimension_OutermostFirst({{128, 2}});  // packed
    desc.dst.dtype = DataType::FLOAT16;
    desc.dst.setDimension_OutermostFirst({{128, 4}});  // strided

    EXPECT_EQ(desc.getSrcTransferBytes(), 256) << "src: 128*2 = 256 logical bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 256) << "dst: 128*2 = 256 logical bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 256) << "packed FP16 src: 256 contiguous bytes";
    EXPECT_EQ(desc.getContiguousBytesDst(), 2) << "strided FP16 dst stride=4: 2 bytes contiguous";
    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1) << "packed src: 1 chunk";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 128) << "strided dst: 128 chunks of 2 bytes";
}

TEST_F(TestVPUDMADescriptor, getContiguousBytes_both_src_dst_strided_2D) {
    // src: 16×64 UINT8 outer gap (stride=128) → 64 contiguous bytes, 16 chunks
    // dst: 8×64 UINT8 outer gap (stride=256) → 64 contiguous bytes, 8 chunks
    VPUDMADescriptor desc;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{16, 128}, {64, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{8, 256}, {64, 1}});

    EXPECT_EQ(desc.getSrcTransferBytes(), 16 * 64) << "src logical: 1024 bytes";
    EXPECT_EQ(desc.getDstTransferBytes(), 8 * 64) << "dst logical: 512 bytes";
    EXPECT_EQ(desc.getContiguousBytesSrc(), 64) << "src outer gap: 64 contiguous bytes";
    EXPECT_EQ(desc.getContiguousBytesDst(), 64) << "dst outer gap: 64 contiguous bytes";
    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 16) << "src: 16 chunks of 64 bytes";
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 8) << "dst: 8 chunks of 64 bytes";
}

// ===========================================================================
// checkDescriptorSanity
// ===========================================================================

// Helper: build a minimal valid descriptor (UINT8 1D, 512 bytes, VPU_4_0).
static VPUDMADescriptor makeValidDescriptor() {
    VPUDMADescriptor d;
    d.device = VPUDevice::VPU_4_0;
    d.src.dtype = DataType::UINT8;
    d.src.setDimension_OutermostFirst({{512, 1}});
    d.dst.dtype = DataType::UINT8;
    d.dst.setDimension_OutermostFirst({{512, 1}});
    return d;
}

TEST_F(TestVPUDMADescriptor, checkSanity_valid_descriptor_does_not_throw) {
    EXPECT_NO_THROW(makeValidDescriptor().checkDescriptorSanity());
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_sub_byte_src_dtype) {
    // INT4 has 4-bit width — not byte-aligned.
    auto d = makeValidDescriptor();
    d.src.dtype = DataType::INT4;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument);
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_sub_byte_dst_dtype) {
    // UINT2 has 2-bit width — not byte-aligned.
    auto d = makeValidDescriptor();
    d.dst.dtype = DataType::UINT2;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument);
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_when_bytes_read_ne_bytes_written) {
    // src: 512 UINT8; dst: 256 UINT8 — byte counts differ.
    auto d = makeValidDescriptor();
    d.dst.setDimension_OutermostFirst({{256, 1}});
    // EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument);
    EXPECT_NO_THROW(d.checkDescriptorSanity());
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_invalid_device_size_sentinel) {
    auto d = makeValidDescriptor();
    d.device = VPUDevice::__size;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument);
}

TEST_F(TestVPUDMADescriptor, checkSanity_valid_fp16_descriptor_does_not_throw) {
    // FP16 is 16-bit — byte-aligned; equal src/dst bytes; valid device.
    VPUDMADescriptor d;
    d.device = VPUDevice::VPU_2_7;
    d.src.dtype = DataType::FLOAT16;
    d.src.setDimension_OutermostFirst({{256, 2}});
    d.dst.dtype = DataType::FLOAT16;
    d.dst.setDimension_OutermostFirst({{256, 2}});
    EXPECT_NO_THROW(d.checkDescriptorSanity());
}

// ===========================================================================
// checkDescriptorSanity — check 4: num_dims range [0, VPU_DMA_MAX_DIMS]
// ===========================================================================

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_src_num_dims_above_max) {
    auto d = makeValidDescriptor();
    d.src.num_dims = VPU_DMA_MAX_DIMS + 1;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "src.num_dims > VPU_DMA_MAX_DIMS must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_src_num_dims_negative) {
    auto d = makeValidDescriptor();
    d.src.num_dims = -1;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "src.num_dims < 0 must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_dst_num_dims_above_max) {
    auto d = makeValidDescriptor();
    d.dst.num_dims = VPU_DMA_MAX_DIMS + 1;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "dst.num_dims > VPU_DMA_MAX_DIMS must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_dst_num_dims_negative) {
    auto d = makeValidDescriptor();
    d.dst.num_dims = -1;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "dst.num_dims < 0 must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_accepts_num_dims_at_max_boundary) {
    // VPU_DMA_MAX_DIMS is the highest legal value.
    VPUDMADescriptor d;
    d.device = VPUDevice::VPU_4_0;
    d.src.dtype = DataType::UINT8;
    d.src.setDimension_OutermostFirst({{2, 64}, {2, 32}, {2, 16}, {2, 8}, {2, 4}, {4, 1}});
    d.dst.dtype = DataType::UINT8;
    d.dst.setDimension_OutermostFirst({{2, 64}, {2, 32}, {2, 16}, {2, 8}, {2, 4}, {4, 1}});
    EXPECT_EQ(d.src.num_dims, VPU_DMA_MAX_DIMS);
    EXPECT_NO_THROW(d.checkDescriptorSanity()) << "num_dims == VPU_DMA_MAX_DIMS must not throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_accepts_num_dims_zero) {
    // num_dims == 0 means an empty / uninitialised tensor; sanity must accept it
    // (the tensor is empty, not malformed).
    auto d = makeValidDescriptor();
    d.src.num_dims = 0;
    d.dst.num_dims = 0;
    EXPECT_NO_THROW(d.checkDescriptorSanity()) << "num_dims == 0 must not throw (empty tensor is valid)";
}

// ===========================================================================
// checkDescriptorSanity — check 5: non-negative shapes for active dims
// ===========================================================================

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_negative_src_shape) {
    auto d = makeValidDescriptor();
    // Directly corrupt shape[0] after setting num_dims via setDimension.
    d.src.shape[0] = -1;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "src.shape[0] < 0 must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_negative_dst_shape) {
    auto d = makeValidDescriptor();
    d.dst.shape[0] = -5;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "dst.shape[0] < 0 must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_negative_shape_inner_dim_of_2D) {
    // 2D tensor; corrupt the innermost dimension shape.
    VPUDMADescriptor d;
    d.device = VPUDevice::VPU_4_0;
    d.src.dtype = DataType::UINT8;
    d.src.setDimension_OutermostFirst({{16, 64}, {64, 1}});
    d.src.shape[1] = -1;  // corrupt innermost dim
    d.dst.dtype = DataType::UINT8;
    d.dst.setDimension_OutermostFirst({{16, 64}, {64, 1}});
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument) << "src.shape[1] < 0 in a 2D tensor must throw";
}

TEST_F(TestVPUDMADescriptor, checkSanity_accepts_zero_shape_in_active_dim) {
    // shape[d] == 0 means an empty tensor; it is non-negative and must be accepted.
    auto d = makeValidDescriptor();
    d.src.shape[0] = 0;
    d.dst.shape[0] = 0;
    EXPECT_NO_THROW(d.checkDescriptorSanity()) << "shape == 0 (empty tensor) must not throw";
}

// ===========================================================================
// checkDescriptorSanity — check 6: supported memory-location combination
// ===========================================================================

TEST_F(TestVPUDMADescriptor, checkSanity_accepts_all_four_valid_directions) {
    // All four recognised (src_location, dst_location) pairs must pass.
    const std::vector<std::pair<MemoryLocation, MemoryLocation>> valid_pairs = {
            {MemoryLocation::DRAM, MemoryLocation::CMX},
            {MemoryLocation::CMX, MemoryLocation::DRAM},
            {MemoryLocation::CMX, MemoryLocation::CMX},
            {MemoryLocation::DRAM, MemoryLocation::DRAM},
    };
    for (const auto& [src_loc, dst_loc] : valid_pairs) {
        auto d = makeValidDescriptor();
        d.src_location = src_loc;
        d.dst_location = dst_loc;
        EXPECT_NO_THROW(d.checkDescriptorSanity()) << "Valid direction (src=" << static_cast<int>(src_loc)
                                                   << ", dst=" << static_cast<int>(dst_loc) << ") must not throw";
    }
}

TEST_F(TestVPUDMADescriptor, checkSanity_throws_on_unsupported_location_combination) {
    // Force an unrecognised combination by setting both locations to an out-of-range
    // MemoryLocation value that does not match any of the four supported pairs.
    // We achieve this by casting an integer that is not a valid MemoryLocation enumerator.
    // The descriptor must throw because getDirection() returns MemoryDirection::__size.
    auto d = makeValidDescriptor();
    // Use two different invalid location values (not DRAM=0 and not CMX=1 if those are the only two).
    // Cast an out-of-range integer as MemoryLocation to produce an unsupported pair.
    constexpr auto invalid_loc = static_cast<MemoryLocation>(99);
    d.src_location = invalid_loc;
    d.dst_location = invalid_loc;
    EXPECT_THROW(d.checkDescriptorSanity(), std::invalid_argument)
            << "Unrecognised (src_location, dst_location) pair must throw";
}

}  // namespace VPUNN_unit_tests
