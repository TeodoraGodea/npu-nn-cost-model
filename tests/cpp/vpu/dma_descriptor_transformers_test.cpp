// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dma_descriptor_transformers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "vpu/dma_descriptors.h"
#include "vpu/dma_types.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a symmetric VPUDMADescriptor where src and dst are identical.
/// Dimensions are specified outermost-first (OF): each pair is {element_count, byte_stride}.
/// @param dims_outermost_first  List of {element_count, byte_stride} pairs, outermost dimension first.
static VPUDMADescriptor makeDesc_OF(VPUDevice device, MemoryLocation src_loc, MemoryLocation dst_loc, DataType dtype,
                                    std::initializer_list<std::pair<int32_t, int32_t>> dims_outermost_first) {
    VPUDMADescriptor desc;
    desc.device = device;
    desc.src_location = src_loc;
    desc.dst_location = dst_loc;

    desc.src.dtype = dtype;
    desc.src.setDimension_OutermostFirst(dims_outermost_first);

    desc.dst.dtype = dtype;
    desc.dst.setDimension_OutermostFirst(dims_outermost_first);

    return desc;
}

/// Build a VPUDMADescriptor with independent src/dst tensors.
/// Dimensions are specified outermost-first (OF): each pair is {element_count, byte_stride}.
/// @param src_dims_outermost_first  Source {element_count, byte_stride} pairs, outermost first.
/// @param dst_dims_outermost_first  Destination {element_count, byte_stride} pairs, outermost first.
static VPUDMADescriptor makeDescAsymmetric_OF(
        VPUDevice device, MemoryLocation src_loc, MemoryLocation dst_loc, DataType src_dtype,
        std::initializer_list<std::pair<int32_t, int32_t>> src_dims_outermost_first, DataType dst_dtype,
        std::initializer_list<std::pair<int32_t, int32_t>> dst_dims_outermost_first) {
    VPUDMADescriptor desc;
    desc.device = device;
    desc.src_location = src_loc;
    desc.dst_location = dst_loc;

    desc.src.dtype = src_dtype;
    desc.src.setDimension_OutermostFirst(src_dims_outermost_first);

    desc.dst.dtype = dst_dtype;
    desc.dst.setDimension_OutermostFirst(dst_dims_outermost_first);

    return desc;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TestDMADescriptorTransformer : public ::testing::Test {
protected:
    /// Convenience: call the transformer.
    DMANNWorkload_NPU40_50 transform(const VPUDMADescriptor& desc) {
        return DMADescriptorTransformer::fromStridedTensors_toDMANNWorkload_NPU40_50(desc);
    }

    /// Verify that a DMANNWorkload_NPU40_50 and its originating VPUDMADescriptor
    /// agree on all transfer-size and contiguity metrics.
    void expectMetricsMatch(const DMANNWorkload_NPU40_50& wl, const VPUDMADescriptor& desc) {
        EXPECT_EQ(wl.getAccessedBytes(), desc.getSrcTransferBytes()) << "src: wl accessed == desc src transfer";
        EXPECT_EQ(wl.getWrittenBytes(), desc.getDstTransferBytes()) << "dst: wl written == desc dst transfer";
        EXPECT_EQ(wl.getContiguousBytesSrc(), desc.getContiguousBytesSrc()) << "src contiguous bytes match";
        EXPECT_EQ(wl.getContiguousBytesDst(), desc.getContiguousBytesDst()) << "dst contiguous bytes match";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), desc.getNumContiguousChunksSrc()) << "src chunks match";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), desc.getNumContiguousChunksDst()) << "dst chunks match";
    }

    /// Convenience: call the N2D transformer (NPU40_50 → VPUDMADescriptor).
    VPUDMADescriptor transformN2D(const DMANNWorkload_NPU40_50& wl) {
        return DMADescriptorTransformer::fromDMANNWorkload_NPU40_50_to_VPUDMADescriptor(wl);
    }

    /// Verify that a VPUDMADescriptor produced by transformN2D() and its originating
    /// DMANNWorkload_NPU40_50 agree on all transfer-size and contiguity metrics.
    void expectDescMetricsMatchWl(const VPUDMADescriptor& desc, const DMANNWorkload_NPU40_50& wl) {
        EXPECT_EQ(desc.getSrcTransferBytes(), wl.getAccessedBytes()) << "src: desc transfer == wl accessed";
        EXPECT_EQ(desc.getDstTransferBytes(), wl.getWrittenBytes()) << "dst: desc transfer == wl written";
        EXPECT_EQ(desc.getContiguousBytesSrc(), wl.getContiguousBytesSrc()) << "src contiguous bytes match";
        EXPECT_EQ(desc.getContiguousBytesDst(), wl.getContiguousBytesDst()) << "dst contiguous bytes match";
        EXPECT_EQ(desc.getNumContiguousChunksSrc(), wl.getNumContiguousChunksSrc()) << "src chunks match";
        EXPECT_EQ(desc.getNumContiguousChunksDst(), wl.getNumContiguousChunksDst()) << "dst chunks match";
    }
};

// ===========================================================================
// 1D — fully contiguous (no extra dimensions)
// ===========================================================================

/// 1D UINT8 packed DDR→CMX: entire buffer is one contiguous block.
/// Expected: src_width = dst_width = total bytes, num_dim = 0.
TEST_F(TestDMADescriptorTransformer, D2NN40_OneDim_UINT8_Packed_DDR2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{256, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.device, VPUDevice::VPU_4_0);
    EXPECT_EQ(wl.transfer_direction, MemoryDirection::DDR2CMX);
    EXPECT_EQ(wl.src_width, 256) << "1D packed UINT8: src_width = total bytes";
    EXPECT_EQ(wl.dst_width, 256) << "1D packed UINT8: dst_width = total bytes";
    EXPECT_EQ(wl.num_dim, 0) << "1D: no extra dimensions";
    EXPECT_EQ(wl.getAccessedBytes(), 256) << "Round-trip check: getAccessedBytes()";
    expectMetricsMatch(wl, desc);
}

/// 1D FP16 packed CMX→DDR.
/// Expected: src_width = dst_width = 128*2 = 256, num_dim = 0.
TEST_F(TestDMADescriptorTransformer, D2NN40_OneDim_FP16_Packed_CMX2DDR) {
    auto desc =
            makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::DRAM, DataType::FLOAT16, {{128, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.transfer_direction, MemoryDirection::CMX2DDR);
    EXPECT_EQ(wl.src_width, 256) << "FP16 128 elems: 128*2=256";
    EXPECT_EQ(wl.dst_width, 256);
    EXPECT_EQ(wl.num_dim, 0);
    expectMetricsMatch(wl, desc);
}

/// 1D CMX→CMX single element.
/// Expected: src_width = dst_width = 1, num_dim = 0.
TEST_F(TestDMADescriptorTransformer, D2NN40_OneDim_SingleElement_CMX2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::CMX, DataType::UINT8, {{1, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.transfer_direction, MemoryDirection::CMX2CMX);
    EXPECT_EQ(wl.src_width, 1);
    EXPECT_EQ(wl.dst_width, 1);
    EXPECT_EQ(wl.num_dim, 0);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// 2D — fully packed (inner + outer contiguous → collapses into 1D)
// ===========================================================================

/// 2D 16×64 UINT8, outer_stride=64 (packed) → collapses to a single 1024-byte block.
/// Expected: src_width = 1024, num_dim = 0.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_FullyPacked_CollapsesTo1D) {
    // outermost-first: {16 elements, stride=64}, {64 elements, stride=1}
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 64}, {64, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 1024) << "2D packed collapses: 16*64 = 1024";
    EXPECT_EQ(wl.dst_width, 1024);
    EXPECT_EQ(wl.num_dim, 0) << "fully packed 2D: no extra dimensions";
    EXPECT_EQ(wl.getAccessedBytes(), 1024);
    expectMetricsMatch(wl, desc);
}

/// 2D 8×32 FP16 packed → collapses: 8*32*2 = 512 bytes.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_FP16_FullyPacked_CollapsesTo1D) {
    // outermost-first: {8 elements, stride=64 (outer, packed: 32*2)}, {32 elements, stride=2 (FP16 packed)}
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{8, 64}, {32, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 8 * 32 * 2) << "FP16 8×32 packed: 512 bytes";
    EXPECT_EQ(wl.dst_width, 512);
    EXPECT_EQ(wl.num_dim, 0);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// 2D — gap in outer dimension → 1 extra dimension
// ===========================================================================

/// 2D 16×64 UINT8, outer_stride=128 (64-byte padding after each row).
/// Inner 64 bytes are contiguous; one extra dimension for the 16 rows.
/// Expected: src_width=64, num_dim=1, e_dim[0]={src_stride=128, src_dim_size=15}.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_OuterGap_OneExtraDim) {
    // outermost-first: {16 elements, stride=128 (gap)}, {64 elements, stride=1 (packed)}
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 128}, {64, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 64) << "inner 64 bytes contiguous";
    EXPECT_EQ(wl.dst_width, 64);
    EXPECT_EQ(wl.num_dim, 1) << "one outer strided dimension";

    // e_dim[0] encodes the outer dimension (0-based size = shape-1)
    EXPECT_EQ(wl.e_dim[0].src_stride, 128) << "outer stride = 128";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 15) << "0-based: 16-1 = 15";
    EXPECT_EQ(wl.e_dim[0].dst_stride, 128);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 15);

    // Round-trip: getAccessedBytes() uses (dim_size+1) multiplication
    EXPECT_EQ(wl.getAccessedBytes(), 64 * 16) << "accessed = src_width * (15+1) = 1024";

    expectMetricsMatch(wl, desc);
}

/// 2D 8×32 FP16, outer_stride=100 (gap vs expected 64).
/// Expected: src_width=64, num_dim=1, e_dim[0].src_stride=100, src_dim_size=7.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_FP16_OuterGap_OneExtraDim) {
    // outermost-first: {8 elements, stride=100 (gap)}, {32 elements, stride=2 (FP16 packed)}
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{8, 100}, {32, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 64) << "FP16 32 elems: 32*2=64 contiguous";
    EXPECT_EQ(wl.num_dim, 1);
    EXPECT_EQ(wl.e_dim[0].src_stride, 100);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 7) << "8-1 = 7";
    EXPECT_EQ(wl.getAccessedBytes(), 64 * 8);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// 3D — various contiguity patterns
// ===========================================================================

/// 3D {4, 8, 32} UINT8 fully packed → collapses entirely, num_dim=0.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_FullyPacked_CollapsesTo1D) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 256}, {8, 32}, {32, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 4 * 8 * 32) << "3D fully packed: 1024 bytes";
    EXPECT_EQ(wl.num_dim, 0);
    EXPECT_EQ(wl.getAccessedBytes(), 1024);
    expectMetricsMatch(wl, desc);
}

/// 3D {4, 8, 32} UINT8, gap only in outermost (stride=512 vs expected 256).
/// Inner 2 dims collapse: src_width=256; one extra dim for the 4 slices.
/// Expected: src_width=256, num_dim=1, e_dim[0]={src_stride=512, src_dim_size=3}.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_GapInOutermost_OneExtraDim) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 512}, {8, 32}, {32, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 8 * 32) << "inner 2 dims packed: 256 bytes";
    EXPECT_EQ(wl.num_dim, 1);
    EXPECT_EQ(wl.e_dim[0].src_stride, 512);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 3) << "4-1=3";
    EXPECT_EQ(wl.getAccessedBytes(), 256 * 4) << "256 * (3+1) = 1024";
    expectMetricsMatch(wl, desc);
}

/// 3D {4, 8, 32} UINT8, gap in middle dimension (stride=128 vs expected 32).
/// Only innermost 32 bytes contiguous; two extra dims.
/// Expected: src_width=32, num_dim=2,
///   e_dim[0]={src_stride=128, src_dim_size=7}   (middle dim, first to break contiguity)
///   e_dim[1]={src_stride=512, src_dim_size=3}   (outermost, processed after)
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_GapInMiddle_TwoExtraDims) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 512}, {8, 128}, {32, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 32) << "only innermost block: 32 bytes";
    EXPECT_EQ(wl.num_dim, 2) << "two outer dims are strided";
    EXPECT_EQ(wl.e_dim[0].src_stride, 128) << "middle dim stride";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 7) << "8-1=7";
    EXPECT_EQ(wl.e_dim[1].src_stride, 512) << "outermost dim stride";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 3) << "4-1=3";
    EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 4) << "32 * 8 * 4 = 1024";
    expectMetricsMatch(wl, desc);
}

/// 3D {4, 8, 32} UINT8, innermost stride=2 (gap between every element).
/// Seed = dtype_to_bytes(UINT8) = 1; innermost stride=2 != 1 → breaks immediately.
/// src_width = 1; all three dims become e_dim entries.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_InnerStrided_MiddlePacked_TwoExtraDims) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 1024}, {8, 64}, {32, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 1) << "innermost stride=2 != elem_bytes=1: src_width = 1";
    EXPECT_EQ(wl.num_dim, 3) << "all three dims are strided from the element perspective";
    EXPECT_EQ(wl.e_dim[0].src_stride, 2) << "innermost dim stride";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 31) << "32-1=31";
    EXPECT_EQ(wl.e_dim[1].src_stride, 64) << "middle dim stride";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 7) << "8-1=7";
    EXPECT_EQ(wl.e_dim[2].src_stride, 1024) << "outermost dim stride";
    EXPECT_EQ(wl.e_dim[2].src_dim_size, 3) << "4-1=3";
    EXPECT_EQ(wl.getAccessedBytes(), 1 * 32 * 8 * 4) << "1 * (31+1) * (7+1) * (3+1) = 1024";
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// Degenerate dimension (shape == 1): must be skipped in e_dim encoding
// ===========================================================================

/// 3D outermost-first: {1, stride=999}, {8, 128}, {32, 1}
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_OutermostDegenerate_SkippedInExtraDims) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{1, 999}, {8, 128}, {32, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 32);
    EXPECT_EQ(wl.num_dim, 1) << "degenerate outermost dim (shape=1) must be skipped";
    EXPECT_EQ(wl.e_dim[0].src_stride, 128) << "middle dim stride";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 7) << "8-1=7";
    EXPECT_EQ(wl.e_dim[1].src_stride, 0) << "slot not filled because outermost was skipped";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 0);
    EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 1);
    expectMetricsMatch(wl, desc);
}

/// 3D: {8, 32}, {1, 999}, {32, 1} — middle dimension degenerate.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_MiddleDegenerate_ContiguityChainContinues) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{8, 32}, {1, 999}, {32, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 32 * 1 * 8) << "degenerate middle dim: chain continues, all collapse";
    EXPECT_EQ(wl.num_dim, 0) << "degenerate middle dim: no extra dims";
    EXPECT_EQ(wl.getAccessedBytes(), 256);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// Asymmetric src/dst layouts
// ===========================================================================

/// Src: 2D 16×64 UINT8 with outer gap (stride=128) → src_width=64, 1 extra src dim.
/// Dst: 1D packed 1024 UINT8 → dst_width=1024, 0 extra dst dims.
TEST_F(TestDMADescriptorTransformer, D2NN40_AsymmetricSrcDst_SrcStrided_DstPacked) {
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                                      {{16, 128}, {64, 1}}, DataType::UINT8, {{1024, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 64);
    EXPECT_EQ(wl.dst_width, 1024);
    EXPECT_EQ(wl.num_dim, 1) << "max(src_extra=1, dst_extra=0) = 1";
    EXPECT_EQ(wl.e_dim[0].src_stride, 128);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 15);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 0) << "dst has no extra dim at slot 0: padded with 0";
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 0);
    EXPECT_EQ(wl.e_dim[1].src_stride, 0) << "src has no extra dim at slot 1: padded with 0";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 0);
    EXPECT_EQ(wl.e_dim[1].dst_stride, 0) << "dst has no extra dim at slot 1: padded with 0";
    EXPECT_EQ(wl.e_dim[1].dst_dim_size, 0);
    expectMetricsMatch(wl, desc);
}

/// Src: 1D packed 1024 UINT8.
/// Dst: 2D 16×64 UINT8 with outer gap (stride=128) → dst_width=64, 1 extra dst dim.
TEST_F(TestDMADescriptorTransformer, D2NN40_AsymmetricSrcDst_SrcPacked_DstStrided) {
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                                      {{1024, 1}}, DataType::UINT8, {{16, 128}, {64, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 1024);
    EXPECT_EQ(wl.dst_width, 64);
    EXPECT_EQ(wl.num_dim, 1);
    EXPECT_EQ(wl.e_dim[0].src_stride, 0) << "src has no extra dim at slot 0";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 0);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 128);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 15);
    expectMetricsMatch(wl, desc);
}

/// Both src and dst strided but with different strides/shapes.
/// Src: 2D {16,128} outer, {64,1} inner → src_width=64, 1 extra src dim (stride=128, size=15).
/// Dst: 2D {8,256} outer, {64,1} inner  → dst_width=64, 1 extra dst dim (stride=256, size=7).
TEST_F(TestDMADescriptorTransformer, D2NN40_AsymmetricSrcDst_BothStrided_DifferentShapes) {
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                                      {{16, 128}, {64, 1}}, DataType::UINT8, {{8, 256}, {64, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 64);
    EXPECT_EQ(wl.dst_width, 64);
    EXPECT_EQ(wl.num_dim, 1);
    EXPECT_EQ(wl.e_dim[0].src_stride, 128);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 15);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 256);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 7);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// Memory direction derivation
// ===========================================================================

TEST_F(TestDMADescriptorTransformer, D2NN40_Direction_DDR2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).transfer_direction, MemoryDirection::DDR2CMX);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_Direction_CMX2DDR) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::DRAM, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).transfer_direction, MemoryDirection::CMX2DDR);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_Direction_CMX2CMX) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::CMX, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).transfer_direction, MemoryDirection::CMX2CMX);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_Direction_DDR2DDR) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::DRAM, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).transfer_direction, MemoryDirection::DDR2DDR);
}

// ===========================================================================
// Device passthrough
// ===========================================================================

TEST_F(TestDMADescriptorTransformer, D2NN40_Device_NPU40_Passthrough) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).device, VPUDevice::VPU_4_0);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_Device_NPU50_Passthrough) {
    auto desc = makeDesc_OF(VPUDevice::NPU_5_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).device, VPUDevice::NPU_5_0);
}

// ===========================================================================
// num_engine default
// ===========================================================================

TEST_F(TestDMADescriptorTransformer, D2NN40_NumEngine_DefaultIsOne) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{64, 1}});
    EXPECT_EQ(transform(desc).num_engine, Num_DMA_Engine::Num_Engine_1);
}

// ===========================================================================
// Round-trip: getAccessedBytes() consistency
// ===========================================================================

/// For any packed descriptor, the transformer result's getAccessedBytes() must equal
/// the source descriptor's getTransferBytes().
TEST_F(TestDMADescriptorTransformer, D2NN40_RoundTrip_AccessedBytes_1D_Packed) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8, {{512, 1}});
    auto wl = transform(desc);
    EXPECT_EQ(wl.getAccessedBytes(), desc.getTransferBytes());
    expectMetricsMatch(wl, desc);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_RoundTrip_AccessedBytes_2D_Strided) {
    // 16×64 with outer gap: transfer bytes = 16*64 = 1024
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{16, 128}, {64, 1}});
    auto wl = transform(desc);
    EXPECT_EQ(wl.getAccessedBytes(), desc.getTransferBytes()) << "round-trip: accessed bytes must match";
    expectMetricsMatch(wl, desc);
}

TEST_F(TestDMADescriptorTransformer, D2NN40_RoundTrip_AccessedBytes_3D_GapMiddle) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{4, 512}, {8, 128}, {32, 1}});
    auto wl = transform(desc);
    EXPECT_EQ(wl.getAccessedBytes(), desc.getTransferBytes());
    expectMetricsMatch(wl, desc);
}

/// Check that every transformer result with extra_dims > 0 has a valid e_dim[] entry:
/// * For every outer strided dimension, there is a matching e_dim[] element.
/// * When the innermost dimension itself has a stride gap, it also becomes an e_dim entry.
TEST_F(TestDMADescriptorTransformer, D2NN40_ExtraDims_ValidContents) {
    // 2D {8,256} outer, {32,2} inner dtype=UINT8:
    // Seed = dtype_to_bytes(UINT8) = 1.
    // d=1 (innermost): stride=2 != 1 → breaks. src_width = 1.
    // Both innermost and outer dims become e_dim entries.
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{8, 256}, {32, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 1) << "innermost stride=2 != elem_bytes=1: src_width = 1";
    EXPECT_EQ(wl.num_dim, 2) << "two e_dim entries: innermost strided + outer strided";
    EXPECT_EQ(wl.e_dim[0].src_stride, 2) << "e_dim[0]: innermost stride";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 31) << "e_dim[0]: 32-1=31";
    EXPECT_EQ(wl.e_dim[1].src_stride, 256) << "e_dim[1]: outer stride";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 7) << "e_dim[1]: 8-1=7";
    EXPECT_EQ(wl.getAccessedBytes(), 1 * 32 * 8) << "accessed = 1 * 32 * 8 = 256";
    expectMetricsMatch(wl, desc);
}

/// Sanity checks on a 6D descriptor with gaps in every outer dimension (MaxExtraDimensions = 5).
TEST_F(TestDMADescriptorTransformer, D2NN40_SixDim_AllOuterDimsStrided_MaxExtraDims) {
    // Outermost-first dims: {2,256},{2,128},{2,64},{2,32},{2,8},{4,1}  dtype=UINT8
    // Innermost block: compute_size_in_bytes(4, UINT8) = 4 bytes.
    // dim[4] (array index 4): stride=8 != 4 → gap, so all 5 outer dims become extra dims.
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{2, 256}, {2, 128}, {2, 64}, {2, 32}, {2, 8}, {4, 1}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 4) << "only innermost block: 4 bytes";
    EXPECT_EQ(wl.num_dim, 5) << "5 outer strided dims, capped at MaxExtraDimensions=5";
    EXPECT_EQ(wl.e_dim[0].src_stride, 8);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 1);
    EXPECT_EQ(wl.e_dim[1].src_stride, 32);
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 1);
    EXPECT_EQ(wl.e_dim[2].src_stride, 64);
    EXPECT_EQ(wl.e_dim[2].src_dim_size, 1);
    EXPECT_EQ(wl.e_dim[3].src_stride, 128);
    EXPECT_EQ(wl.e_dim[3].src_dim_size, 1);
    EXPECT_EQ(wl.e_dim[4].src_stride, 256);
    EXPECT_EQ(wl.e_dim[4].src_dim_size, 1);

    // Round-trip: src_width * (1+1)^5 = 4 * 2^5 = 128
    EXPECT_EQ(wl.getAccessedBytes(), 4 * 32) << "accessed = 4 * 2^5 = 128";
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// Edge case: num_dims == 0 (empty descriptor)
// ===========================================================================

/// A descriptor with num_dims=0 on both sides: widths=0, num_dim=0.
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroDims_ProducesZeroWidths) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;
    desc.src.num_dims = 0;
    desc.dst.num_dims = 0;
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "empty src: width=0";
    EXPECT_EQ(wl.dst_width, 0) << "empty dst: width=0";
    EXPECT_EQ(wl.num_dim, 0);
}

// ===========================================================================
// Multi-byte dtype — additional stride pattern coverage
// ===========================================================================

// ---------------------------------------------------------------------------
// FP16: gap in innermost dimension (stride != elem_bytes)
// ---------------------------------------------------------------------------

/// FP16 2D, innermost stride=4 (2-byte gap between each 2-byte element).
/// Seed = dtype_to_bytes(FLOAT16) = 2.
/// d=1 (innermost): stride=4 != 2 → breaks immediately. src_width = 2.
/// Both innermost and outer dims become e_dim entries.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_FP16_InnerStrided_BothDimsInEdim) {
    // outermost-first: {8, 256}, {32, 4}  dtype=FLOAT16
    // innermost has a 2-byte gap after each FP16 element
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{8, 256}, {32, 4}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 2) << "FP16 innermost stride=4 != elem_bytes=2: src_width = 2";
    EXPECT_EQ(wl.dst_width, 2);
    EXPECT_EQ(wl.num_dim, 2) << "innermost strided + outer: two e_dim entries";
    EXPECT_EQ(wl.e_dim[0].src_stride, 4);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 31);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 4);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 31);
    EXPECT_EQ(wl.e_dim[1].src_stride, 256);
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 7);
    EXPECT_EQ(wl.e_dim[1].dst_stride, 256);
    EXPECT_EQ(wl.e_dim[1].dst_dim_size, 7);
    EXPECT_EQ(wl.getAccessedBytes(), 2 * 32 * 8) << "accessed = 2 * 32 * 8 = 512";
    expectMetricsMatch(wl, desc);
}

// ---------------------------------------------------------------------------
// FP16: gap in middle dimension of a 3D tensor
// ---------------------------------------------------------------------------

/// FP16 3D {4, 8, 32}, outermost-first strides {512, 200, 2}.
/// Innermost is packed (stride=2==elem_bytes=2): absorb → 64.
/// Middle has gap (stride=200 != 64): breaks at d=1.
/// src_width=64, num_dim=2, e_dim[0]=middle, e_dim[1]=outermost.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_FP16_GapInMiddle_TwoExtraDims) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{4, 512}, {8, 200}, {32, 2}});
    auto wl = transform(desc);

    // Seed=2; d=2: stride=2==2 → absorb → 64; d=1: stride=200 != 64 → breaks.
    EXPECT_EQ(wl.src_width, 64) << "FP16 32 elems × 2 = 64 contiguous bytes";
    EXPECT_EQ(wl.dst_width, 64);
    EXPECT_EQ(wl.num_dim, 2) << "middle and outermost are strided: two e_dim entries";
    EXPECT_EQ(wl.e_dim[0].src_stride, 200);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 7);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 200);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 7);
    EXPECT_EQ(wl.e_dim[1].src_stride, 512);
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 3);
    EXPECT_EQ(wl.e_dim[1].dst_stride, 512);
    EXPECT_EQ(wl.e_dim[1].dst_dim_size, 3);
    EXPECT_EQ(wl.getAccessedBytes(), 64 * 8 * 4) << "accessed = 64 * 8 * 4 = 2048";
    expectMetricsMatch(wl, desc);
}

// ---------------------------------------------------------------------------
// FP16: gap only in outermost dimension of a 3D tensor
// ---------------------------------------------------------------------------

/// FP16 3D {4, 8, 32}: innermost stride=2 (packed), middle stride=64 (packed),
/// outermost stride=1024 (gap vs expected 4*8*32*2/4 = 512).
/// The inner two dims fully collapse into 512 bytes; one extra dim for the 4 slices.
TEST_F(TestDMADescriptorTransformer, D2NN40_ThreeDim_FP16_GapInOutermost_OneExtraDim) {
    // expected packed outer stride = 8*32*2 = 512; using 1024 introduces a gap
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                            {{4, 1024}, {8, 64}, {32, 2}});
    auto wl = transform(desc);

    // Seed=2; d=2: stride=2==2 → absorb → 64; d=1: stride=64==64 → absorb → 512;
    // d=0: stride=1024 != 512 → breaks.
    EXPECT_EQ(wl.src_width, 8 * 32 * 2) << "FP16 inner two dims packed: 8*32*2 = 512 bytes";
    EXPECT_EQ(wl.dst_width, 512);
    EXPECT_EQ(wl.num_dim, 1) << "only outermost is strided: one e_dim entry";
    EXPECT_EQ(wl.e_dim[0].src_stride, 1024);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 3);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 1024);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 3);
    EXPECT_EQ(wl.getAccessedBytes(), 512 * 4) << "accessed = 512 * 4 = 2048";
    expectMetricsMatch(wl, desc);
}

// ---------------------------------------------------------------------------
// INT32 (4-byte element): outer gap in 2D
// ---------------------------------------------------------------------------

/// INT32 2D {16, 256}, {32, 4}: innermost packed (stride=4==elem_bytes=4) → absorb → 128.
/// Outermost gap (stride=256 != 128): breaks at d=0.
/// src_width=128, num_dim=1, e_dim[0]={src_stride=256, dim_size=15}.
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_INT32_OuterGap_OneExtraDim) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::INT32,
                            {{16, 256}, {32, 4}});
    auto wl = transform(desc);

    // Seed=4 (INT32). d=1: stride=4==4 → absorb → 128; d=0: stride=256 != 128 → breaks.
    EXPECT_EQ(wl.src_width, 32 * 4) << "INT32 32 elems × 4 = 128 contiguous bytes";
    EXPECT_EQ(wl.dst_width, 128);
    EXPECT_EQ(wl.num_dim, 1) << "one outer strided dimension";
    EXPECT_EQ(wl.e_dim[0].src_stride, 256);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 15);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 256);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 15);
    EXPECT_EQ(wl.getAccessedBytes(), 128 * 16) << "accessed = 128 * 16 = 2048";
    expectMetricsMatch(wl, desc);
}

/// INT32 2D: innermost stride=8 (gap between elements).
TEST_F(TestDMADescriptorTransformer, D2NN40_TwoDim_INT32_InnerStrided_BothDimsInEdim) {
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::INT32,
                            {{8, 256}, {16, 8}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 4) << "INT32 innermost stride=8 != elem_bytes=4: src_width = 4";
    EXPECT_EQ(wl.dst_width, 4);
    EXPECT_EQ(wl.num_dim, 2) << "innermost strided + outer: two e_dim entries";
    EXPECT_EQ(wl.e_dim[0].src_stride, 8);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 15);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 8);
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 15);
    EXPECT_EQ(wl.e_dim[1].src_stride, 256);
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 7);
    EXPECT_EQ(wl.e_dim[1].dst_stride, 256);
    EXPECT_EQ(wl.e_dim[1].dst_dim_size, 7);
    EXPECT_EQ(wl.getAccessedBytes(), 4 * 16 * 8) << "accessed = 4 * 16 * 8 = 512";
    expectMetricsMatch(wl, desc);
}

// ---------------------------------------------------------------------------
// FP16 asymmetric: src has strided innermost, dst is 1D packed
// ---------------------------------------------------------------------------

/// Src: FP16 2D with innermost gap (stride=4 ≠ 2): src_width=2, 2 e_dim entries.
/// Dst: 1D packed FP16 flat buffer with 512 elements (1024 bytes total).
/// Verifies that the asymmetric padding in e_dim[] and round-trip metrics are correct.
TEST_F(TestDMADescriptorTransformer, D2NN40_Asymmetric_FP16_SrcInnerStrided_DstPacked) {
    // src: 8 rows × 32 FP16 elements, inner stride=4 (gap), outer stride=256
    // dst: flat packed buffer with 512 FP16 elements (1024 bytes)
    auto desc = makeDescAsymmetric_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::FLOAT16,
                                      {{8, 256}, {32, 4}}, DataType::FLOAT16, {{512, 2}});
    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 2);
    EXPECT_EQ(wl.dst_width, 1024) << "dst 512 FP16 elems × 2 = 1024 bytes";
    EXPECT_EQ(wl.num_dim, 2) << "max(src_extra=2, dst_extra=0) = 2";
    EXPECT_EQ(wl.e_dim[0].src_stride, 4);
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 31);
    EXPECT_EQ(wl.e_dim[0].dst_stride, 0) << "dst has no extra dim at slot 0: padded with 0";
    EXPECT_EQ(wl.e_dim[0].dst_dim_size, 0);
    EXPECT_EQ(wl.e_dim[1].src_stride, 256);
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 7);
    EXPECT_EQ(wl.e_dim[1].dst_stride, 0) << "dst has no extra dim at slot 1: padded with 0";
    EXPECT_EQ(wl.e_dim[1].dst_dim_size, 0);
    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// Overflow: 6D tensor with strided innermost dimension
// ===========================================================================
//
// When the innermost dimension has a stride gap it consumes e_dim[0], leaving
// only four slots for the five remaining outer dimensions.  The fix in
// collapse_side folds each overflow (unrepresentable) outer dimension into the
// last filled slot by multiplying its element count:
//
//   last_slot.src_dim_size = (last_slot.src_dim_size + 1) * shape[overflow] - 1
//
// This preserves getAccessedBytes() exactly at the cost of losing the stride
// of the folded dimension (an acceptable lossy collapse).

/// 6D UINT8 tensor, all dims non-degenerate, innermost itself strided.
/// outermost-first layout:
///   dim[0]: shape=3, stride=999   <- OUTERMOST – overflows, folded into e_dim[4]
///   dim[1]: shape=2, stride=500
///   dim[2]: shape=2, stride=250
///   dim[3]: shape=2, stride=100
///   dim[4]: shape=2, stride=50
///   dim[5]: shape=8, stride=3     <- INNERMOST – stride(3) != elem_bytes(1): gap
///
/// After folding: e_dim[4].src_dim_size = (1+1)*3 - 1 = 5
/// getAccessedBytes() = 1 * 8 * 2 * 2 * 2 * 6 = 384  (correct total)
TEST_F(TestDMADescriptorTransformer,
       D2NN40_SixDim_InnermostStrided_OutermostDimFoldedIntoLastSlot) {
    // 6D UINT8: innermost stride=3 (gap, not 1=elem_bytes).
    auto desc = makeDesc_OF(VPUDevice::VPU_4_0, MemoryLocation::DRAM, MemoryLocation::CMX, DataType::UINT8,
                            {{3, 999}, {2, 500}, {2, 250}, {2, 100}, {2, 50}, {8, 3}});

    // Total transfer: 8 * 2 * 2 * 2 * 2 * 3 = 384 bytes for UINT8.
    EXPECT_EQ(desc.getSrcTransferBytes(), 384) << "descriptor total transfer size";

    auto wl = transform(desc);

    // Innermost stride gap → src_width = elem_bytes = 1.
    EXPECT_EQ(wl.src_width, 1) << "innermost stride gap: src_width = elem_bytes = 1";
    EXPECT_EQ(wl.dst_width, 1);

    // Five slots filled (MaxExtraDimensions = 5); outermost folded, not dropped.
    EXPECT_EQ(wl.num_dim, 5) << "num_dim capped at MaxExtraDimensions=5";

    // e_dim[0..3]: innermost and the next three outer dims encoded normally.
    EXPECT_EQ(wl.e_dim[0].src_stride,  3)  << "e_dim[0]: innermost stride";
    EXPECT_EQ(wl.e_dim[0].src_dim_size, 7) << "e_dim[0]: shape=8 -> 8-1=7";
    EXPECT_EQ(wl.e_dim[1].src_stride,  50) << "e_dim[1]: dim[4] stride";
    EXPECT_EQ(wl.e_dim[1].src_dim_size, 1) << "e_dim[1]: shape=2 -> 2-1=1";
    EXPECT_EQ(wl.e_dim[2].src_stride, 100) << "e_dim[2]: dim[3] stride";
    EXPECT_EQ(wl.e_dim[2].src_dim_size, 1);
    EXPECT_EQ(wl.e_dim[3].src_stride, 250) << "e_dim[3]: dim[2] stride";
    EXPECT_EQ(wl.e_dim[3].src_dim_size, 1);

    // e_dim[4]: dim[1] was encoded normally (stride=500, size=1); then dim[0]
    // (shape=3) overflowed and was folded in: (1+1)*3 - 1 = 5.
    // The stride of dim[0] (999) is intentionally lost.
    EXPECT_EQ(wl.e_dim[4].src_stride, 500) << "e_dim[4]: stride from dim[1] (dim[0] stride lost)";
    EXPECT_EQ(wl.e_dim[4].src_dim_size, 5) << "e_dim[4]: (1+1)*3 - 1 = 5 (dim[0] shape folded in)";

    // getAccessedBytes() must equal the original transfer size.
    // 1 * (7+1) * (1+1) * (1+1) * (1+1) * (5+1) = 1*8*2*2*2*6 = 384
    EXPECT_EQ(wl.getAccessedBytes(), desc.getSrcTransferBytes())
            << "folded encoding preserves total accessed bytes";

    expectMetricsMatch(wl, desc);
}

// ===========================================================================
// W2D_ — fromDMAWorkload_to_VPUDMADescriptor
// ===========================================================================
//
// Abbreviation: W2D = "Workload-to-Descriptor"
//
// The function always produces a VPUDMADescriptor with 1D packed UINT8 tensors,
// derived independently for src (from wl.input) and dst (from wl.output):
//   dtype           = UINT8  (always — normalised regardless of original dtype)
//   num_dims        = 1
//   shape[0]        = total_bytes  (VPUTensor::size() of the respective tensor)
//   byte_strides[0] = 1            (packed, one byte per element)
//
// The memory footprint is always preserved exactly.

namespace {  // anonymous helpers local to this section

/// Calls the function under test.
VPUDMADescriptor w2d(const DMAWorkload& wl) {
    return DMADescriptorTransformer::fromDMAWorkload_to_VPUDMADescriptor(wl);
}

/// Build a symmetric DMAWorkload (same tensor on both input and output sides).
///
/// Elements are placed in the Z dimension (shape index 2) because ZXY is the default
/// layout and Z is its innermost dimension. This ensures sub-byte types (UINT4, UINT1,
/// etc.) can pack multiple elements per byte as packmode_0 requires: the innermost
/// dimension count must be a multiple of types_per_byte for the chosen dtype.
DMAWorkload makeSymWL(VPUDevice device, unsigned int num_elems, DataType dtype, MemoryLocation src_loc,
                      MemoryLocation dst_loc) {
    const VPUTensor t({1u, 1u, num_elems, 1u}, dtype);  // shape: {X=1, Y=1, Z=num_elems, B=1}
    return DMAWorkload{device, t, t, src_loc, dst_loc};
}

/// Build an asymmetric DMAWorkload (different input / output tensors).
/// Elements are placed in the Z dimension for the same packing reason as makeSymWL.
DMAWorkload makeAsymWL(VPUDevice device, unsigned int src_elems, DataType src_dtype, unsigned int dst_elems,
                       DataType dst_dtype, MemoryLocation src_loc, MemoryLocation dst_loc) {
    const VPUTensor src_t({1u, 1u, src_elems, 1u}, src_dtype);  // shape: {X=1, Y=1, Z=src_elems, B=1}
    const VPUTensor dst_t({1u, 1u, dst_elems, 1u}, dst_dtype);  // shape: {X=1, Y=1, Z=dst_elems, B=1}
    return DMAWorkload{device, src_t, dst_t, src_loc, dst_loc};
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// W2D_Device — device field is copied verbatim
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer, W2D_Device_NPU40) {
    EXPECT_EQ(
            w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT8, MemoryLocation::DRAM, MemoryLocation::CMX)).device,
            VPUDevice::VPU_4_0);
}

TEST_F(TestDMADescriptorTransformer, W2D_Device_NPU50) {
    EXPECT_EQ(
            w2d(makeSymWL(VPUDevice::NPU_5_0, 64u, DataType::UINT8, MemoryLocation::DRAM, MemoryLocation::CMX)).device,
            VPUDevice::NPU_5_0);
}

// ---------------------------------------------------------------------------
// W2D_Locations — memory locations are copied: input→src, output→dst
// ---------------------------------------------------------------------------

TEST_F(TestDMADescriptorTransformer, W2D_Locations_DDR2CMX) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT8, MemoryLocation::DRAM, MemoryLocation::CMX));
    EXPECT_EQ(d.src_location, MemoryLocation::DRAM) << "input_location → src_location";
    EXPECT_EQ(d.dst_location, MemoryLocation::CMX) << "output_location → dst_location";
}

TEST_F(TestDMADescriptorTransformer, W2D_Locations_CMX2DDR) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT8, MemoryLocation::CMX, MemoryLocation::DRAM));
    EXPECT_EQ(d.src_location, MemoryLocation::CMX);
    EXPECT_EQ(d.dst_location, MemoryLocation::DRAM);
}

TEST_F(TestDMADescriptorTransformer, W2D_Locations_CMX2CMX) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT8, MemoryLocation::CMX, MemoryLocation::CMX));
    EXPECT_EQ(d.src_location, MemoryLocation::CMX);
    EXPECT_EQ(d.dst_location, MemoryLocation::CMX);
}

// ---------------------------------------------------------------------------
// W2D_UINT8 — elem_bytes=1 → shape==total_bytes, stride==1
// ---------------------------------------------------------------------------

/// 512 UINT8 elements → num_dims=1, shape[0]=512, stride=1, total=512 bytes.
TEST_F(TestDMADescriptorTransformer, W2D_UINT8_512elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 512u, DataType::UINT8, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.num_dims, 1) << "always 1D";
    EXPECT_EQ(d.src.shape[0], 512) << "512 UINT8 elements";
    EXPECT_EQ(d.src.byte_strides[0], 1) << "elem_bytes = 1";
    EXPECT_EQ(d.dst.dtype, DataType::UINT8);
    EXPECT_EQ(d.dst.num_dims, 1);
    EXPECT_EQ(d.dst.shape[0], 512);
    EXPECT_EQ(d.dst.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 512);
    EXPECT_EQ(d.getDstTransferBytes(), 512);
    EXPECT_EQ(d.getContiguousBytesSrc(), 512) << "1D packed: all bytes contiguous";
    EXPECT_EQ(d.getContiguousBytesDst(), 512);
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
    EXPECT_EQ(d.getNumContiguousChunksDst(), 1);
}

/// Single UINT8 element — degenerate but valid.
TEST_F(TestDMADescriptorTransformer, W2D_UINT8_SingleElement) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 1u, DataType::UINT8, MemoryLocation::CMX, MemoryLocation::CMX));
    EXPECT_EQ(d.src.num_dims, 1);
    EXPECT_EQ(d.src.shape[0], 1);
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 1);
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// W2D_FP16 — normalised to UINT8: 256 FP16 × 2 = 512 bytes → shape=512, stride=1
// ---------------------------------------------------------------------------

/// 256 FP16 elements → total=512 bytes → dtype preserved as FLOAT16: shape=256, stride=2.
TEST_F(TestDMADescriptorTransformer, W2D_FP16_256elems) {
    const auto d =
            w2d(makeSymWL(VPUDevice::VPU_4_0, 256u, DataType::FLOAT16, MemoryLocation::DRAM, MemoryLocation::CMX));

    // fromDMAWorkload_to_VPUDMADescriptor preserves byte-aligned dtypes:
    // 256 FP16 elements × 2 bytes = 512 bytes → FLOAT16 shape=256, stride=2.
    EXPECT_EQ(d.src.dtype, DataType::FLOAT16);
    EXPECT_EQ(d.src.num_dims, 1);
    EXPECT_EQ(d.src.shape[0], 256) << "256 FP16 elements";
    EXPECT_EQ(d.src.byte_strides[0], 2) << "FLOAT16 packed: stride = 2";
    EXPECT_EQ(d.dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(d.dst.shape[0], 256);
    EXPECT_EQ(d.dst.byte_strides[0], 2);
    EXPECT_EQ(d.getSrcTransferBytes(), 512) << "256 FP16 × 2 = 512 bytes preserved";
    EXPECT_EQ(d.getContiguousBytesSrc(), 512) << "1D packed: entire buffer contiguous";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// W2D_INT32 — normalised to UINT8: 64 INT32 × 4 = 256 bytes → shape=256, stride=1
// ---------------------------------------------------------------------------

/// 64 INT32 elements → total=256 bytes → dtype preserved as INT32: shape=64, stride=4.
TEST_F(TestDMADescriptorTransformer, W2D_INT32_64elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::INT32, MemoryLocation::DRAM, MemoryLocation::CMX));

    // fromDMAWorkload_to_VPUDMADescriptor preserves byte-aligned dtypes:
    // 64 INT32 elements × 4 bytes = 256 bytes → INT32 shape=64, stride=4.
    EXPECT_EQ(d.src.dtype, DataType::INT32);
    EXPECT_EQ(d.src.num_dims, 1);
    EXPECT_EQ(d.src.shape[0], 64) << "64 INT32 elements";
    EXPECT_EQ(d.src.byte_strides[0], 4) << "INT32 packed: stride = 4";
    EXPECT_EQ(d.getSrcTransferBytes(), 256) << "64 × 4 = 256 bytes preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// W2D_Packed — 1D descriptor is always fully packed (invariant)
// ---------------------------------------------------------------------------

/// Output of fromDMAWorkload_to_VPUDMADescriptor is always 1D+packed:
/// contiguous bytes == total bytes, chunks == 1.
TEST_F(TestDMADescriptorTransformer, W2D_Always1DPacked) {
    const auto d =
            w2d(makeSymWL(VPUDevice::VPU_4_0, 128u, DataType::FLOAT16, MemoryLocation::DRAM, MemoryLocation::CMX));
    // 128 FP16 = 256 bytes
    EXPECT_EQ(d.getSrcTransferBytes(), 256);
    EXPECT_EQ(d.getDstTransferBytes(), 256);
    EXPECT_EQ(d.getContiguousBytesSrc(), d.getSrcTransferBytes()) << "1D packed: contiguous src == total src bytes";
    EXPECT_EQ(d.getContiguousBytesDst(), d.getDstTransferBytes()) << "1D packed: contiguous dst == total dst bytes";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
    EXPECT_EQ(d.getNumContiguousChunksDst(), 1);
}

// ---------------------------------------------------------------------------
// W2D_Asymmetric — src and dst derived independently from their own dtypes
// ---------------------------------------------------------------------------

/// Src: FP16 128 elems (256 bytes).  Dst: UINT8 256 elems (256 bytes).
/// Src dtype preserved as FLOAT16, dst stays UINT8.
TEST_F(TestDMADescriptorTransformer, W2D_Asymmetric_FP16src_UINT8dst) {
    const auto d = w2d(makeAsymWL(VPUDevice::VPU_4_0, 128u, DataType::FLOAT16, 256u, DataType::UINT8,
                                  MemoryLocation::DRAM, MemoryLocation::CMX));

    // src: 128 FP16 × 2 = 256 bytes → FLOAT16 shape=128, stride=2
    EXPECT_EQ(d.src.dtype, DataType::FLOAT16);
    EXPECT_EQ(d.src.shape[0], 128) << "src: 128 FP16 elements";
    EXPECT_EQ(d.src.byte_strides[0], 2) << "src: FLOAT16 stride = 2";
    EXPECT_EQ(d.getSrcTransferBytes(), 256);
    // dst: 256 UINT8 × 1 = 256 bytes → UINT8 shape=256, stride=1 (unchanged)
    EXPECT_EQ(d.dst.dtype, DataType::UINT8);
    EXPECT_EQ(d.dst.shape[0], 256) << "dst: 256 UINT8 bytes";
    EXPECT_EQ(d.dst.byte_strides[0], 1) << "dst: UINT8 stride = 1";
    EXPECT_EQ(d.getDstTransferBytes(), 256);
}

/// Src: UINT8 1024 elems (1024 bytes).  Dst: FP16 512 elems (1024 bytes).
/// Src stays UINT8, dst dtype preserved as FLOAT16.
TEST_F(TestDMADescriptorTransformer, W2D_Asymmetric_UINT8src_FP16dst) {
    const auto d = w2d(makeAsymWL(VPUDevice::VPU_4_0, 1024u, DataType::UINT8, 512u, DataType::FLOAT16,
                                  MemoryLocation::DRAM, MemoryLocation::CMX));

    // src: 1024 UINT8 × 1 = 1024 bytes → UINT8 shape=1024, stride=1 (unchanged)
    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.shape[0], 1024) << "src: 1024 UINT8 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 1024);
    // dst: 512 FP16 × 2 = 1024 bytes → FLOAT16 shape=512, stride=2
    EXPECT_EQ(d.dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(d.dst.shape[0], 512) << "dst: 512 FP16 elements";
    EXPECT_EQ(d.dst.byte_strides[0], 2) << "dst: FLOAT16 stride = 2";
    EXPECT_EQ(d.getDstTransferBytes(), 1024);
}

/// Src: INT32 64 elems (256 bytes).  Dst: FP16 128 elems (256 bytes).
/// Src dtype preserved as INT32, dst dtype preserved as FLOAT16.
TEST_F(TestDMADescriptorTransformer, W2D_Asymmetric_INT32src_FP16dst) {
    const auto d = w2d(makeAsymWL(VPUDevice::VPU_4_0, 64u, DataType::INT32, 128u, DataType::FLOAT16,
                                  MemoryLocation::DRAM, MemoryLocation::CMX));

    // src: 64 INT32 × 4 = 256 bytes → INT32 shape=64, stride=4
    EXPECT_EQ(d.src.dtype, DataType::INT32);
    EXPECT_EQ(d.src.shape[0], 64) << "src: 64 INT32 elements";
    EXPECT_EQ(d.src.byte_strides[0], 4) << "src: INT32 stride = 4";
    EXPECT_EQ(d.getSrcTransferBytes(), 256);
    // dst: 128 FP16 × 2 = 256 bytes → FLOAT16 shape=128, stride=2
    EXPECT_EQ(d.dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(d.dst.shape[0], 128) << "dst: 128 FP16 elements";
    EXPECT_EQ(d.dst.byte_strides[0], 2) << "dst: FLOAT16 stride = 2";
    EXPECT_EQ(d.getDstTransferBytes(), 256);
}

// ---------------------------------------------------------------------------
// W2D_RoundTrip — descriptor-level consistency after conversion
//
// A DMAWorkload-derived descriptor is always 1D packed: the src/dst tensors
// have num_dims==1, stride==elem_bytes, and all bytes are contiguous.
// These tests verify that the VPUDMADescriptor fields are self-consistent
// (no calls to fromStridedTensors_toDMANNWorkload_NPU40_50 here).
// ---------------------------------------------------------------------------

/// 256 UINT8 DDR→CMX: verify src and dst descriptor tensors are self-consistent.
TEST_F(TestDMADescriptorTransformer, W2D_RoundTrip_UINT8_DDR2CMX) {
    const auto desc =
            w2d(makeSymWL(VPUDevice::VPU_4_0, 256u, DataType::UINT8, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(desc.device, VPUDevice::VPU_4_0);
    EXPECT_EQ(desc.src_location, MemoryLocation::DRAM);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.src.num_dims, 1);
    EXPECT_EQ(desc.src.shape[0], 256) << "256 UINT8 elements";
    EXPECT_EQ(desc.src.byte_strides[0], 1) << "packed: stride == elem_bytes";
    EXPECT_EQ(desc.dst.num_dims, 1);
    EXPECT_EQ(desc.dst.shape[0], 256);
    EXPECT_EQ(desc.dst.byte_strides[0], 1);
    EXPECT_EQ(desc.getSrcTransferBytes(), 256);
    EXPECT_EQ(desc.getDstTransferBytes(), 256);
    EXPECT_EQ(desc.getContiguousBytesSrc(), 256) << "fully packed: contiguous == total";
    EXPECT_EQ(desc.getContiguousBytesDst(), 256);
    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1);
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 1);
}

/// 128 FP16 CMX→CMX: dtype preserved as FLOAT16. shape=128, stride=2.
TEST_F(TestDMADescriptorTransformer, W2D_RoundTrip_FP16_CMX2CMX) {
    const auto desc =
            w2d(makeSymWL(VPUDevice::VPU_4_0, 128u, DataType::FLOAT16, MemoryLocation::CMX, MemoryLocation::CMX));

    EXPECT_EQ(desc.device, VPUDevice::VPU_4_0);
    EXPECT_EQ(desc.src_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);
    // fromDMAWorkload_to_VPUDMADescriptor preserves byte-aligned dtypes:
    // 128 FP16 × 2 bytes = 256 bytes → FLOAT16 shape=128, stride=2.
    EXPECT_EQ(desc.src.dtype, DataType::FLOAT16);
    EXPECT_EQ(desc.src.num_dims, 1);
    EXPECT_EQ(desc.src.shape[0], 128) << "128 FP16 elements";
    EXPECT_EQ(desc.src.byte_strides[0], 2) << "FLOAT16 packed: stride = 2";
    EXPECT_EQ(desc.dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(desc.dst.num_dims, 1);
    EXPECT_EQ(desc.dst.shape[0], 128);
    EXPECT_EQ(desc.dst.byte_strides[0], 2);
    EXPECT_EQ(desc.getSrcTransferBytes(), 256) << "128 FP16 × 2 = 256 bytes preserved";
    EXPECT_EQ(desc.getDstTransferBytes(), 256);
    EXPECT_EQ(desc.getContiguousBytesSrc(), 256);
    EXPECT_EQ(desc.getContiguousBytesDst(), 256);
    EXPECT_EQ(desc.getNumContiguousChunksSrc(), 1);
    EXPECT_EQ(desc.getNumContiguousChunksDst(), 1);
}

// ===========================================================================
// W2D_SubByte — sub-byte dtypes normalised to UINT8
//
// VPUDMADescriptor does not accept sub-byte types.
// fromDMAWorkload_to_VPUDMADescriptor converts any sub-byte dtype to UINT8
// by reinterpreting the packed bytes:
//   UINT4 / INT4  : 2 elements per byte
//   UINT2 / INT2  : 4 elements per byte
//   UINT1 / INT1  : 8 elements per byte
//
// The element counts in the DMAWorkload must be byte-aligned (a multiple of
// types_per_byte) because VPUTensor::size() uses packmode_0, which requires
// the innermost dimension to fully pack every byte it occupies.
//
// In all cases: UINT8 shape[0] == VPUTensor::size() == the original byte count.
// ===========================================================================

// ---------------------------------------------------------------------------
// UINT4 / INT4 (2 elements per byte)
// ---------------------------------------------------------------------------

/// 256 UINT4 elements → 256/2 = 128 bytes → UINT8 shape=128, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_UINT4_256elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 256u, DataType::UINT4, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.num_dims, 1);
    EXPECT_EQ(d.src.shape[0], 128) << "256 UINT4 packed into 128 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1) << "UINT8 stride = 1";
    EXPECT_EQ(d.dst.dtype, DataType::UINT8);
    EXPECT_EQ(d.dst.shape[0], 128);
    EXPECT_EQ(d.dst.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 128) << "memory footprint preserved";
    EXPECT_EQ(d.getDstTransferBytes(), 128);
    EXPECT_EQ(d.getContiguousBytesSrc(), 128);
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

/// 128 INT4 elements → 128/2 = 64 bytes → promoted to INT8 shape=64, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_INT4_128elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 128u, DataType::INT4, MemoryLocation::CMX, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::INT8);
    EXPECT_EQ(d.src.shape[0], 64) << "128 INT4 packed into 64 bytes = 64 INT8 elements";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 64) << "memory footprint preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// UINT2 / INT2 (4 elements per byte)
// ---------------------------------------------------------------------------

/// 64 UINT2 elements → 64/4 = 16 bytes → UINT8 shape=16, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_UINT2_64elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT2, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.shape[0], 16) << "64 UINT2 packed into 16 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 16) << "memory footprint preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

/// 64 INT2 elements → 64/4 = 16 bytes → promoted to INT8 shape=16, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_INT2_64elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::INT2, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::INT8);
    EXPECT_EQ(d.src.shape[0], 16) << "64 INT2 packed into 16 bytes = 16 INT8 elements";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 16) << "memory footprint preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// UINT1 / INT1 (8 elements per byte)
// ---------------------------------------------------------------------------

/// 64 UINT1 elements → 64/8 = 8 bytes → UINT8 shape=8, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_UINT1_64elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT1, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.shape[0], 8) << "64 UINT1 packed into 8 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 8) << "memory footprint preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

/// 64 INT1 elements → 64/8 = 8 bytes → promoted to INT8 shape=8, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_INT1_64elems) {
    const auto d = w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::INT1, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::INT8);
    EXPECT_EQ(d.src.shape[0], 8) << "64 INT1 packed into 8 bytes = 8 INT8 elements";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 8) << "memory footprint preserved";
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// FLOAT4 (4 elements per byte — float sub-byte, promoted to BF8)
// ---------------------------------------------------------------------------

/// 64 FLOAT4 elements → 64/2 = 32 bytes → promoted to BF8 shape=32, stride=1.
/// FLOAT4 is the float sub-byte type; its 8-bit sibling is BF8.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_FLOAT4_64elems) {
    const auto d =
            w2d(makeSymWL(VPUDevice::VPU_4_0, 64u, DataType::FLOAT4, MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::BF8);
    EXPECT_EQ(d.src.num_dims, 1);
    EXPECT_EQ(d.src.shape[0], 32) << "64 FLOAT4 packed into 32 bytes = 32 BF8 elements";
    EXPECT_EQ(d.src.byte_strides[0], 1) << "BF8 stride = 1";
    EXPECT_EQ(d.dst.dtype, DataType::BF8);
    EXPECT_EQ(d.dst.shape[0], 32);
    EXPECT_EQ(d.dst.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 32) << "memory footprint preserved";
    EXPECT_EQ(d.getDstTransferBytes(), 32);
    EXPECT_EQ(d.getNumContiguousChunksSrc(), 1);
}

// ---------------------------------------------------------------------------
// W2D_SubByte_Asymmetric — sub-byte src, byte-aligned dst: footprints must match
// ---------------------------------------------------------------------------

/// Src: 256 UINT4 (128 bytes).  Dst: 128 UINT8 (128 bytes).
/// Both normalised to UINT8 shape=128, stride=1 — same footprint.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_Asymmetric_UINT4src_UINT8dst) {
    const auto d = w2d(makeAsymWL(VPUDevice::VPU_4_0, 256u, DataType::UINT4, 128u, DataType::UINT8,
                                  MemoryLocation::DRAM, MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.shape[0], 128) << "src: 256 UINT4 → 128 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 128);
    EXPECT_EQ(d.dst.dtype, DataType::UINT8);
    EXPECT_EQ(d.dst.shape[0], 128) << "dst: 128 UINT8 → 128 bytes";
    EXPECT_EQ(d.dst.byte_strides[0], 1);
    EXPECT_EQ(d.getDstTransferBytes(), 128);
}

/// Src: 64 UINT1 (8 bytes).  Dst: 8 UINT8 (8 bytes).
/// Both normalised to UINT8 shape=8, stride=1.
TEST_F(TestDMADescriptorTransformer, W2D_SubByte_Asymmetric_UINT1src_UINT8dst) {
    const auto d = w2d(makeAsymWL(VPUDevice::VPU_4_0, 64u, DataType::UINT1, 8u, DataType::UINT8, MemoryLocation::DRAM,
                                  MemoryLocation::CMX));

    EXPECT_EQ(d.src.dtype, DataType::UINT8);
    EXPECT_EQ(d.src.shape[0], 8) << "src: 64 UINT1 → 8 bytes";
    EXPECT_EQ(d.src.byte_strides[0], 1);
    EXPECT_EQ(d.getSrcTransferBytes(), 8);
    EXPECT_EQ(d.dst.dtype, DataType::UINT8);
    EXPECT_EQ(d.dst.shape[0], 8) << "dst: 8 UINT8 → 8 bytes";
    EXPECT_EQ(d.dst.byte_strides[0], 1);
    EXPECT_EQ(d.getDstTransferBytes(), 8);
}

// ===========================================================================
// Empty-tensor guard: shape[d] == 0  (collapse_side early-exit)
//
// Before the fix, collapse_side() did not handle shape[d]==0 and could:
//   (a) multiply out_width by 0, silently producing out_width=0, and
//   (b) encode shape[d]-1 == -1 as src_dim_size/dst_dim_size in e_dim[],
//       which is an invalid HW descriptor value.
//
// After the fix, any active dimension with shape==0 causes collapse_side()
// to return immediately with out_width=0 and out_extra_count=0, matching
// VPUDMATensor::getContiguousBytes() which also returns 0 in this case.
// ===========================================================================

/// 1D src and dst, each with the single dimension having shape=0.
/// Expected: src_width=0, dst_width=0, num_dim=0, no e_dim entries.
/// Before fix: out_width would be 0 via multiplication AND the e_dim
/// encoding loop would never run (first_non_packed=-1), so this specific
/// 1D case would accidentally produce the right width but could still
/// misbehave in multi-dim cases — see tests below.
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroShape_1D_BothSides_ProducesZeroWidth) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{0, 1}});  // shape=0 — empty tensor
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{0, 1}});

    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "empty 1D src: width must be 0";
    EXPECT_EQ(wl.dst_width, 0) << "empty 1D dst: width must be 0";
    EXPECT_EQ(wl.num_dim, 0) << "no active e_dim entries for empty tensors";
    EXPECT_EQ(wl.getAccessedBytes(), 0) << "empty src: 0 accessed bytes";
    EXPECT_EQ(wl.getWrittenBytes(), 0) << "empty dst: 0 written bytes";
}

/// 2D tensor where the innermost dimension has shape=0.
/// Before fix: the contiguity walk multiplied contiguous_bytes by 0, giving
/// out_width=0 via multiplication, but the e_dim encoding loop would then
/// emit e_dim[0].src_dim_size = shape[outermost]-1 = positive value, while
/// out_width was already zero — inconsistent descriptor.
/// After fix: early-exit → out_width=0, out_extra_count=0, no e_dim entries.
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroShape_2D_InnermostZero_NoEdimEntries) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;
    desc.src.dtype = DataType::UINT8;
    // outermost-first: outer={16, 64}, innermost={0, 1}  — innermost shape is 0
    desc.src.setDimension_OutermostFirst({{16, 64}, {0, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{16, 64}, {0, 1}});

    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "innermost shape=0: width must be 0";
    EXPECT_EQ(wl.dst_width, 0);
    EXPECT_EQ(wl.num_dim, 0) << "no e_dim entries must be emitted for empty tensor";
    // Verify that no e_dim slot carries src_dim_size == -1 (the pre-fix bug).
    for (int i = 0; i < DMANNWorkload_NPU40_50::MaxExtraDimensions; ++i) {
        EXPECT_GE(wl.e_dim[i].src_dim_size, 0) << "e_dim[" << i << "].src_dim_size must not be -1";
        EXPECT_GE(wl.e_dim[i].dst_dim_size, 0) << "e_dim[" << i << "].dst_dim_size must not be -1";
    }
}

/// 2D tensor where the outermost dimension has shape=0.
/// Before fix: the contiguity walk would absorb the innermost dim normally
/// (out_width = innermost elements × elem_bytes), then the e_dim encoding
/// loop would emit shape[outermost]-1 == -1 — an invalid HW value.
/// After fix: early-exit → out_width=0, out_extra_count=0.
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroShape_2D_OutermostZero_NoEdimEntries) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;
    desc.src.dtype = DataType::UINT8;
    // outermost-first: outer={0, 64}, innermost={64, 1}  — outermost shape is 0
    desc.src.setDimension_OutermostFirst({{0, 64}, {64, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{0, 64}, {64, 1}});

    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "outermost shape=0: width must be 0";
    EXPECT_EQ(wl.dst_width, 0);
    EXPECT_EQ(wl.num_dim, 0) << "no e_dim entries must be emitted for empty tensor";
    for (int i = 0; i < DMANNWorkload_NPU40_50::MaxExtraDimensions; ++i) {
        EXPECT_GE(wl.e_dim[i].src_dim_size, 0) << "e_dim[" << i << "].src_dim_size must not be -1";
        EXPECT_GE(wl.e_dim[i].dst_dim_size, 0) << "e_dim[" << i << "].dst_dim_size must not be -1";
    }
}

/// 3D tensor where the middle dimension has shape=0.
/// The innermost dim is packed, the middle has shape=0, outermost is non-zero.
/// Before fix: innermost gets absorbed, then the middle shape=0 gets encoded
/// as e_dim[0].src_dim_size = -1, which is invalid.
/// After fix: early-exit → out_width=0, out_extra_count=0.
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroShape_3D_MiddleDimZero_NoEdimEntries) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;
    desc.src.dtype = DataType::UINT8;
    // outermost-first: {4, 256}, {0, 32}, {32, 1} — middle shape is 0
    desc.src.setDimension_OutermostFirst({{4, 256}, {0, 32}, {32, 1}});
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{4, 256}, {0, 32}, {32, 1}});

    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "middle dim shape=0: width must be 0";
    EXPECT_EQ(wl.dst_width, 0);
    EXPECT_EQ(wl.num_dim, 0) << "no e_dim entries must be emitted for empty tensor";
    for (int i = 0; i < DMANNWorkload_NPU40_50::MaxExtraDimensions; ++i) {
        EXPECT_GE(wl.e_dim[i].src_dim_size, 0) << "e_dim[" << i << "].src_dim_size must not be -1";
        EXPECT_GE(wl.e_dim[i].dst_dim_size, 0) << "e_dim[" << i << "].dst_dim_size must not be -1";
    }
}

/// Asymmetric: src is empty (2D outermost shape=0), dst is a normal packed 2D tensor.
/// Each side must be handled independently: dst produces a valid descriptor while
/// src collapses to width=0 with no e_dim entries.
/// num_dim is driven by the larger of the two extra-dim counts (dst's = 0 here,
/// since dst is fully packed → num_dim=0).
TEST_F(TestDMADescriptorTransformer, D2NN40_ZeroShape_AsymmetricSrcEmptyDstNormal) {
    VPUDMADescriptor desc;
    desc.device = VPUDevice::VPU_4_0;
    desc.src_location = MemoryLocation::DRAM;
    desc.dst_location = MemoryLocation::CMX;

    // src: 2D with outermost shape=0 — empty
    desc.src.dtype = DataType::UINT8;
    desc.src.setDimension_OutermostFirst({{0, 64}, {64, 1}});

    // dst: 2D fully packed — normal
    desc.dst.dtype = DataType::UINT8;
    desc.dst.setDimension_OutermostFirst({{16, 64}, {64, 1}});

    auto wl = transform(desc);

    EXPECT_EQ(wl.src_width, 0) << "empty src: width must be 0";
    EXPECT_EQ(wl.dst_width, 1024) << "normal packed dst: 16*64 = 1024 bytes";
    EXPECT_EQ(wl.num_dim, 0) << "dst is fully packed (0 extra dims), src is empty: num_dim=0";
    EXPECT_EQ(wl.getAccessedBytes(), 0) << "empty src: 0 accessed bytes";
    EXPECT_EQ(wl.getWrittenBytes(), 1024) << "normal dst: 1024 written bytes";
    for (int i = 0; i < DMANNWorkload_NPU40_50::MaxExtraDimensions; ++i) {
        EXPECT_GE(wl.e_dim[i].src_dim_size, 0) << "e_dim[" << i << "].src_dim_size must not be -1";
        EXPECT_GE(wl.e_dim[i].dst_dim_size, 0) << "e_dim[" << i << "].dst_dim_size must not be -1";
    }
}

// ===========================================================================
// N2D_ — fromDMANNWorkload_NPU40_50_to_VPUDMADescriptor
// ===========================================================================
//
// Abbreviation: N2D = "NPU40_50-to-Descriptor"
//
// Encoding convention (inverse of D2NN40_):
//   • src_width / dst_width  → innermost VPUDMATensor block; dtype always UINT8, stride always 1.
//   • e_dim[i] (innermost-first in the HW encoding) → outer VPUDMATensor dimensions.
//   • The resulting VPUDMATensor is stored outermost-first, so the list is reversed:
//       shape[0]        = e_dim[num_dim-1].size + 1   (outermost)
//       byte_strides[0] = e_dim[num_dim-1].stride
//       …
//       shape[num_dim]  = width                       (innermost block)
//       byte_strides[num_dim] = 1
//   • transfer_direction → (src_location, dst_location) via DMAWorkloadTransformer::create_locations().
//   • device is copied verbatim.
//
// Key invariant (checked by expectDescMetricsMatchWl):
//   desc.getSrcTransferBytes()  == wl.getAccessedBytes()
//   desc.getDstTransferBytes()  == wl.getWrittenBytes()
//   desc.getContiguousBytesSrc() == wl.getContiguousBytesSrc()
//   desc.getContiguousBytesDst() == wl.getContiguousBytesDst()
//   desc.getNumContiguousChunksSrc() == wl.getNumContiguousChunksSrc()
//   desc.getNumContiguousChunksDst() == wl.getNumContiguousChunksDst()

// ===========================================================================
// 1D — num_dim == 0 (no extra dimensions)
// ===========================================================================

/// 1D UINT8 transfer DDR→CMX: entire buffer is one contiguous block.
/// Expected: src/dst VPUDMATensor 1D, dtype=UINT8, shape[0]=256, byte_strides[0]=1.
TEST_F(TestDMADescriptorTransformer, N2D_OneDim_UINT8_Packed_DDR2CMX) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 256;
    wl.dst_width = 256;
    wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.device, VPUDevice::VPU_4_0);
    EXPECT_EQ(desc.src_location, MemoryLocation::DRAM);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);

    EXPECT_EQ(desc.src.dtype, DataType::UINT8);
    EXPECT_EQ(desc.src.num_dims, 1);
    EXPECT_EQ(desc.src.shape[0], 256) << "1D: shape == src_width";
    EXPECT_EQ(desc.src.byte_strides[0], 1) << "innermost block: stride == 1";

    EXPECT_EQ(desc.dst.dtype, DataType::UINT8);
    EXPECT_EQ(desc.dst.num_dims, 1);
    EXPECT_EQ(desc.dst.shape[0], 256);
    EXPECT_EQ(desc.dst.byte_strides[0], 1);

    EXPECT_EQ(desc.getSrcTransferBytes(), 256);
    expectDescMetricsMatchWl(desc, wl);
}

/// 1D single-byte CMX→CMX transfer.
TEST_F(TestDMADescriptorTransformer, N2D_OneDim_SingleByte_CMX2CMX) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 1;
    wl.dst_width = 1;
    wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::CMX2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.src.num_dims, 1);
    EXPECT_EQ(desc.src.shape[0], 1);
    EXPECT_EQ(desc.src.byte_strides[0], 1);
    EXPECT_EQ(desc.dst.num_dims, 1);
    EXPECT_EQ(desc.dst.shape[0], 1);
    EXPECT_EQ(desc.dst.byte_strides[0], 1);
    expectDescMetricsMatchWl(desc, wl);
}

// ===========================================================================
// 2D — num_dim == 1 (one extra dimension)
// ===========================================================================

/// 2D 16×64 UINT8, outer gap (stride=128 > width=64).
/// Expected: 2D VPUDMATensor outermost-first:
///   shape[0]=16, byte_strides[0]=128  (outer strided dim)
///   shape[1]=64, byte_strides[1]=1    (innermost byte block)
TEST_F(TestDMADescriptorTransformer, N2D_TwoDim_OuterGap_OneExtraDim) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;
    wl.dst_width = 64;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride  = 128;  wl.e_dim[0].src_dim_size = 15;  // 0-based: 16 rows
    wl.e_dim[0].dst_stride  = 128;  wl.e_dim[0].dst_dim_size = 15;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 2);
    EXPECT_EQ(desc.src.dtype, DataType::UINT8);
    EXPECT_EQ(desc.src.shape[0], 16) << "outermost: e_dim[0].size+1 = 16";
    EXPECT_EQ(desc.src.byte_strides[0], 128) << "outermost stride = 128";
    EXPECT_EQ(desc.src.shape[1], 64) << "innermost block: src_width = 64";
    EXPECT_EQ(desc.src.byte_strides[1], 1) << "innermost block stride = 1";

    EXPECT_EQ(desc.dst.num_dims, 2);
    EXPECT_EQ(desc.dst.shape[0], 16);
    EXPECT_EQ(desc.dst.byte_strides[0], 128);
    EXPECT_EQ(desc.dst.shape[1], 64);
    EXPECT_EQ(desc.dst.byte_strides[1], 1);

    EXPECT_EQ(desc.getSrcTransferBytes(), 64 * 16) << "accessed = 64 * 16 = 1024";
    expectDescMetricsMatchWl(desc, wl);
}

/// 2D with innermost gap: src_width=1, e_dim[0]={stride=2, dim_size=31}.
/// The innermost BYTE block is just 1; the extra dim carries the per-element stride.
/// Expected: shape[0]=32, byte_strides[0]=2; shape[1]=1, byte_strides[1]=1.
TEST_F(TestDMADescriptorTransformer, N2D_TwoDim_InnerStrided_GapAtElemLevel) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 1;
    wl.dst_width = 1;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride  = 2;   wl.e_dim[0].src_dim_size = 31;  // 32 elements, stride 2
    wl.e_dim[0].dst_stride  = 2;   wl.e_dim[0].dst_dim_size = 31;
    wl.transfer_direction = MemoryDirection::CMX2DDR;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 2);
    EXPECT_EQ(desc.src.shape[0], 32) << "outermost: 31+1 = 32";
    EXPECT_EQ(desc.src.byte_strides[0], 2);
    EXPECT_EQ(desc.src.shape[1], 1) << "innermost block = 1 byte";
    EXPECT_EQ(desc.src.byte_strides[1], 1);
    EXPECT_EQ(desc.getSrcTransferBytes(), 32) << "1 * 32 = 32 bytes";
    expectDescMetricsMatchWl(desc, wl);
}

/// 2D packed: src_width=64, e_dim[0]={stride=64, dim_size=15} — stride == width (fully packed).
/// The VPUDMATensor is 2D with shape[0]=16, stride[0]=64, shape[1]=64, stride[1]=1.
/// getContiguousBytes() absorbs both dims → contiguous = 1024.
TEST_F(TestDMADescriptorTransformer, N2D_TwoDim_FullyPacked_BothDimsContiguous) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;
    wl.dst_width = 64;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride  = 64;   wl.e_dim[0].src_dim_size = 15;  // packed
    wl.e_dim[0].dst_stride  = 64;   wl.e_dim[0].dst_dim_size = 15;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 2);
    EXPECT_EQ(desc.src.shape[0], 16);
    EXPECT_EQ(desc.src.byte_strides[0], 64);
    EXPECT_EQ(desc.src.shape[1], 64);
    EXPECT_EQ(desc.src.byte_strides[1], 1);
    // Fully packed: all 1024 bytes contiguous
    EXPECT_EQ(desc.getContiguousBytesSrc(), 1024) << "packed 2D: contiguous == total bytes";
    EXPECT_EQ(desc.getSrcTransferBytes(), 1024);
    expectDescMetricsMatchWl(desc, wl);
}

// ===========================================================================
// 3D — num_dim == 2 (two extra dimensions)
// ===========================================================================

/// 3D {4, 8, 32} UINT8, gap in middle (e_dim[0]=middle, e_dim[1]=outermost).
/// Expected outermost-first VPUDMATensor:
///   shape[0]=4, stride[0]=512   (outermost: e_dim[1])
///   shape[1]=8, stride[1]=128   (middle:    e_dim[0])
///   shape[2]=32, stride[2]=1    (innermost block)
TEST_F(TestDMADescriptorTransformer, N2D_ThreeDim_GapInMiddle_TwoExtraDims) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 32;
    wl.dst_width = 32;
    wl.num_dim = 2;
    wl.e_dim[0].src_stride = 128;  wl.e_dim[0].src_dim_size = 7;   // middle dim (innermost extra)
    wl.e_dim[0].dst_stride = 128;  wl.e_dim[0].dst_dim_size = 7;
    wl.e_dim[1].src_stride = 512;  wl.e_dim[1].src_dim_size = 3;   // outermost dim
    wl.e_dim[1].dst_stride = 512;  wl.e_dim[1].dst_dim_size = 3;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 3);
    EXPECT_EQ(desc.src.dtype, DataType::UINT8);
    EXPECT_EQ(desc.src.shape[0], 4) << "outermost: e_dim[1].size+1 = 4";
    EXPECT_EQ(desc.src.byte_strides[0], 512);
    EXPECT_EQ(desc.src.shape[1], 8) << "middle: e_dim[0].size+1 = 8";
    EXPECT_EQ(desc.src.byte_strides[1], 128);
    EXPECT_EQ(desc.src.shape[2], 32) << "innermost block = src_width";
    EXPECT_EQ(desc.src.byte_strides[2], 1);

    EXPECT_EQ(desc.dst.num_dims, 3);
    EXPECT_EQ(desc.dst.shape[0], 4);
    EXPECT_EQ(desc.dst.byte_strides[0], 512);
    EXPECT_EQ(desc.dst.shape[1], 8);
    EXPECT_EQ(desc.dst.byte_strides[1], 128);
    EXPECT_EQ(desc.dst.shape[2], 32);
    EXPECT_EQ(desc.dst.byte_strides[2], 1);

    EXPECT_EQ(desc.getSrcTransferBytes(), 32 * 8 * 4) << "accessed = 32 * 8 * 4 = 1024";
    expectDescMetricsMatchWl(desc, wl);
}

/// 3D {4, 8, 32}, gap only in outermost (e_dim[0] packed, e_dim[1] strided).
/// src_width=32, e_dim[0]={stride=32, dim_size=7} packed, e_dim[1]={stride=512, dim_size=3} gap.
/// Expected: shape[0]=4, stride[0]=512; shape[1]=8, stride[1]=32; shape[2]=32, stride[2]=1.
/// getContiguousBytes(): d=2 absorbs (32), d=1 stride=32==32 absorbs (256), d=0 stride=512!=256 breaks → 256.
TEST_F(TestDMADescriptorTransformer, N2D_ThreeDim_GapInOutermostOnly_OneContiguousBlock) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 32;
    wl.dst_width = 32;
    wl.num_dim = 2;
    wl.e_dim[0].src_stride = 32;   wl.e_dim[0].src_dim_size = 7;   // packed
    wl.e_dim[0].dst_stride = 32;   wl.e_dim[0].dst_dim_size = 7;
    wl.e_dim[1].src_stride = 512;  wl.e_dim[1].src_dim_size = 3;   // gap (packed would be 256)
    wl.e_dim[1].dst_stride = 512;  wl.e_dim[1].dst_dim_size = 3;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 3);
    EXPECT_EQ(desc.src.shape[0], 4);   EXPECT_EQ(desc.src.byte_strides[0], 512);
    EXPECT_EQ(desc.src.shape[1], 8);   EXPECT_EQ(desc.src.byte_strides[1], 32);
    EXPECT_EQ(desc.src.shape[2], 32);  EXPECT_EQ(desc.src.byte_strides[2], 1);
    EXPECT_EQ(desc.getContiguousBytesSrc(), 256) << "inner 2 dims packed: 32*8 = 256";
    EXPECT_EQ(desc.getSrcTransferBytes(), 32 * 8 * 4);
    expectDescMetricsMatchWl(desc, wl);
}

/// 3D with gap in all dims: src_width=1, e_dim[0]={stride=2,dim_size=31}, e_dim[1]={stride=64,dim_size=7}.
/// Every level has a gap from the byte perspective.
TEST_F(TestDMADescriptorTransformer, N2D_ThreeDim_GapInAllDims_AllLevelsStrided) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 1;
    wl.dst_width = 1;
    wl.num_dim = 2;
    wl.e_dim[0].src_stride = 2;   wl.e_dim[0].src_dim_size = 31;  // 32 elements, 2-byte stride
    wl.e_dim[0].dst_stride = 2;   wl.e_dim[0].dst_dim_size = 31;
    wl.e_dim[1].src_stride = 64;  wl.e_dim[1].src_dim_size = 7;   // 8 rows, 64-byte stride
    wl.e_dim[1].dst_stride = 64;  wl.e_dim[1].dst_dim_size = 7;
    wl.transfer_direction = MemoryDirection::CMX2DDR;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 3);
    EXPECT_EQ(desc.src.shape[0], 8);   EXPECT_EQ(desc.src.byte_strides[0], 64);
    EXPECT_EQ(desc.src.shape[1], 32);  EXPECT_EQ(desc.src.byte_strides[1], 2);
    EXPECT_EQ(desc.src.shape[2], 1);   EXPECT_EQ(desc.src.byte_strides[2], 1);
    EXPECT_EQ(desc.getSrcTransferBytes(), 1 * 32 * 8) << "accessed = 256";
    expectDescMetricsMatchWl(desc, wl);
}

// ===========================================================================
// Asymmetric src/dst layouts
// ===========================================================================

/// Src: 2D strided (width=64, e_dim[0]={stride=128, size=15}).
/// Dst: 1D packed (width=1024, num_dim uses src's extra-dim count → dst e_dim[0] is zero-padded).
/// Expected src VPUDMATensor: 2D {16,128},{64,1}.
/// Expected dst VPUDMATensor: 2D {1,0},{1024,1} — degenerate outermost (shape=1 is skipped by
///   getContiguousBytes), so dst contiguous == 1024 == wl.getContiguousBytesDst().
TEST_F(TestDMADescriptorTransformer, N2D_Asymmetric_SrcStrided_DstPacked) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;
    wl.dst_width = 1024;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride  = 128;  wl.e_dim[0].src_dim_size = 15;
    wl.e_dim[0].dst_stride  = 0;    wl.e_dim[0].dst_dim_size = 0;  // dst has no outer dim
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    // src: 2D with outer gap
    EXPECT_EQ(desc.src.num_dims, 2);
    EXPECT_EQ(desc.src.shape[0], 16);   EXPECT_EQ(desc.src.byte_strides[0], 128);
    EXPECT_EQ(desc.src.shape[1], 64);   EXPECT_EQ(desc.src.byte_strides[1], 1);

    // dst: 2D with degenerate outermost (shape=1, stride=0)
    EXPECT_EQ(desc.dst.num_dims, 2);
    EXPECT_EQ(desc.dst.shape[0], 1) << "dst: e_dim[0].dst_dim_size=0 → shape = 0+1 = 1 (degenerate)";
    EXPECT_EQ(desc.dst.byte_strides[0], 0);
    EXPECT_EQ(desc.dst.shape[1], 1024);  EXPECT_EQ(desc.dst.byte_strides[1], 1);

    EXPECT_EQ(desc.getSrcTransferBytes(), 64 * 16);
    EXPECT_EQ(desc.getDstTransferBytes(), 1 * 1024) << "1 * 1024 = 1024";
    expectDescMetricsMatchWl(desc, wl);
}

/// Both src and dst strided, different shapes.
/// Src: width=64, e_dim[0]={stride=128, size=15}.
/// Dst: width=64, e_dim[0]={stride=256, size=7}.
TEST_F(TestDMADescriptorTransformer, N2D_Asymmetric_BothStrided_DifferentShapes) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;
    wl.dst_width = 64;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride  = 128;  wl.e_dim[0].src_dim_size = 15;
    wl.e_dim[0].dst_stride  = 256;  wl.e_dim[0].dst_dim_size = 7;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 2);
    EXPECT_EQ(desc.src.shape[0], 16);  EXPECT_EQ(desc.src.byte_strides[0], 128);
    EXPECT_EQ(desc.src.shape[1], 64);  EXPECT_EQ(desc.src.byte_strides[1], 1);

    EXPECT_EQ(desc.dst.num_dims, 2);
    EXPECT_EQ(desc.dst.shape[0], 8);   EXPECT_EQ(desc.dst.byte_strides[0], 256);
    EXPECT_EQ(desc.dst.shape[1], 64);  EXPECT_EQ(desc.dst.byte_strides[1], 1);

    EXPECT_EQ(desc.getSrcTransferBytes(), 64 * 16);
    EXPECT_EQ(desc.getDstTransferBytes(), 64 * 8);
    expectDescMetricsMatchWl(desc, wl);
}

// ===========================================================================
// Memory direction → memory locations
// ===========================================================================

TEST_F(TestDMADescriptorTransformer, N2D_Direction_DDR2CMX) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;
    auto desc = transformN2D(wl);
    EXPECT_EQ(desc.src_location, MemoryLocation::DRAM);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);
}

TEST_F(TestDMADescriptorTransformer, N2D_Direction_CMX2DDR) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::CMX2DDR;
    auto desc = transformN2D(wl);
    EXPECT_EQ(desc.src_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.dst_location, MemoryLocation::DRAM);
}

TEST_F(TestDMADescriptorTransformer, N2D_Direction_CMX2CMX) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::CMX2CMX;
    auto desc = transformN2D(wl);
    EXPECT_EQ(desc.src_location, MemoryLocation::CMX);
    EXPECT_EQ(desc.dst_location, MemoryLocation::CMX);
}

TEST_F(TestDMADescriptorTransformer, N2D_Direction_DDR2DDR) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2DDR;
    auto desc = transformN2D(wl);
    EXPECT_EQ(desc.src_location, MemoryLocation::DRAM);
    EXPECT_EQ(desc.dst_location, MemoryLocation::DRAM);
}

// ===========================================================================
// Device passthrough
// ===========================================================================

TEST_F(TestDMADescriptorTransformer, N2D_Device_NPU40_Passthrough) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;
    EXPECT_EQ(transformN2D(wl).device, VPUDevice::VPU_4_0);
}

TEST_F(TestDMADescriptorTransformer, N2D_Device_NPU50_Passthrough) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::NPU_5_0};
    wl.src_width = 64;  wl.dst_width = 64;  wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;
    EXPECT_EQ(transformN2D(wl).device, VPUDevice::NPU_5_0);
}

// ===========================================================================
// dtype is always UINT8
// ===========================================================================

/// The DMANNWorkload_NPU40_50 carries no dtype; the output VPUDMATensor always uses UINT8
/// because widths and strides in the HW descriptor are expressed in bytes.
TEST_F(TestDMADescriptorTransformer, N2D_Dtype_AlwaysUINT8) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 128;
    wl.dst_width = 256;
    wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.dtype, DataType::UINT8) << "src dtype is always UINT8";
    EXPECT_EQ(desc.dst.dtype, DataType::UINT8) << "dst dtype is always UINT8";
}

// ===========================================================================
// 6D — num_dim == 5 (MaxExtraDimensions)
// ===========================================================================

/// 6D tensor: 5 extra dimensions + innermost block of 4 bytes.
/// e_dim[] ordered innermost-first: {8,1},{32,1},{64,1},{128,1},{256,1}.
/// Expected VPUDMATensor (outermost-first, reversed):
///   shape[0]=2, stride[0]=256; shape[1]=2, stride[1]=128; shape[2]=2, stride[2]=64;
///   shape[3]=2, stride[3]=32;  shape[4]=2, stride[4]=8;   shape[5]=4, stride[5]=1.
TEST_F(TestDMADescriptorTransformer, N2D_SixDim_MaxExtraDims) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 4;
    wl.dst_width = 4;
    wl.num_dim = 5;
    wl.e_dim[0].src_stride = 8;    wl.e_dim[0].src_dim_size = 1;   // innermost extra
    wl.e_dim[1].src_stride = 32;   wl.e_dim[1].src_dim_size = 1;
    wl.e_dim[2].src_stride = 64;   wl.e_dim[2].src_dim_size = 1;
    wl.e_dim[3].src_stride = 128;  wl.e_dim[3].src_dim_size = 1;
    wl.e_dim[4].src_stride = 256;  wl.e_dim[4].src_dim_size = 1;   // outermost extra
    for (int i = 0; i < 5; ++i) {
        wl.e_dim[i].dst_stride   = wl.e_dim[i].src_stride;
        wl.e_dim[i].dst_dim_size = wl.e_dim[i].src_dim_size;
    }
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 6);
    EXPECT_EQ(desc.src.dtype, DataType::UINT8);
    // outermost-first = reverse of innermost-first list
    EXPECT_EQ(desc.src.shape[0], 2);  EXPECT_EQ(desc.src.byte_strides[0], 256);  // e_dim[4] outermost
    EXPECT_EQ(desc.src.shape[1], 2);  EXPECT_EQ(desc.src.byte_strides[1], 128);  // e_dim[3]
    EXPECT_EQ(desc.src.shape[2], 2);  EXPECT_EQ(desc.src.byte_strides[2], 64);   // e_dim[2]
    EXPECT_EQ(desc.src.shape[3], 2);  EXPECT_EQ(desc.src.byte_strides[3], 32);   // e_dim[1]
    EXPECT_EQ(desc.src.shape[4], 2);  EXPECT_EQ(desc.src.byte_strides[4], 8);    // e_dim[0] innermost extra
    EXPECT_EQ(desc.src.shape[5], 4);  EXPECT_EQ(desc.src.byte_strides[5], 1);    // byte block

    // total = 4 * 2^5 = 128 bytes
    EXPECT_EQ(desc.getSrcTransferBytes(), 4 * 32) << "4 * 2^5 = 128";
    expectDescMetricsMatchWl(desc, wl);
}

// ===========================================================================
// Edge case: zero width
// ===========================================================================

/// src_width=0, dst_width=0, num_dim=0 → VPUDMATensor 1D with shape[0]=0.
TEST_F(TestDMADescriptorTransformer, N2D_ZeroWidth_ProducesZeroShape) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 0;
    wl.dst_width = 0;
    wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;

    auto desc = transformN2D(wl);

    EXPECT_EQ(desc.src.num_dims, 1);
    EXPECT_EQ(desc.src.shape[0], 0) << "zero-width src: shape[0] = 0";
    EXPECT_EQ(desc.src.byte_strides[0], 1);
    EXPECT_EQ(desc.dst.num_dims, 1);
    EXPECT_EQ(desc.dst.shape[0], 0) << "zero-width dst: shape[0] = 0";
    EXPECT_EQ(desc.getSrcTransferBytes(), 0);
    EXPECT_EQ(desc.getDstTransferBytes(), 0);
}

// ===========================================================================
// Round-trip: metric consistency between DMANNWorkload_NPU40_50 and N2D result
// ===========================================================================

/// For any workload, the metrics of the transformed VPUDMADescriptor must equal
/// the metrics of the original DMANNWorkload_NPU40_50.
TEST_F(TestDMADescriptorTransformer, N2D_RoundTrip_Metrics_1D_Packed) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 512;
    wl.dst_width = 512;
    wl.num_dim = 0;
    wl.transfer_direction = MemoryDirection::DDR2CMX;
    expectDescMetricsMatchWl(transformN2D(wl), wl);
}

TEST_F(TestDMADescriptorTransformer, N2D_RoundTrip_Metrics_2D_Strided) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 64;
    wl.dst_width = 64;
    wl.num_dim = 1;
    wl.e_dim[0].src_stride = 128;  wl.e_dim[0].src_dim_size = 15;
    wl.e_dim[0].dst_stride = 128;  wl.e_dim[0].dst_dim_size = 15;
    wl.transfer_direction = MemoryDirection::CMX2DDR;
    expectDescMetricsMatchWl(transformN2D(wl), wl);
}

TEST_F(TestDMADescriptorTransformer, N2D_RoundTrip_Metrics_3D_GapMiddle) {
    DMANNWorkload_NPU40_50 wl{VPUDevice::VPU_4_0};
    wl.src_width = 32;
    wl.dst_width = 32;
    wl.num_dim = 2;
    wl.e_dim[0].src_stride = 128;  wl.e_dim[0].src_dim_size = 7;
    wl.e_dim[0].dst_stride = 128;  wl.e_dim[0].dst_dim_size = 7;
    wl.e_dim[1].src_stride = 512;  wl.e_dim[1].src_dim_size = 3;
    wl.e_dim[1].dst_stride = 512;  wl.e_dim[1].dst_dim_size = 3;
    wl.transfer_direction = MemoryDirection::DDR2CMX;
    expectDescMetricsMatchWl(transformN2D(wl), wl);
}

}  // namespace VPUNN_unit_tests