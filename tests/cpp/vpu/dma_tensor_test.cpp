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

namespace VPUNN_unit_tests {
using namespace VPUNN;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a 1-D VPUDMATensor (innermost dimension only).
static VPUDMATensor make1D(int32_t num_elems, int32_t stride_bytes, DataType dtype = DataType::UINT8) {
    VPUDMATensor t;
    t.dtype = dtype;
    t.setDimension_OutermostFirst({{num_elems, stride_bytes}});
    return t;
}

/// Build a 2-D VPUDMATensor (outermost-first: shape = {outer, inner}).
/// Strides are byte distances between consecutive element starts per dimension.
static VPUDMATensor make2D(int32_t outer_elems, int32_t outer_stride_bytes, int32_t inner_elems,
                           int32_t inner_stride_bytes, DataType dtype = DataType::UINT8) {
    VPUDMATensor t;
    t.dtype = dtype;
    t.setDimension_OutermostFirst({{outer_elems, outer_stride_bytes}, {inner_elems, inner_stride_bytes}});
    return t;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TestVPUDMATensor : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

// ===========================================================================
// getAccessedBytes — strides are irrelevant, only logical shape × dtype size
// ===========================================================================

TEST_F(TestVPUDMATensor, getAccessedBytes_zero_dims) {
    VPUDMATensor t;
    t.num_dims = 0;
    EXPECT_EQ(t.getAccessedBytes(), 0) << "num_dims=0 should return 0";
    EXPECT_EQ(t.getContiguousBytes(), 0) << "num_dims=0: no contiguous bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 0) << "num_dims=0: no memory footprint";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_1D_packed) {
    auto t = make1D(256, 1);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D UINT8 packed: 256 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "1D UINT8 packed: all 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "1D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_1D_strided_ignores_stride) {
    auto t = make1D(256, 2);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D UINT8 strided: getAccessedBytes counts elements, ignores gaps";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "1D UINT8 stride=2: only 1 byte contiguous per element";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "1D UINT8 stride=2: footprint 256*2 = 512";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_1D_fp16_packed) {
    auto t = make1D(128, 2, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D FP16 packed: 128 * 2 = 256 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "1D FP16 packed: all 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "1D FP16 packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_1D_fp16_strided_ignores_stride) {
    auto t = make1D(128, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D FP16 strided: getAccessedBytes ignores stride gaps";
    EXPECT_EQ(t.getContiguousBytes(), 2) << "1D FP16 stride=4: only 2 bytes (1 element) contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "1D FP16 stride=4: footprint 128*4 = 512";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_2D) {
    auto t = make2D(16, 64, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "2D: 16 * 64 = 1024 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1024) << "2D packed: all 1024 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_3D) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 256}, {8, 32}, {32, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8 * 32) << "3D: 4*8*32 = 1024 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 4 * 8 * 32) << "3D packed: all 1024 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "3D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getAccessedBytes_6D) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{2, 64}, {2, 32}, {2, 16}, {2, 8}, {2, 4}, {4, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 2 * 2 * 2 * 2 * 2) << "6D: 4*2^5 = 128 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 4 * 2 * 2 * 2 * 2 * 2) << "6D packed: all 128 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 128) << "6D packed: footprint == accessed";
}

// ===========================================================================
// Innermost-dimension behaviour — tested through getContiguousBytes()
// ===========================================================================

TEST_F(TestVPUDMATensor, innermost_zero_dims) {
    VPUDMATensor t;
    t.num_dims = 0;
    EXPECT_EQ(t.getAccessedBytes(), 0);
    EXPECT_EQ(t.getContiguousBytes(), 0);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 0);
}

TEST_F(TestVPUDMATensor, innermost_1D_uint8_packed) {
    auto t = make1D(64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "64 UINT8 packed: 64 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 64) << "stride(1)==elem_bytes(1): all 64 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 64) << "packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, innermost_1D_uint8_strided) {
    auto t = make1D(64, 2);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "64 UINT8 strided: 64 logical bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "stride(2) != elem_bytes(1): only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 128) << "footprint 64*2 = 128 (trailing gap included)";
}

TEST_F(TestVPUDMATensor, innermost_1D_fp16_packed) {
    auto t = make1D(64, 2, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 128) << "64 FP16 packed: 128 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 128) << "stride(2)==elem_bytes(2): all 128 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 128) << "packed FP16: footprint == accessed";
}

TEST_F(TestVPUDMATensor, innermost_1D_fp16_strided) {
    auto t = make1D(64, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 128) << "64 FP16 strided: 128 logical bytes";
    EXPECT_EQ(t.getContiguousBytes(), 2) << "stride(4) != elem_bytes(2): only 2 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "footprint 64*4 = 256 (trailing gap included)";
}

TEST_F(TestVPUDMATensor, innermost_2D_inner_dim_drives_contiguity) {
    auto t = make2D(16, 64, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "16*64 UINT8: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1024) << "inner stride=1==elem, outer stride=64==inner: fully contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, innermost_single_element_ignores_stride) {
    auto t = make1D(1, 999);
    EXPECT_EQ(t.getAccessedBytes(), 1) << "1 UINT8 element: 1 accessed byte";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "shape=1: degenerate, stride irrelevant, returns elem_bytes=1";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1) << "single element: footprint = elem_bytes = 1";
}

// ===========================================================================
// getContiguousBytes — full multi-dimensional contiguity analysis
// ===========================================================================

TEST_F(TestVPUDMATensor, getContiguousBytes_zero_dims) {
    VPUDMATensor t;
    t.num_dims = 0;
    EXPECT_EQ(t.getAccessedBytes(), 0);
    EXPECT_EQ(t.getContiguousBytes(), 0);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 0);
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_packed) {
    auto t = make1D(256, 1);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D packed: 256 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "1D packed: all 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "1D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_uint8_strided) {
    auto t = make1D(256, 2);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D UINT8 stride=2: 256 logical bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "1D UINT8 stride=2: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "1D UINT8 stride=2: footprint 256*2 = 512";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_fp16_packed) {
    auto t = make1D(128, 2, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D FP16 packed: 256 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "1D FP16 packed: 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "1D FP16 packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_fp16_strided) {
    auto t = make1D(128, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "1D FP16 stride=4: 256 logical bytes";
    EXPECT_EQ(t.getContiguousBytes(), 2) << "1D FP16 stride=4: only 2 bytes (1 element) contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "1D FP16 stride=4: footprint 128*4 = 512";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_single_element_packed) {
    auto t = make1D(1, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1) << "1 UINT8: 1 accessed byte";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "1 UINT8: 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1) << "1 UINT8: footprint = 1 byte";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_2D_both_packed) {
    auto t = make2D(16, 64, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "2D: 16 * 64 = 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 64 * 16) << "2D fully packed: 1024 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_1D_fp32_packed) {
    auto t = make1D(64, 4, DataType::FLOAT32);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "64 FP32 packed: 256 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "stride(4)==elem_bytes(4): all 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "packed FP32: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_2D_gap_in_outer) {
    auto t = make2D(16, 128, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "2D: 16 * 64 = 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 64) << "2D outer gap: only 64 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1984) << "2D outer gap: footprint 1 + 15*128 + 63 = 1984";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_2D_gap_in_inner_stops_outer) {
    auto t = make2D(16, 1, 64, 2);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "2D 16*64 UINT8: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner gap stops chain: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 143) << "2D inner gap: footprint 15*1 + 64*2 = 143";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_2D_gap_in_inner_and_outer) {
    auto t = make2D(16, 256, 64, 2);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "2D 16*64 UINT8: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner gap stops chain: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 3968) << "2D inner+outer gap: footprint 15*256 + 64*2 = 3968";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_2D_FP16_gap_in_inner_and_outer) {
    auto t = make2D(16, 300, 64, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 16 * 64 * 2) << "2D 16*64 FP16: 2048 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 2) << "inner gap stops chain: only 2 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 4756) << "2D FP16 inner+outer gap: footprint 15*300 + 64*4 = 4756";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_3D_fully_packed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 32 * 8}, {8, 32}, {32, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8 * 32) << "3D packed: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 4 * 8 * 32) << "3D fully packed: 1024 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "3D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_3D_gap_in_outermost) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 512}, {8, 32}, {32, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8 * 32) << "3D: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 32 * 8) << "3D outermost gap: 256 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1792) << "3D outermost gap: footprint 1 + 3*512 + 7*32 + 31 = 1792";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_3D_gap_in_middle_dim) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 1024}, {8, 128}, {32, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8 * 32) << "3D: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 32) << "3D middle gap: only innermost 32 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 4000) << "3D middle gap: footprint 1 + 3*1024 + 7*128 + 31 = 4000";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_6D_fully_packed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{2, 64}, {2, 32}, {2, 16}, {2, 8}, {2, 4}, {4, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 2 * 2 * 2 * 2 * 2) << "6D packed: 128 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 4 * 2 * 2 * 2 * 2 * 2) << "6D fully packed: 128 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 128) << "6D packed: footprint == accessed";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_negative_outer_stride_breaks_chain) {
    auto t = make2D(8, -64, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 8 * 64) << "8*64 UINT8: 512 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 64) << "negative outer stride: only inner 64 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "negative stride: abs value used, footprint 1+7*64+63 = 512";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_broadcast_outer_stride_zero) {
    auto t = make2D(256, 0, 10, 1);
    EXPECT_EQ(t.getAccessedBytes(), 256 * 10) << "256*10 UINT8: 2560 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 10) << "broadcast outer stride=0: only inner 10 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 10) << "broadcast stride=0: footprint = innermost span = 10";
}

TEST_F(TestVPUDMATensor, getContiguousBytes_degenerate_outer_dim_shape_one_not_breaking_chain) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{8, 64}, {1, 999}, {64, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 64 * 1 * 8) << "Logical bytes: 512";
    EXPECT_EQ(t.getContiguousBytes(), 512)
            << "Degenerate dim (shape=1, stride=999) must not break contiguity; expected 512";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "degenerate dim (shape=1): (1-1)*999=0, footprint = 512";
}

// ===========================================================================
// getNumContiguousChunks
// ===========================================================================

TEST_F(TestVPUDMATensor, getNumContiguousChunks_zero_dims) {
    VPUDMATensor t;
    t.num_dims = 0;
    EXPECT_EQ(t.getAccessedBytes(), 0);
    EXPECT_EQ(t.getContiguousBytes(), 0);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 0);
    EXPECT_EQ(t.getNumContiguousChunks(), 0);
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_zero_elements) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{0, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 0) << "zero elements: 0 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 0) << "zero elements: 0 contiguous bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 0) << "zero elements: footprint = 0";
    EXPECT_EQ(t.getNumContiguousChunks(), 0) << "zero elements => 0 chunks";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_1D_packed_single_chunk) {
    auto t = make1D(256, 1);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "256 UINT8 packed: 256 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 256) << "256 UINT8 packed: all contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "256 UINT8 packed: footprint == accessed";
    EXPECT_EQ(t.getNumContiguousChunks(), 1) << "1D packed: single chunk";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_1D_uint8_strided_one_chunk_per_element) {
    auto t = make1D(256, 2);
    EXPECT_EQ(t.getAccessedBytes(), 256);
    EXPECT_EQ(t.getContiguousBytes(), 1);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 512) << "UINT8 stride=2: footprint 256*2 = 512";
    EXPECT_EQ(t.getNumContiguousChunks(), 256) << "1D UINT8 stride=2: 256 chunks of 1 byte";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_1D_fp16_strided_one_chunk_per_element) {
    auto t = make1D(100, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 200);
    EXPECT_EQ(t.getContiguousBytes(), 2);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 400) << "FP16 stride=4: footprint 100*4 = 400";
    EXPECT_EQ(t.getNumContiguousChunks(), 100) << "1D FP16 stride=4: 100 chunks of 2 bytes";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_2D_fully_packed) {
    auto t = make2D(16, 64, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 1024) << "16*64 UINT8: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1024) << "2D fully packed: all contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
    EXPECT_EQ(t.getNumContiguousChunks(), 1) << "2D fully packed: 1 chunk";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_2D_outer_gap_one_chunk_per_row) {
    auto t = make2D(16, 128, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 16 * 64);
    EXPECT_EQ(t.getContiguousBytes(), 64);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1984) << "2D outer gap: footprint 15*128 + 64 = 1984";
    EXPECT_EQ(t.getNumContiguousChunks(), 16) << "One chunk per row: 16";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_3D_outer_gap) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 512}, {8, 32}, {32, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 32 * 8 * 4);
    EXPECT_EQ(t.getContiguousBytes(), 32 * 8);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1792) << "3D outer gap: footprint 3*512 + 7*32 + 32 = 1792";
}

TEST_F(TestVPUDMATensor, getNumContiguousChunks_many_small_chunks) {
    auto t = make2D(100, 4, 1, 1);
    EXPECT_EQ(t.getAccessedBytes(), 100);
    EXPECT_EQ(t.getContiguousBytes(), 1);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 397) << "100 outer rows 1 byte each stride=4: footprint 99*4 + 1 = 397";
    EXPECT_EQ(t.getNumContiguousChunks(), 100) << "100 single-byte chunks";
}

// ===========================================================================
// getMemoryFootprintBytes — extended tests
// ===========================================================================

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_strided_larger_than_accessed) {
    auto t = make1D(10, 4);
    EXPECT_EQ(t.getAccessedBytes(), 10);
    EXPECT_EQ(t.getMemoryFootprintBytes(), 40) << "1D stride=4: footprint 10*4 = 40 > accessed=10";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_fp32_strided) {
    auto t = make1D(10, 8, DataType::FLOAT32);
    EXPECT_EQ(t.getAccessedBytes(), 40) << "accessed ignores gaps: 10 * 4 = 40";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 80) << "FP32 stride=8: footprint 10*8 = 80";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_packed_row_major) {
    auto t = make2D(4, 8, 8, 1);
    EXPECT_EQ(t.getAccessedBytes(), 32);
    EXPECT_EQ(t.getContiguousBytes(), 32) << "fully packed";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 32) << "packed: footprint == accessed == 32";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_packed_fp16) {
    auto t = make2D(8, 32, 16, 2, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 8 * 16 * 2);
    EXPECT_EQ(t.getContiguousBytes(), 8 * 16 * 2) << "fully packed FP16";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 256) << "packed FP16: footprint == accessed == 256";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_outer_row_padding) {
    auto t = make2D(4, 16, 8, 1);
    EXPECT_EQ(t.getAccessedBytes(), 32) << "accessed ignores row padding";
    EXPECT_EQ(t.getContiguousBytes(), 8) << "gap between rows: only one row contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 56) << "row padding: footprint (4-1)*16 + 8*1 = 56";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_large_outer_stride) {
    auto t = make2D(3, 256, 10, 1);
    EXPECT_EQ(t.getAccessedBytes(), 30) << "accessed: 3*10 = 30";
    EXPECT_EQ(t.getContiguousBytes(), 10) << "gap between rows";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 522) << "large outer stride: footprint 2*256 + 10 = 522";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_inner_col_stride) {
    auto t = make2D(4, 16, 8, 2);
    EXPECT_EQ(t.getAccessedBytes(), 32) << "accessed: 4*8 = 32";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner stride=2: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 64) << "inner col stride: footprint 3*16 + 8*2 = 64";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_inner_col_stride_fp16) {
    auto t = make2D(4, 32, 8, 4, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "accessed: 4*8*2 = 64";
    EXPECT_EQ(t.getContiguousBytes(), 2) << "inner FP16 stride=4: only 2 bytes contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 128) << "inner FP16 col stride: footprint 3*32 + 8*4 = 128";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_both_outer_and_inner_gaps) {
    auto t = make2D(4, 32, 8, 2);
    EXPECT_EQ(t.getAccessedBytes(), 32) << "accessed: 4*8 = 32";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner gap: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 112) << "both gaps: footprint 3*32 + 8*2 = 112";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_outer_stride_zero_broadcast) {
    auto t = make2D(8, 0, 16, 1);
    EXPECT_EQ(t.getAccessedBytes(), 128) << "logical: 8*16 = 128";
    EXPECT_EQ(t.getContiguousBytes(), 16) << "inner packed but outer stride=0 breaks outer chain";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 16) << "broadcast: all rows alias same 16 bytes, footprint=16";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_outer_stride_zero_broadcast_fp16) {
    auto t = make2D(4, 0, 32, 2, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 256) << "logical: 4*32*2 = 256";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 64) << "broadcast FP16: all rows alias same 64 bytes";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_outer_stride_zero_broadcast_inner_strided) {
    auto t = make2D(5, 0, 10, 2);
    EXPECT_EQ(t.getAccessedBytes(), 50) << "logical: 5*10 = 50";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner stride=2: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 20)
            << "broadcast+inner gap: all rows alias same 10*2=20 bytes, footprint=20";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_negative_outer_stride) {
    auto t = make2D(4, -8, 8, 1);
    EXPECT_EQ(t.getAccessedBytes(), 32) << "accessed: 4*8 = 32";
    EXPECT_EQ(t.getContiguousBytes(), 8) << "inner packed (8 bytes), but outer stride negative => chain breaks";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 32) << "negative outer stride: abs value = 8, footprint same as packed = 32";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_negative_outer_stride_with_gap) {
    auto t = make2D(4, -16, 8, 1);
    EXPECT_EQ(t.getAccessedBytes(), 32) << "accessed: 4*8 = 32";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 56) << "negative stride with gap: footprint 3*16 + 8 = 56";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_3D_outermost_broadcast_middle_packed_inner_packed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{5, 0}, {4, 8}, {8, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 5 * 4 * 8) << "logical: 160 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 32) << "outermost broadcast: footprint = inner block only, 3*8 + 8 = 32";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_3D_middle_broadcast_outer_and_inner_packed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{4, 16}, {8, 0}, {16, 1}});
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8 * 16) << "logical: 512 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 64) << "middle broadcast: footprint 3*16 + 7*0 + 16*1 = 64";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_3D_outer_gap_middle_broadcast_inner_strided) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{3, 0}, {5, 20}, {10, 2}});
    EXPECT_EQ(t.getAccessedBytes(), 3 * 5 * 10) << "logical: 150 bytes";
    EXPECT_EQ(t.getContiguousBytes(), 1) << "inner stride=2: only 1 byte contiguous";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 100)
            << "outer broadcast, middle packed, inner strided: footprint 2*0 + 4*20 + 10*2 = 100";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_footprint_never_less_than_accessed_for_positive_strides) {
    auto t = make1D(10, 3);
    EXPECT_EQ(t.getAccessedBytes(), 10);
    EXPECT_GE(t.getMemoryFootprintBytes(), t.getAccessedBytes())
            << "positive strides >= dtype_bytes: footprint >= accessed";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 30) << "footprint 10*3 = 30";
}

TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_broadcast_makes_footprint_smaller_than_accessed) {
    auto t = make2D(100, 0, 64, 1);
    EXPECT_EQ(t.getAccessedBytes(), 100 * 64) << "logical: 6400 bytes";
    EXPECT_LT(t.getMemoryFootprintBytes(), t.getAccessedBytes())
            << "broadcast: footprint < accessed because rows alias same memory";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 64) << "broadcast: footprint = 64";
}

// ===========================================================================
// setDimension_OutermostFirst  and  setDimension_InnermostFirst
// Each test sets the same logical tensor via both methods and asserts
// identical field values and query results.
// ===========================================================================

TEST_F(TestVPUDMATensor, setDimension_1D_packed_uint8_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // 1D: OF and IF lists are identical (only one dimension)
    of_t.setDimension_OutermostFirst({{256, 1}});
    if_t.setDimension_InnermostFirst({{256, 1}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 1) << "num_dims must be 1";
        EXPECT_EQ(t->shape[0], 256) << "shape[0] must be 256";
        EXPECT_EQ(t->byte_strides[0], 1) << "byte_strides[0] must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 256) << "1D packed: 256 accessed";
        EXPECT_EQ(t->getContiguousBytes(), 256) << "1D packed: fully contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 256) << "1D packed: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_1D_strided_uint8_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{64, 2}});
    if_t.setDimension_InnermostFirst({{64, 2}});  // 1D: same

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 1);
        EXPECT_EQ(t->shape[0], 64) << "shape[0] must be 64";
        EXPECT_EQ(t->byte_strides[0], 2) << "byte_strides[0] must be 2";
        EXPECT_EQ(t->getAccessedBytes(), 64) << "accessed: 64 elements × 1 byte";
        EXPECT_EQ(t->getContiguousBytes(), 1) << "stride=2 != elem_bytes=1: only 1 byte contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 128) << "footprint: 64*2 = 128";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_1D_fp16_packed_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::FLOAT16;
    of_t.setDimension_OutermostFirst({{128, 2}});
    if_t.setDimension_InnermostFirst({{128, 2}});  // 1D: same

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 1);
        EXPECT_EQ(t->shape[0], 128) << "shape[0] must be 128";
        EXPECT_EQ(t->byte_strides[0], 2) << "byte_strides[0] must be 2";
        EXPECT_EQ(t->getAccessedBytes(), 256) << "FP16: 128*2 = 256";
        EXPECT_EQ(t->getContiguousBytes(), 256) << "packed FP16: fully contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 256) << "packed FP16: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_1D_single_element_large_stride_stores_stride) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{1, 999}});
    if_t.setDimension_InnermostFirst({{1, 999}});  // 1D: same

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 1);
        EXPECT_EQ(t->shape[0], 1) << "shape[0] must be 1";
        EXPECT_EQ(t->byte_strides[0], 999) << "stride 999 must be stored as supplied";
        EXPECT_EQ(t->getAccessedBytes(), 1) << "1 element: 1 byte accessed";
        EXPECT_EQ(t->getContiguousBytes(), 1) << "degenerate: stride irrelevant";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 1) << "degenerate: footprint = elem_bytes";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_packed_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: outermost={16,64}, innermost={64,1}
    // IF: innermost={64,1}, outermost={16,64}  (reversed)
    of_t.setDimension_OutermostFirst({{16, 64}, {64, 1}});
    if_t.setDimension_InnermostFirst({{64, 1}, {16, 64}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2) << "num_dims must be 2";
        EXPECT_EQ(t->shape[0], 16) << "shape[0] (outer) must be 16";
        EXPECT_EQ(t->byte_strides[0], 64) << "byte_strides[0] (outer) must be 64";
        EXPECT_EQ(t->shape[1], 64) << "shape[1] (inner) must be 64";
        EXPECT_EQ(t->byte_strides[1], 1) << "byte_strides[1] (inner) must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 1024) << "16*64 = 1024";
        EXPECT_EQ(t->getContiguousBytes(), 1024) << "fully packed: 1024 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 1024) << "packed: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_outer_gap_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{16, 128}, {64, 1}});
    if_t.setDimension_InnermostFirst({{64, 1}, {16, 128}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->shape[0], 16) << "outer shape must be 16";
        EXPECT_EQ(t->byte_strides[0], 128) << "outer stride must be 128";
        EXPECT_EQ(t->shape[1], 64) << "inner shape must be 64";
        EXPECT_EQ(t->byte_strides[1], 1) << "inner stride must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 1024) << "16*64 = 1024 accessed";
        EXPECT_EQ(t->getContiguousBytes(), 64) << "outer gap: only 64 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 1984) << "footprint: 15*128 + 64 = 1984";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_broadcast_outer_stores_zero_stride) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{256, 0}, {10, 1}});
    if_t.setDimension_InnermostFirst({{10, 1}, {256, 0}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->shape[0], 256) << "outer shape must be 256";
        EXPECT_EQ(t->byte_strides[0], 0) << "outer broadcast stride must be stored as 0";
        EXPECT_EQ(t->shape[1], 10) << "inner shape must be 10";
        EXPECT_EQ(t->byte_strides[1], 1) << "inner stride must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 2560) << "logical: 256*10 = 2560";
        EXPECT_EQ(t->getContiguousBytes(), 10) << "broadcast outer: only inner 10 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 10) << "broadcast: footprint = inner span = 10";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_negative_outer_stride_stores_negative) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{8, -64}, {64, 1}});
    if_t.setDimension_InnermostFirst({{64, 1}, {8, -64}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->shape[0], 8) << "outer shape must be 8";
        EXPECT_EQ(t->byte_strides[0], -64) << "negative stride must be stored as-is";
        EXPECT_EQ(t->shape[1], 64) << "inner shape must be 64";
        EXPECT_EQ(t->byte_strides[1], 1) << "inner stride must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 8 * 64) << "8*64 = 512 accessed";
        EXPECT_EQ(t->getContiguousBytes(), 64) << "negative outer: chain breaks at outer, only 64 contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 512) << "abs(-64): footprint same as packed = 512";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_fp16_inner_strided_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::FLOAT16;
    of_t.setDimension_OutermostFirst({{4, 32}, {8, 4}});
    if_t.setDimension_InnermostFirst({{8, 4}, {4, 32}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->shape[0], 4) << "outer shape must be 4";
        EXPECT_EQ(t->byte_strides[0], 32) << "outer stride must be 32";
        EXPECT_EQ(t->shape[1], 8) << "inner shape must be 8";
        EXPECT_EQ(t->byte_strides[1], 4) << "inner stride must be 4";
        EXPECT_EQ(t->getAccessedBytes(), 64) << "accessed: 4*8*2 = 64";
        EXPECT_EQ(t->getContiguousBytes(), 2) << "inner FP16 stride=4: only 2 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 128) << "footprint: 3*32 + 8*4 = 128";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_3D_fully_packed_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: outermost → innermost = {4,256}, {8,32}, {32,1}
    // IF: innermost → outermost = {32,1}, {8,32}, {4,256}
    of_t.setDimension_OutermostFirst({{4, 256}, {8, 32}, {32, 1}});
    if_t.setDimension_InnermostFirst({{32, 1}, {8, 32}, {4, 256}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 3);
        EXPECT_EQ(t->shape[0], 4);
        EXPECT_EQ(t->byte_strides[0], 256) << "outermost stride";
        EXPECT_EQ(t->shape[1], 8);
        EXPECT_EQ(t->byte_strides[1], 32) << "middle stride";
        EXPECT_EQ(t->shape[2], 32);
        EXPECT_EQ(t->byte_strides[2], 1) << "innermost stride";
        EXPECT_EQ(t->getAccessedBytes(), 4 * 8 * 32) << "accessed: 1024";
        EXPECT_EQ(t->getContiguousBytes(), 4 * 8 * 32) << "fully packed: 1024 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 1024) << "packed: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_3D_outermost_broadcast_stores_zero_stride) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{5, 0}, {4, 8}, {8, 1}});
    if_t.setDimension_InnermostFirst({{8, 1}, {4, 8}, {5, 0}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 3);
        EXPECT_EQ(t->shape[0], 5);
        EXPECT_EQ(t->byte_strides[0], 0) << "outermost broadcast stride=0";
        EXPECT_EQ(t->shape[1], 4);
        EXPECT_EQ(t->byte_strides[1], 8) << "middle stride=8";
        EXPECT_EQ(t->shape[2], 8);
        EXPECT_EQ(t->byte_strides[2], 1) << "innermost stride=1";
        EXPECT_EQ(t->getAccessedBytes(), 5 * 4 * 8) << "logical: 160 bytes";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 32) << "broadcast outer: footprint = 3*8 + 8 = 32";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

/// Verifies that the pair ordering convention is correctly handled:
/// OF stores index 0 = outermost; IF stores the reversed list so index 0 is still outermost.
TEST_F(TestVPUDMATensor, setDimension_pair_order_outermost_first_verified_by_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: first pair → outermost (index 0), last pair → innermost (index 2)
    of_t.setDimension_OutermostFirst({{10, 500}, {20, 200}, {30, 1}});
    // IF: first pair → innermost (index 2), last pair → outermost (index 0)
    if_t.setDimension_InnermostFirst({{30, 1}, {20, 200}, {10, 500}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->shape[0], 10) << "outermost (index 0): shape must be 10";
        EXPECT_EQ(t->byte_strides[0], 500) << "outermost (index 0): stride must be 500";
        EXPECT_EQ(t->shape[1], 20) << "middle (index 1): shape must be 20";
        EXPECT_EQ(t->byte_strides[1], 200) << "middle (index 1): stride must be 200";
        EXPECT_EQ(t->shape[2], 30) << "innermost (index 2): shape must be 30";
        EXPECT_EQ(t->byte_strides[2], 1) << "innermost (index 2): stride must be 1";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_does_not_modify_dtype) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::FLOAT16;

    of_t.setDimension_OutermostFirst({{64, 2}});
    if_t.setDimension_InnermostFirst({{64, 2}});
    EXPECT_EQ(of_t.dtype, DataType::FLOAT16) << "OF: dtype must be unchanged after 1D call";
    EXPECT_EQ(if_t.dtype, DataType::FLOAT16) << "IF: dtype must be unchanged after 1D call";

    of_t.setDimension_OutermostFirst({{4, 256}, {8, 32}, {32, 1}});
    if_t.setDimension_InnermostFirst({{32, 1}, {8, 32}, {4, 256}});
    EXPECT_EQ(of_t.dtype, DataType::FLOAT16) << "OF: dtype must be unchanged after 3D call";
    EXPECT_EQ(if_t.dtype, DataType::FLOAT16) << "IF: dtype must be unchanged after 3D call";
}

TEST_F(TestVPUDMATensor, setDimension_overwrites_num_dims_and_fields_on_second_call) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;

    of_t.setDimension_OutermostFirst({{4, 256}, {8, 32}, {32, 1}});
    if_t.setDimension_InnermostFirst({{32, 1}, {8, 32}, {4, 256}});
    EXPECT_EQ(of_t.num_dims, 3);
    EXPECT_EQ(if_t.num_dims, 3);

    // Overwrite with a 1D descriptor
    of_t.setDimension_OutermostFirst({{100, 1}});
    if_t.setDimension_InnermostFirst({{100, 1}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 1) << "num_dims must be overwritten to 1";
        EXPECT_EQ(t->shape[0], 100) << "shape[0] must be 100";
        EXPECT_EQ(t->byte_strides[0], 1) << "byte_strides[0] must be 1";
        EXPECT_EQ(t->getAccessedBytes(), 100) << "1D packed 100: 100 accessed";
        EXPECT_EQ(t->getContiguousBytes(), 100) << "1D packed 100: fully contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 100) << "1D packed 100: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_num_dims_tracks_pair_count_1_through_6) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;

    of_t.setDimension_OutermostFirst({{1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}});
    EXPECT_EQ(of_t.num_dims, 1);
    EXPECT_EQ(if_t.num_dims, 1);

    of_t.setDimension_OutermostFirst({{1, 1}, {1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}, {1, 1}});
    EXPECT_EQ(of_t.num_dims, 2);
    EXPECT_EQ(if_t.num_dims, 2);

    of_t.setDimension_OutermostFirst({{1, 1}, {1, 1}, {1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}, {1, 1}, {1, 1}});
    EXPECT_EQ(of_t.num_dims, 3);
    EXPECT_EQ(if_t.num_dims, 3);

    of_t.setDimension_OutermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
    EXPECT_EQ(of_t.num_dims, 4);
    EXPECT_EQ(if_t.num_dims, 4);

    of_t.setDimension_OutermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    EXPECT_EQ(of_t.num_dims, 5);
    EXPECT_EQ(if_t.num_dims, 5);

    of_t.setDimension_OutermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    if_t.setDimension_InnermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    EXPECT_EQ(of_t.num_dims, 6);
    EXPECT_EQ(if_t.num_dims, 6);
}

TEST_F(TestVPUDMATensor, setDimension_2D_degenerate_inner_dim_shape1) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: outermost={8,1}, innermost={1,999}
    // IF: innermost={1,999}, outermost={8,1}
    of_t.setDimension_OutermostFirst({{8, 1}, {1, 999}});
    if_t.setDimension_InnermostFirst({{1, 999}, {8, 1}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->shape[0], 8);
        EXPECT_EQ(t->byte_strides[0], 1);
        EXPECT_EQ(t->shape[1], 1);
        EXPECT_EQ(t->byte_strides[1], 999) << "stride stored as-is";
        EXPECT_EQ(t->getAccessedBytes(), 8) << "8*1 = 8 accessed";
        EXPECT_EQ(t->getContiguousBytes(), 8) << "inner shape=1 degenerate: outer stride=1==contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 8) << "footprint: (8-1)*1 + 1*1 = 8";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_2D_both_broadcast_strides_zero) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    of_t.setDimension_OutermostFirst({{5, 0}, {1, 0}});
    if_t.setDimension_InnermostFirst({{1, 0}, {5, 0}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 2);
        EXPECT_EQ(t->byte_strides[0], 0) << "outer broadcast stride=0 stored";
        EXPECT_EQ(t->byte_strides[1], 0) << "inner broadcast stride=0 stored";
        EXPECT_EQ(t->getAccessedBytes(), 5) << "logical: 5 elements";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 1) << "all alias same single byte: footprint=1";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

// ===========================================================================
// setDimension_OutermostFirst — error handling (invalid dimension count)
// ===========================================================================

TEST_F(TestVPUDMATensor, setDimension_OutermostFirst_throws_on_zero_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    EXPECT_THROW(t.setDimension_OutermostFirst({}), std::invalid_argument)
            << "Empty initializer list (0 dims) must throw std::invalid_argument";
}

TEST_F(TestVPUDMATensor, setDimension_OutermostFirst_throws_on_too_many_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    EXPECT_THROW(t.setDimension_OutermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}),
                 std::invalid_argument)
            << "7 dimensions (> VPU_DMA_MAX_DIMS=6) must throw std::invalid_argument";
}

TEST_F(TestVPUDMATensor, setDimension_OutermostFirst_no_throw_on_max_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    EXPECT_NO_THROW(t.setDimension_OutermostFirst({{2, 64}, {2, 32}, {2, 16}, {2, 8}, {2, 4}, {4, 1}}))
            << "Exactly VPU_DMA_MAX_DIMS=6 dimensions must not throw";
    EXPECT_EQ(t.num_dims, 6);
}

// ===========================================================================
// setDimension_OutermostFirst — 4D and 5D field-level verification
// ===========================================================================

TEST_F(TestVPUDMATensor, setDimension_4D_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: outermost→innermost = {2,120}, {3,40}, {4,10}, {10,1}
    // IF: reversed             = {10,1},  {4,10}, {3,40}, {2,120}
    of_t.setDimension_OutermostFirst({{2, 120}, {3, 40}, {4, 10}, {10, 1}});
    if_t.setDimension_InnermostFirst({{10, 1}, {4, 10}, {3, 40}, {2, 120}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 4);
        EXPECT_EQ(t->shape[0], 2);
        EXPECT_EQ(t->byte_strides[0], 120) << "outermost stride";
        EXPECT_EQ(t->shape[1], 3);
        EXPECT_EQ(t->byte_strides[1], 40) << "second stride";
        EXPECT_EQ(t->shape[2], 4);
        EXPECT_EQ(t->byte_strides[2], 10) << "third stride";
        EXPECT_EQ(t->shape[3], 10);
        EXPECT_EQ(t->byte_strides[3], 1) << "innermost stride";
        EXPECT_EQ(t->getAccessedBytes(), 2 * 3 * 4 * 10) << "accessed: 240 bytes";
        EXPECT_EQ(t->getContiguousBytes(), 2 * 3 * 4 * 10) << "fully packed 4D: 240 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 240) << "packed: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

TEST_F(TestVPUDMATensor, setDimension_5D_stores_correct_fields) {
    VPUDMATensor of_t, if_t;
    of_t.dtype = if_t.dtype = DataType::UINT8;
    // OF: outermost→innermost = {2,60}, {3,20}, {4,5}, {5,1}, ... wait - need packed strides
    // 2*3*4*5*1 = 120 total; packed: innermost stride=1, then 5,20,60,120... but last dim is outermost
    // shape: {2,3,4,5,6}, packed strides: {3*4*5*6, 4*5*6, 5*6, 6, 1} = {360,120,30,6,1}
    of_t.setDimension_OutermostFirst({{2, 360}, {3, 120}, {4, 30}, {5, 6}, {6, 1}});
    if_t.setDimension_InnermostFirst({{6, 1}, {5, 6}, {4, 30}, {3, 120}, {2, 360}});

    for (auto* t : {&of_t, &if_t}) {
        EXPECT_EQ(t->num_dims, 5);
        EXPECT_EQ(t->shape[0], 2);
        EXPECT_EQ(t->byte_strides[0], 360) << "outermost stride";
        EXPECT_EQ(t->shape[1], 3);
        EXPECT_EQ(t->byte_strides[1], 120) << "second stride";
        EXPECT_EQ(t->shape[2], 4);
        EXPECT_EQ(t->byte_strides[2], 30) << "third stride";
        EXPECT_EQ(t->shape[3], 5);
        EXPECT_EQ(t->byte_strides[3], 6) << "fourth stride";
        EXPECT_EQ(t->shape[4], 6);
        EXPECT_EQ(t->byte_strides[4], 1) << "innermost stride";
        EXPECT_EQ(t->getAccessedBytes(), 2 * 3 * 4 * 5 * 6) << "accessed: 720 bytes";
        EXPECT_EQ(t->getContiguousBytes(), 2 * 3 * 4 * 5 * 6) << "fully packed 5D: 720 bytes contiguous";
        EXPECT_EQ(t->getMemoryFootprintBytes(), 720) << "packed: footprint == accessed";
    }
    EXPECT_EQ(of_t.shape, if_t.shape);
    EXPECT_EQ(of_t.byte_strides, if_t.byte_strides);
}

// ===========================================================================
// setDimension_InnermostFirst — error handling (invalid dimension count)
// ===========================================================================

TEST_F(TestVPUDMATensor, setDimension_InnermostFirst_throws_on_zero_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    EXPECT_THROW(t.setDimension_InnermostFirst({}), std::invalid_argument)
            << "Empty initializer list (0 dims) must throw std::invalid_argument";
}

TEST_F(TestVPUDMATensor, setDimension_InnermostFirst_throws_on_too_many_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    EXPECT_THROW(t.setDimension_InnermostFirst({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}),
                 std::invalid_argument)
            << "7 dimensions (> VPU_DMA_MAX_DIMS=6) must throw std::invalid_argument";
}

TEST_F(TestVPUDMATensor, setDimension_InnermostFirst_no_throw_on_max_dims) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    // Innermost-first: innermost={4,1}, ..., outermost={2,64}
    EXPECT_NO_THROW(t.setDimension_InnermostFirst({{4, 1}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}}))
            << "Exactly VPU_DMA_MAX_DIMS=6 dimensions must not throw";
    EXPECT_EQ(t.num_dims, 6);
    // Stored outermost-first, so index 0 must be the last pair supplied (outermost = {2,64})
    EXPECT_EQ(t.shape[0], 2) << "outermost shape stored at index 0";
    EXPECT_EQ(t.byte_strides[0], 64) << "outermost stride stored at index 0";
    EXPECT_EQ(t.shape[5], 4) << "innermost shape stored at index 5";
    EXPECT_EQ(t.byte_strides[5], 1) << "innermost stride stored at index 5";
}

// ===========================================================================
// setDimension_InnermostFirst — slot isolation after dimension count changes
// ===========================================================================

/// After a 3D IF call followed by a 2D IF call, slots [2..5] must be zeroed
/// by the second call so that consumers reading all VPU_DMA_MAX_DIMS entries
/// never see stale values.
TEST_F(TestVPUDMATensor, setDimension_InnermostFirst_slots_beyond_num_dims_are_zeroed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;

    // First call: 3D IF — stored outermost-first as [{4,256},{8,32},{32,1}]
    t.setDimension_InnermostFirst({{32, 1}, {8, 32}, {4, 256}});
    EXPECT_EQ(t.num_dims, 3);
    EXPECT_EQ(t.shape[2], 32) << "slot 2 (innermost) set by 3D IF call";
    EXPECT_EQ(t.byte_strides[2], 1) << "slot 2 stride set by 3D IF call";

    // Second call: 2D IF — stored outermost-first as [{16,64},{64,1}]; slot 2 must be zeroed
    t.setDimension_InnermostFirst({{64, 1}, {16, 64}});
    EXPECT_EQ(t.num_dims, 2) << "num_dims updated to 2";
    EXPECT_EQ(t.shape[0], 16) << "slot 0 (outermost) updated";
    EXPECT_EQ(t.byte_strides[0], 64) << "slot 0 stride updated";
    EXPECT_EQ(t.shape[1], 64) << "slot 1 (innermost) updated";
    EXPECT_EQ(t.byte_strides[1], 1) << "slot 1 stride updated";
    EXPECT_EQ(t.shape[2], 0) << "slot 2 must be zeroed after 2D IF call (no stale value)";
    EXPECT_EQ(t.byte_strides[2], 0) << "slot 2 stride must be zeroed after 2D IF call";

    // Query methods must return correct 2D results
    EXPECT_EQ(t.getAccessedBytes(), 16 * 64) << "2D tensor: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 16 * 64) << "2D fully packed: 1024 contiguous bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
}

// ===========================================================================
// setDimension_OutermostFirst — slot isolation after dimension count changes
// ===========================================================================

/// After a 3D call followed by a 2D call, slots [2..5] must be zeroed by the
/// second call so that consumers reading all VPU_DMA_MAX_DIMS entries (e.g.
/// CSV serialization) never see stale values.
TEST_F(TestVPUDMATensor, setDimension_OutermostFirst_slots_beyond_num_dims_are_zeroed) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;

    // First call: 3D
    t.setDimension_OutermostFirst({{4, 256}, {8, 32}, {32, 1}});
    EXPECT_EQ(t.num_dims, 3);
    EXPECT_EQ(t.shape[2], 32) << "slot 2 set by 3D call";
    EXPECT_EQ(t.byte_strides[2], 1) << "slot 2 stride set by 3D call";

    // Second call: 2D — slots [2..5] must be zeroed
    t.setDimension_OutermostFirst({{16, 64}, {64, 1}});
    EXPECT_EQ(t.num_dims, 2) << "num_dims updated to 2";
    EXPECT_EQ(t.shape[0], 16) << "slot 0 updated";
    EXPECT_EQ(t.shape[1], 64) << "slot 1 updated";
    EXPECT_EQ(t.shape[2], 0) << "slot 2 must be zeroed after 2D call (no stale value)";
    EXPECT_EQ(t.byte_strides[2], 0) << "slot 2 stride must be zeroed after 2D call";

    // Query methods must return correct 2D results
    EXPECT_EQ(t.getAccessedBytes(), 16 * 64) << "2D tensor: 1024 accessed bytes";
    EXPECT_EQ(t.getContiguousBytes(), 16 * 64) << "2D fully packed: 1024 contiguous bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1024) << "2D packed: footprint == accessed";
}

// ===========================================================================
// getMemoryFootprintBytes — broadcast (stride==0) at the INNERMOST dimension
//
// When byte_strides[innermost] == 0 all shape[innermost] logical elements alias
// the same physical byte address.  The tensor is non-empty and still occupies
// exactly elem_bytes of physical memory regardless of shape[innermost].
//
// Expected: footprint = elem_bytes  (not 0, not shape*0)
// ===========================================================================

// 1D, UINT8, broadcast innermost, shape > 1 → footprint must equal elem_bytes (1).
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_broadcast_innermost_uint8) {
    auto t = make1D(64, /*stride=*/0);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "logical: 64 elements × 1 byte = 64";
    // All 64 elements alias the same byte → physical footprint is 1 byte.
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1)
            << "broadcast innermost (stride=0, shape=64): footprint must be elem_bytes=1, not 0";
}

// 1D, FLOAT16, broadcast innermost, shape > 1 → footprint must equal elem_bytes (2).
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_broadcast_innermost_fp16) {
    auto t = make1D(32, /*stride=*/0, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "logical: 32 elements × 2 bytes = 64";
    // All 32 elements alias the same FP16 value → physical footprint is 2 bytes.
    EXPECT_EQ(t.getMemoryFootprintBytes(), 2)
            << "broadcast innermost (stride=0, shape=32): footprint must be elem_bytes=2, not 0";
}

// 1D, FLOAT32, broadcast innermost, shape > 1 → footprint must equal elem_bytes (4).
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_broadcast_innermost_fp32) {
    auto t = make1D(16, /*stride=*/0, DataType::FLOAT32);
    EXPECT_EQ(t.getAccessedBytes(), 64) << "logical: 16 elements × 4 bytes = 64";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 4)
            << "broadcast innermost (stride=0, shape=16): footprint must be elem_bytes=4, not 0";
}

// 2D: packed outer dimension, broadcast innermost → outer contribution is normal,
// inner contribution must be elem_bytes (not 0).
// Layout: outer={4 rows, stride=1 (pointing to the single aliased byte)},
//         inner={8 elems, stride=0 (broadcast)}.
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_packed_outer_broadcast_innermost_uint8) {
    // outer stride=1 means each "row start" is 1 byte apart; inner broadcast means all
    // 8 columns within a row alias the same byte.
    // Footprint = (outer_shape - 1) * |outer_stride| + elem_bytes
    //           = (4 - 1) * 1 + 1 = 4
    auto t = make2D(/*outer*/ 4, /*outer_stride*/ 1, /*inner*/ 8, /*inner_stride*/ 0);
    EXPECT_EQ(t.getAccessedBytes(), 4 * 8) << "logical: 32 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 4)
            << "2D packed-outer + broadcast-inner: footprint = (4-1)*1 + elem_bytes(1) = 4";
}

// 2D: strided outer dimension, broadcast innermost.
// outer={3 rows, stride=16}, inner={10 elems, stride=0 (broadcast)}.
// Footprint = (3-1)*16 + elem_bytes = 32 + 1 = 33
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_strided_outer_broadcast_innermost_uint8) {
    auto t = make2D(3, 16, 10, 0);
    EXPECT_EQ(t.getAccessedBytes(), 30) << "logical: 3*10 = 30 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 33)
            << "2D strided-outer + broadcast-inner: footprint = (3-1)*16 + elem_bytes(1) = 33";
}

// 2D FP16: strided outer, broadcast innermost.
// outer={4 rows, stride=32}, inner={16 elems, stride=0 (broadcast)}.
// elem_bytes=2.  Footprint = (4-1)*32 + 2 = 96 + 2 = 98
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_2D_strided_outer_broadcast_innermost_fp16) {
    auto t = make2D(4, 32, 16, 0, DataType::FLOAT16);
    EXPECT_EQ(t.getAccessedBytes(), 4 * 16 * 2) << "logical: 128 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 98)
            << "2D FP16 strided-outer + broadcast-inner: footprint = (4-1)*32 + 2 = 98";
}

// Both dimensions broadcast (all elements alias the same single elem).
// 1D degenerate: shape=1, stride=0 → footprint = shape*elem_bytes = 1*1 = 1 (the shape<=1 branch).
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_1D_broadcast_innermost_shape_one) {
    auto t = make1D(1, 0);
    EXPECT_EQ(t.getAccessedBytes(), 1) << "single element";
    // shape==1 branch: footprint = 1 * elem_bytes = 1. Unchanged by the broadcast fix.
    EXPECT_EQ(t.getMemoryFootprintBytes(), 1)
            << "shape=1 + stride=0: footprint = elem_bytes = 1 (degenerate, not the broadcast path)";
}

// 3D: broadcast outermost, packed middle, broadcast innermost.
// outer={5, stride=0 (broadcast)}, middle={4, stride=8}, inner={8, stride=0 (broadcast)}.
// Footprint = (5-1)*0  (outer broadcast → 0)
//           + (4-1)*8  (middle packed → 24)
//           + elem_bytes (inner broadcast → 1)
//           = 0 + 24 + 1 = 25
TEST_F(TestVPUDMATensor, getMemoryFootprintBytes_3D_broadcast_outer_packed_middle_broadcast_innermost) {
    VPUDMATensor t;
    t.dtype = DataType::UINT8;
    t.setDimension_OutermostFirst({{5, 0}, {4, 8}, {8, 0}});
    EXPECT_EQ(t.getAccessedBytes(), 5 * 4 * 8) << "logical: 160 bytes";
    EXPECT_EQ(t.getMemoryFootprintBytes(), 25)
            << "3D outer-broadcast + packed-middle + inner-broadcast: footprint = 4*0 + 3*8 + elem_bytes(1) = 25";
}

}  // namespace VPUNN_unit_tests
