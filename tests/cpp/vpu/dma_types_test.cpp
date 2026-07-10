// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/dma_types.h"

#include <gtest/gtest.h>

#include "vpu/cycles_interface_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMATypes : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUDevice device{VPUDevice::VPU_4_0};  ///< default device for tests

private:
};

/// Test cases covering the getAccessedBytes  for DMANNWorkload_NPU40_50 class
TEST_F(TestDMATypes, getAccessedBytes_test) {
    // Test 1D transfer (num_dim = 0)
    {
        DMANNWorkload_NPU40_50 wl{device, 256, 256, 0};
        EXPECT_EQ(wl.getAccessedBytes(), 256) << "1D transfer: should return src_width";
    }

    // Test 2D transfer (num_dim = 1)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements (0-based, so 15+1)
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 64;

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 16) << "2D transfer: 64 bytes * 16 rows = 1024 bytes";
    }

    // Test 3D transfer (num_dim = 2)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: 8 elements (7+1)
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;
        wl.e_dim[0].dst_stride = 32;

        // Second dimension: 4 elements (3+1)
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;  // 32 * 8
        wl.e_dim[1].dst_stride = 256;

        EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 4) << "3D transfer: 32 * 8 * 4 = 1024 bytes";
    }

    // Test 4D transfer (num_dim = 3)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 16;
        wl.dst_width = 16;
        wl.num_dim = 3;

        wl.e_dim[0].src_dim_size = 3;  // 4 elements
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 16;
        wl.e_dim[0].dst_stride = 16;

        wl.e_dim[1].src_dim_size = 1;  // 2 elements
        wl.e_dim[1].dst_dim_size = 1;
        wl.e_dim[1].src_stride = 64;  // 16 * 4
        wl.e_dim[1].dst_stride = 64;

        wl.e_dim[2].src_dim_size = 2;  // 3 elements
        wl.e_dim[2].dst_dim_size = 2;
        wl.e_dim[2].src_stride = 128;  // 16 * 4 * 2
        wl.e_dim[2].dst_stride = 128;

        EXPECT_EQ(wl.getAccessedBytes(), 16 * 4 * 2 * 3) << "4D transfer: 16 * 4 * 2 * 3 = 384 bytes";
    }

    // Test single element transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;
        wl.dst_width = 1;
        wl.num_dim = 0;

        EXPECT_EQ(wl.getAccessedBytes(), 1) << "Single byte transfer";
    }

    // Test maximum dimensions (6D, num_dim = 5)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 4;
        wl.dst_width = 4;
        wl.num_dim = 5;

        // Set all dimensions to 1 element each (0 means 1 element)
        int accumulated_stride = 4;
        for (int i = 0; i < 5; i++) {
            wl.e_dim[i].src_dim_size = 1;  // 2 elements each
            wl.e_dim[i].dst_dim_size = 1;
            wl.e_dim[i].src_stride = accumulated_stride;
            wl.e_dim[i].dst_stride = accumulated_stride;
            accumulated_stride *= 2;  // Each dimension has 2 elements
        }

        EXPECT_EQ(wl.getAccessedBytes(), 4 * 2 * 2 * 2 * 2 * 2) << "6D transfer: 4 * 2^5 = 128 bytes";
    }

    // Test with zero-based dimension sizes
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 100;
        wl.dst_width = 100;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 0;  // 1 element (0-based)
        wl.e_dim[0].dst_dim_size = 0;
        wl.e_dim[0].src_stride = 100;
        wl.e_dim[0].dst_stride = 100;

        wl.e_dim[1].src_dim_size = 0;  // 1 element (0-based)
        wl.e_dim[1].dst_dim_size = 0;
        wl.e_dim[1].src_stride = 100;
        wl.e_dim[1].dst_stride = 100;

        EXPECT_EQ(wl.getAccessedBytes(), 100 * 1 * 1) << "Zero-based dims: 100 * 1 * 1 = 100 bytes";
    }

    // Test large transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1024;
        wl.dst_width = 1024;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 63;  // 64 elements
        wl.e_dim[0].dst_dim_size = 63;
        wl.e_dim[0].src_stride = 1024;
        wl.e_dim[0].dst_stride = 1024;

        wl.e_dim[1].src_dim_size = 255;  // 256 elements
        wl.e_dim[1].dst_dim_size = 255;
        wl.e_dim[1].src_stride = 65536;  // 1024 * 64
        wl.e_dim[1].dst_stride = 65536;

        EXPECT_EQ(wl.getAccessedBytes(), 1024 * 64 * 256) << "Large transfer: 1024 * 64 * 256 = 16777216 bytes";
    }
}

/// Test cases covering the getContiguousBytesSrc for DMANNWorkload_NPU40_50 class
TEST_F(TestDMATypes, getContiguousBytesSrc_test) {
    // Test 1D transfer (num_dim = 0) - always contiguous
    {
        DMANNWorkload_NPU40_50 wl{device, 256, 256, 0};
        EXPECT_EQ(wl.getContiguousBytesSrc(), 256) << "1D transfer: all bytes are contiguous";
    }

    // Test 2D transfer with contiguous memory (stride equals width)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;  // Contiguous: stride = width
        wl.e_dim[0].dst_stride = 64;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 64 * 16) << "2D contiguous: 64 * 16 = 1024 bytes";
    }

    // Test 2D transfer with gap (stride > width) - not contiguous
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 128;  // Gap: stride > width, only 64 bytes contiguous per row
        wl.e_dim[0].dst_stride = 128;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 64) << "2D with gap: only first dimension is contiguous (64 bytes)";
    }

    // Test 3D fully contiguous transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: 8 elements, contiguous
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 32;

        // Second dimension: 4 elements, contiguous
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;  // Contiguous: 32 * 8
        wl.e_dim[1].dst_stride = 256;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 32 * 8 * 4) << "3D fully contiguous: 32 * 8 * 4 = 1024 bytes";
    }

    // Test 3D partially contiguous (gap in second dimension)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: 8 elements, contiguous
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 32;

        // Second dimension: has gap
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 512;  // Gap: should be 256 for contiguous
        wl.e_dim[1].dst_stride = 512;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 32 * 8) << "3D partial: contiguous up to first dim = 256 bytes";
    }

    // Test 4D with gap in middle dimension
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 16;
        wl.dst_width = 16;
        wl.num_dim = 3;

        wl.e_dim[0].src_dim_size = 3;  // 4 elements
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 16;  // Contiguous
        wl.e_dim[0].dst_stride = 16;

        wl.e_dim[1].src_dim_size = 1;  // 2 elements, but has gap
        wl.e_dim[1].dst_dim_size = 1;
        wl.e_dim[1].src_stride = 128;  // Gap: should be 64 for contiguous
        wl.e_dim[1].dst_stride = 128;

        wl.e_dim[2].src_dim_size = 2;  // 3 elements
        wl.e_dim[2].dst_dim_size = 2;
        wl.e_dim[2].src_stride = 256;
        wl.e_dim[2].dst_stride = 256;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 16 * 4) << "4D with gap in dim 1: contiguous = 16 * 4 = 64 bytes";
    }

    // Test single byte transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;
        wl.dst_width = 1;
        wl.num_dim = 0;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 1) << "Single byte is always contiguous";
    }

    // Test all dimensions contiguous (6D)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 4;
        wl.dst_width = 4;
        wl.num_dim = 5;

        int accumulated_stride = 4;
        for (int i = 0; i < 5; i++) {
            wl.e_dim[i].src_dim_size = 1;  // 2 elements each
            wl.e_dim[i].dst_dim_size = 1;
            wl.e_dim[i].src_stride = accumulated_stride;  // Properly aligned for contiguous
            wl.e_dim[i].dst_stride = accumulated_stride;
            accumulated_stride *= 2;
        }

        EXPECT_EQ(wl.getContiguousBytesSrc(), 4 * 2 * 2 * 2 * 2 * 2) << "6D fully contiguous: 128 bytes";
    }

    // Test stride of zero (edge case - no gap)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 100;
        wl.dst_width = 100;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 0;  // 1 element
        wl.e_dim[0].dst_dim_size = 0;
        wl.e_dim[0].src_stride = 0;  // Zero stride - not contiguous
        wl.e_dim[0].dst_stride = 0;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 100) << "Zero stride: only src_width is contiguous";
    }

    // Test negative stride (valid for backward memory access)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = -64;  // Negative stride - not contiguous in forward direction
        wl.e_dim[0].dst_stride = -64;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 64) << "Negative stride: only src_width is contiguous";
    }

    // Test perfect tiling pattern (common in image processing)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 256;  // Row width
        wl.dst_width = 256;
        wl.num_dim = 2;

        // Height: 16 rows
        wl.e_dim[0].src_dim_size = 15;
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 256;  // Contiguous rows
        wl.e_dim[0].dst_stride = 256;

        // Depth: 3 planes
        wl.e_dim[1].src_dim_size = 2;
        wl.e_dim[1].dst_dim_size = 2;
        wl.e_dim[1].src_stride = 4096;  // Contiguous: 256 * 16
        wl.e_dim[1].dst_stride = 4096;

        EXPECT_EQ(wl.getContiguousBytesSrc(), 256 * 16 * 3) << "Tiled pattern fully contiguous: 12288 bytes";
    }
}

/// Test cases covering the getContiguousBytesDst for DMANNWorkload_NPU40_50 class
TEST_F(TestDMATypes, getContiguousBytesDst_test) {
    // Test 1D transfer (num_dim = 0) - always contiguous
    {
        DMANNWorkload_NPU40_50 wl{device, 256, 256, 0};
        EXPECT_EQ(wl.getContiguousBytesDst(), 256) << "1D transfer: all bytes are contiguous";
    }

    // Test 2D transfer with contiguous memory (stride equals width)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 64;  // Contiguous: stride = width

        EXPECT_EQ(wl.getContiguousBytesDst(), 64 * 16) << "2D contiguous: 64 * 16 = 1024 bytes";
    }

    // Test 2D transfer with gap (stride > width) - not contiguous
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 128;  // Gap: stride > width, only 64 bytes contiguous per row

        EXPECT_EQ(wl.getContiguousBytesDst(), 64) << "2D with gap: only first dimension is contiguous (64 bytes)";
    }

    // Test 3D fully contiguous transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: 8 elements, contiguous
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;
        wl.e_dim[0].dst_stride = 32;  // Contiguous

        // Second dimension: 4 elements, contiguous
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;
        wl.e_dim[1].dst_stride = 256;  // Contiguous: 32 * 8

        EXPECT_EQ(wl.getContiguousBytesDst(), 32 * 8 * 4) << "3D fully contiguous: 32 * 8 * 4 = 1024 bytes";
    }

    // Test 3D partially contiguous (gap in second dimension)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: 8 elements, contiguous
        wl.e_dim[0].src_dim_size = 7;
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 32;  // Contiguous

        // Second dimension: has gap
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;
        wl.e_dim[1].dst_stride = 512;  // Gap: should be 256 for contiguous

        EXPECT_EQ(wl.getContiguousBytesDst(), 32 * 8) << "3D partial: contiguous up to first dim = 256 bytes";
    }

    // Test 4D with gap in middle dimension
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 16;
        wl.dst_width = 16;
        wl.num_dim = 3;

        wl.e_dim[0].src_dim_size = 3;  // 4 elements
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 16;  // Contiguous
        wl.e_dim[0].dst_stride = 16;  // Contiguous

        wl.e_dim[1].src_dim_size = 1;  // 2 elements, but has gap
        wl.e_dim[1].dst_dim_size = 1;
        wl.e_dim[1].src_stride = 64;
        wl.e_dim[1].dst_stride = 128;  // Gap: should be 64 for contiguous

        wl.e_dim[2].src_dim_size = 2;  // 3 elements
        wl.e_dim[2].dst_dim_size = 2;
        wl.e_dim[2].src_stride = 128;
        wl.e_dim[2].dst_stride = 256;

        EXPECT_EQ(wl.getContiguousBytesDst(), 16 * 4) << "4D with gap in dim 1: contiguous = 16 * 4 = 64 bytes";
    }

    // Test single byte transfer
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;
        wl.dst_width = 1;
        wl.num_dim = 0;

        EXPECT_EQ(wl.getContiguousBytesDst(), 1) << "Single byte is always contiguous";
    }

    // Test all dimensions contiguous (6D)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 4;
        wl.dst_width = 4;
        wl.num_dim = 5;

        int src_accumulated_stride = 4;
        int dst_accumulated_stride = 4;
        for (int i = 0; i < 5; i++) {
            wl.e_dim[i].src_dim_size = 1;  // 2 elements each
            wl.e_dim[i].dst_dim_size = 1;
            wl.e_dim[i].src_stride = src_accumulated_stride;
            wl.e_dim[i].dst_stride = dst_accumulated_stride;  // Properly aligned for contiguous
            src_accumulated_stride *= 2;
            dst_accumulated_stride *= 2;
        }

        EXPECT_EQ(wl.getContiguousBytesDst(), 4 * 2 * 2 * 2 * 2 * 2) << "6D fully contiguous: 128 bytes";
    }

    // Test stride of zero (edge case - no gap)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 100;
        wl.dst_width = 100;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 0;  // 1 element
        wl.e_dim[0].dst_dim_size = 0;
        wl.e_dim[0].src_stride = 0;
        wl.e_dim[0].dst_stride = 0;  // Zero stride - not contiguous

        EXPECT_EQ(wl.getContiguousBytesDst(), 100) << "Zero stride: only dst_width is contiguous";
    }

    // Test negative stride (valid for backward memory access)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = -64;  // Negative stride - not contiguous in forward direction

        EXPECT_EQ(wl.getContiguousBytesDst(), 64) << "Negative stride: only dst_width is contiguous";
    }

    // Test perfect tiling pattern (common in image processing)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 256;  // Row width
        wl.dst_width = 256;
        wl.num_dim = 2;

        // Height: 16 rows
        wl.e_dim[0].src_dim_size = 15;
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 256;
        wl.e_dim[0].dst_stride = 256;  // Contiguous rows

        // Depth: 3 planes
        wl.e_dim[1].src_dim_size = 2;
        wl.e_dim[1].dst_dim_size = 2;
        wl.e_dim[1].src_stride = 4096;
        wl.e_dim[1].dst_stride = 4096;  // Contiguous: 256 * 16

        EXPECT_EQ(wl.getContiguousBytesDst(), 256 * 16 * 3) << "Tiled pattern fully contiguous: 12288 bytes";
    }

    // Test different src/dst widths
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 128;  // Different from src
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;
        wl.e_dim[0].dst_stride = 128;  // Contiguous based on dst_width

        EXPECT_EQ(wl.getContiguousBytesDst(), 128 * 8) << "Different widths: contiguous = 128 * 8 = 1024 bytes";
    }

    // Test asymmetric dim sizes (different src and dst dimension sizes)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        // First dimension: different sizes
        wl.e_dim[0].src_dim_size = 7;   // 8 elements src
        wl.e_dim[0].dst_dim_size = 15;  // 16 elements dst
        wl.e_dim[0].src_stride = 32;
        wl.e_dim[0].dst_stride = 32;  // Contiguous

        // Second dimension
        wl.e_dim[1].src_dim_size = 3;
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;
        wl.e_dim[1].dst_stride = 512;  // Contiguous: 32 * 16

        EXPECT_EQ(wl.getContiguousBytesDst(), 32 * 16 * 4)
                << "Asymmetric dims: dst contiguous = 32 * 16 * 4 = 2048 bytes";
    }

    // Test src contiguous but dst not contiguous
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;  // Contiguous for src
        wl.e_dim[0].dst_stride = 96;  // Gap for dst

        EXPECT_EQ(wl.getContiguousBytesSrc(), 64 * 8) << "Src contiguous: 512 bytes";
        EXPECT_EQ(wl.getContiguousBytesDst(), 64) << "Dst not contiguous: only 64 bytes";
    }
}

/// Test cases covering the getNumContiguousChunksSrc and getNumContiguousChunksDst methods
TEST_F(TestDMATypes, getChunksCount_test) {
    // Test 1D transfer - single contiguous chunk
    {
        DMANNWorkload_NPU40_50 wl{device, 256, 256, 0};
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "1D transfer: single chunk for source";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "1D transfer: single chunk for destination";
    }

    // Test 2D fully contiguous - single chunk
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 elements
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 64;  // Contiguous
        wl.e_dim[0].dst_stride = 64;  // Contiguous

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 16) << "Total bytes: 1024";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 64 * 16) << "Fully contiguous source";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "2D fully contiguous: 1 chunk";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "2D fully contiguous: 1 chunk";
    }

    // Test 2D with gap - multiple chunks (one per row)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;
        wl.e_dim[0].src_dim_size = 15;  // 16 rows
        wl.e_dim[0].dst_dim_size = 15;
        wl.e_dim[0].src_stride = 128;  // Gap: stride > width
        wl.e_dim[0].dst_stride = 128;

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 16) << "Total bytes: 1024";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 64) << "Only one row is contiguous";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 16) << "2D with gap: 16 chunks (one per row)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 16) << "2D with gap: 16 chunks (one per row)";
    }

    // Test 3D fully contiguous - single chunk
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 32;

        wl.e_dim[1].src_dim_size = 3;  // 4 elements
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;  // Contiguous: 32 * 8
        wl.e_dim[1].dst_stride = 256;

        EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 4) << "Total: 1024 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 32 * 8 * 4) << "Fully contiguous";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "3D fully contiguous: 1 chunk";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "3D fully contiguous: 1 chunk";
    }

    // Test 3D partially contiguous (gap in second dimension)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 32;  // Contiguous
        wl.e_dim[0].dst_stride = 32;  // Contiguous

        wl.e_dim[1].src_dim_size = 3;  // 4 planes
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 512;  // Gap: should be 256 for contiguous
        wl.e_dim[1].dst_stride = 512;

        EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 4) << "Total: 1024 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 32 * 8) << "Contiguous per plane: 256 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 4) << "3D partial: 4 chunks (one per plane)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 4) << "3D partial: 4 chunks (one per plane)";
    }

    // Test asymmetric chunks (different src and dst chunk counts)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 7;  // 8 elements
        wl.e_dim[0].dst_dim_size = 7;
        wl.e_dim[0].src_stride = 64;  // Contiguous for src
        wl.e_dim[0].dst_stride = 96;  // Gap for dst

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 8) << "Total accessed: 512 bytes";
        EXPECT_EQ(wl.getWrittenBytes(), 64 * 8) << "Total written: 512 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 64 * 8) << "Src fully contiguous";
        EXPECT_EQ(wl.getContiguousBytesDst(), 64) << "Dst has gaps";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Src: 1 chunk (fully contiguous)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 8) << "Dst: 8 chunks (one per element)";
    }

    // Test complex 4D scenario
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 16;
        wl.dst_width = 16;
        wl.num_dim = 3;

        wl.e_dim[0].src_dim_size = 3;  // 4 elements
        wl.e_dim[0].dst_dim_size = 3;
        wl.e_dim[0].src_stride = 16;  // Contiguous
        wl.e_dim[0].dst_stride = 16;

        wl.e_dim[1].src_dim_size = 1;  // 2 elements
        wl.e_dim[1].dst_dim_size = 1;
        wl.e_dim[1].src_stride = 128;  // Gap: should be 64
        wl.e_dim[1].dst_stride = 128;

        wl.e_dim[2].src_dim_size = 2;  // 3 elements
        wl.e_dim[2].dst_dim_size = 2;
        wl.e_dim[2].src_stride = 256;  // Contiguous based on previous gap
        wl.e_dim[2].dst_stride = 256;

        EXPECT_EQ(wl.getAccessedBytes(), 16 * 4 * 2 * 3) << "Total: 384 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 16 * 4) << "Contiguous up to first dim: 64 bytes";
        // 384 / 64 = 6 chunks
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 6) << "4D: 6 chunks (2*3 from remaining dims)";
    }

    // Test with different src and dst dimension sizes
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;
        wl.dst_width = 32;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 7;   // 8 elements src
        wl.e_dim[0].dst_dim_size = 15;  // 16 elements dst
        wl.e_dim[0].src_stride = 32;
        wl.e_dim[0].dst_stride = 32;

        wl.e_dim[1].src_dim_size = 3;  // 4 planes
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 256;  // Contiguous: 32 * 8
        wl.e_dim[1].dst_stride = 512;  // Contiguous: 32 * 16

        EXPECT_EQ(wl.getAccessedBytes(), 32 * 8 * 4) << "Src total: 1024 bytes";
        EXPECT_EQ(wl.getWrittenBytes(), 32 * 16 * 4) << "Dst total: 2048 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 32 * 8 * 4) << "Src fully contiguous";
        EXPECT_EQ(wl.getContiguousBytesDst(), 32 * 16 * 4) << "Dst fully contiguous";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Src: 1 chunk";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Dst: 1 chunk";
    }

    // Test large transfer with many chunks
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;
        wl.dst_width = 64;
        wl.num_dim = 2;

        wl.e_dim[0].src_dim_size = 63;  // 64 rows
        wl.e_dim[0].dst_dim_size = 63;
        wl.e_dim[0].src_stride = 96;  // Gap
        wl.e_dim[0].dst_stride = 96;

        wl.e_dim[1].src_dim_size = 3;  // 4 planes
        wl.e_dim[1].dst_dim_size = 3;
        wl.e_dim[1].src_stride = 6144;  // Gap: 96 * 64
        wl.e_dim[1].dst_stride = 6144;

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 64 * 4) << "Total: 16384 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 64) << "Only one row contiguous: 64 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 256) << "256 chunks (64 rows * 4 planes)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 256) << "256 chunks (64 rows * 4 planes)";
    }

    // Test edge case: zero-sized contiguous block should be handled
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 0;
        wl.dst_width = 0;
        wl.num_dim = 0;

        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 0) << "Zero width: 0 chunks";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 0) << "Zero width: 0 chunks";
    }

    // Test single element repeated many times with gaps
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 1;
        wl.dst_width = 1;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 99;  // 100 elements
        wl.e_dim[0].dst_dim_size = 99;
        wl.e_dim[0].src_stride = 4;  // Gap: 3 bytes between elements
        wl.e_dim[0].dst_stride = 4;

        EXPECT_EQ(wl.getAccessedBytes(), 100) << "Total: 100 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 1) << "Single byte per chunk";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 100) << "100 chunks (one per byte)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 100) << "100 chunks (one per byte)";
    }

    // Test single contiguous chunk smaller than 256 bytes threshold (relevant for stride penalty)
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 128;  // Less than 256 bytes
        wl.dst_width = 128;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 1;  // 2 elements total
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 128;  // Contiguous
        wl.e_dim[0].dst_stride = 128;

        EXPECT_EQ(wl.getAccessedBytes(), 128 * 2) << "Total: 256 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 128 * 2) << "Fully contiguous: 256 bytes";
        EXPECT_EQ(wl.getContiguousBytesDst(), 128 * 2) << "Fully contiguous: 256 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Single chunk of 256 bytes (at threshold)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Single chunk of 256 bytes (at threshold)";
    }

    // Test single contiguous chunk well below 256 bytes threshold
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 64;  // Well below 256 bytes
        wl.dst_width = 64;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 1;  // 2 elements total
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 64;  // Contiguous
        wl.e_dim[0].dst_stride = 64;

        EXPECT_EQ(wl.getAccessedBytes(), 64 * 2) << "Total: 128 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 64 * 2) << "Fully contiguous: 128 bytes";
        EXPECT_EQ(wl.getContiguousBytesDst(), 64 * 2) << "Fully contiguous: 128 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Single chunk of 128 bytes (below 256 threshold)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Single chunk of 128 bytes (below 256 threshold)";
    }

    // Test single contiguous chunk slightly above 256 bytes threshold
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 200;
        wl.dst_width = 200;
        wl.num_dim = 1;

        wl.e_dim[0].src_dim_size = 1;  // 2 elements total
        wl.e_dim[0].dst_dim_size = 1;
        wl.e_dim[0].src_stride = 200;  // Contiguous
        wl.e_dim[0].dst_stride = 200;

        EXPECT_EQ(wl.getAccessedBytes(), 200 * 2) << "Total: 400 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 200 * 2) << "Fully contiguous: 400 bytes";
        EXPECT_EQ(wl.getContiguousBytesDst(), 200 * 2) << "Fully contiguous: 400 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Single chunk of 400 bytes (above 256 threshold)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Single chunk of 400 bytes (above 256 threshold)";
    }

    // Test very small single contiguous chunk
    {
        DMANNWorkload_NPU40_50 wl{device};
        wl.src_width = 32;  // Very small
        wl.dst_width = 32;
        wl.num_dim = 0;  // 1D

        EXPECT_EQ(wl.getAccessedBytes(), 32) << "Total: 32 bytes";
        EXPECT_EQ(wl.getContiguousBytesSrc(), 32) << "Fully contiguous: 32 bytes";
        EXPECT_EQ(wl.getContiguousBytesDst(), 32) << "Fully contiguous: 32 bytes";
        EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "Single very small chunk (32 bytes, well below 256)";
        EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Single very small chunk (32 bytes, well below 256)";
    }
}

/// Regression test: a degenerate dimension (dim_size==0, meaning only 1 element) must NOT
/// break the contiguity chain.  A single-element dimension has no "next element" to reach
/// via the stride, so its stride value is physically irrelevant and should be ignored when
/// deciding whether contiguous bytes continue into the next dimension.
TEST_F(TestDMATypes, getContiguousBytesSrc_degenerate_dim_size_zero_bug) {
    // Workload layout:
    //   src_width = 64
    //   dim[0]:  stride=999 (arbitrary), dim_size=0  =>  1 element, stride never used
    //   dim[1]:  stride=64,              dim_size=7  =>  8 elements, stride == accumulated 64
    //
    // Correct contiguous bytes: 64 * 1 * 8 = 512
    // Buggy   contiguous bytes: 64           (loop breaks at dim[0] because 999 != 64)
    DMANNWorkload_NPU40_50 wl{device};
    wl.src_width = 64;
    wl.dst_width = 64;
    wl.num_dim = 2;

    wl.e_dim[0].src_stride = 999;  // arbitrary — must be ignored because dim_size==0
    wl.e_dim[0].dst_stride = 999;
    wl.e_dim[0].src_dim_size = 0;  // 1 element: stride is irrelevant
    wl.e_dim[0].dst_dim_size = 0;

    wl.e_dim[1].src_stride = 64;  // contiguous: equals the accumulated contiguous_bytes
    wl.e_dim[1].dst_stride = 64;
    wl.e_dim[1].src_dim_size = 7;  // 8 elements => 64 * 8 = 512 B
    wl.e_dim[1].dst_dim_size = 7;

    // Logical bytes: 64 * (0+1) * (7+1) = 512
    EXPECT_EQ(wl.getAccessedBytes(), 512);
    EXPECT_EQ(wl.getWrittenBytes(), 512);

    // BUG: currently returns 64 instead of 512
    EXPECT_EQ(wl.getContiguousBytesSrc(), 512)
            << "BUG: degenerate dim[0] (size=0, stride=999) must not break contiguity; "
               "expected 512 contiguous bytes";
    EXPECT_EQ(wl.getContiguousBytesDst(), 512) << "BUG: same issue on dst side";

    // Downstream consequence: chunk count is inflated from 1 to 8
    EXPECT_EQ(wl.getNumContiguousChunksSrc(), 1) << "BUG: should be 1 fully-contiguous chunk";
    EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "BUG: should be 1 fully-contiguous chunk";
}
/// Test the broadcast / alias DMA pattern where src stride=0 causes all rows to map to the
/// same 10 physical bytes, while the destination is packed contiguously.
///
/// Layout:
///   src_width = dst_width = 10
///   dim[0]: 256 rows (dst_dim_size = 255), src_stride = 0 (broadcast), dst_stride = 10 (packed)
///
/// Expected:
///   - Logical bytes read/written : 10 * 256 = 2560
///   - Contiguous src bytes       : 10   (stride 0 != 10, chain breaks immediately)
///   - Contiguous dst bytes       : 2560 (stride 10 == dst_width, fully packed)
TEST_F(TestDMATypes, getContiguousBytes_broadcast_src_stride_zero) {
    DMANNWorkload_NPU40_50 wl{device};
    wl.src_width = 10;
    wl.dst_width = 10;
    wl.num_dim = 1;

    wl.e_dim[0].src_dim_size = 255;  // 256 rows (0-based)
    wl.e_dim[0].src_stride = 0;      // stride 0: every row aliases the same 10 bytes in memory

    wl.e_dim[0].dst_dim_size = 255;  // 256 rows
    wl.e_dim[0].dst_stride = 10;     // packed/contiguous destination

    // Logical transfer sizes
    EXPECT_EQ(wl.getAccessedBytes(), 10 * 256) << "Src logical bytes: 10 * 256 = 2560";
    EXPECT_EQ(wl.getWrittenBytes(), 10 * 256) << "Dst logical bytes: 10 * 256 = 2560";

    // Source: stride 0 != contiguous_bytes (10), chain breaks at dim[0] -> only 10 bytes contiguous
    EXPECT_EQ(wl.getContiguousBytesSrc(), 10)
            << "Src broadcast: stride=0 breaks contiguity, only innermost 10 bytes are contiguous";

    // Destination: stride 10 == dst_width -> all 10 * 256 = 2560 bytes are contiguous
    EXPECT_EQ(wl.getContiguousBytesDst(), 10 * 256)
            << "Dst packed: stride equals dst_width, all 2560 bytes are contiguous";

    // Chunk counts follow from the contiguous sizes
    EXPECT_EQ(wl.getNumContiguousChunksSrc(), 256) << "Src: 256 chunks of 10 bytes each (one per broadcast row)";
    EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Dst: 1 fully-contiguous chunk";
}

/// Extension of the broadcast test: adds a 3rd dimension (vertical slices) on the source,
/// with stride=20 between slices (10-byte payload + 10-byte gap) and 5 slices (dim_size=4).
/// The src stride=0 on the row dimension still collapses contiguity to 10 bytes while the fully-packed dst remains
/// 12800 bytes contiguous.
///
/// Layout:
///   src_width = dst_width = 10
///   dim[0]: 256 rows,  src_stride=0  (broadcast – same 10 bytes repeated), dst_stride=10  (packed)
///   dim[1]: 5  slices, src_stride=20 (10-byte payload + 10-byte gap per slice), dst_stride=2560 (packed: 10*256)
///
/// Physical src memory: 5 slices × 10 bytes each, spaced 20 bytes apart → 90 bytes footprint, 50 bytes payload
/// Logical src bytes  : 10 * 256 * 5 = 12800  (getAccessedBytes counts repetitions, not physical footprint)
/// Logical dst bytes  : 10 * 256 * 5 = 12800
///
/// Expected:
///   - getContiguousBytesSrc = 10   (stride-0 in dim[0] breaks the chain at the first extra dim)
///   - getContiguousBytesDst = 12800 (all three dims packed end-to-end)
TEST_F(TestDMATypes, getContiguousBytes_broadcast_src_stride_zero_3rd_dim) {
    DMANNWorkload_NPU40_50 wl{device};
    wl.src_width = 10;
    wl.dst_width = 10;
    wl.num_dim = 2;  // two extra dims: rows (dim[0]) + slices (dim[1])

    // dim[0]: 256 rows – src broadcasts the same 10 bytes; dst is packed
    wl.e_dim[0].src_dim_size = 255;  // 256 rows (0-based)
    wl.e_dim[0].src_stride = 0;      // broadcast: every row reads the same 10 bytes
    wl.e_dim[0].dst_dim_size = 255;  // 256 rows
    wl.e_dim[0].dst_stride = 10;     // packed rows on dst

    // dim[1]: 5 slices – src has a 10-byte gap between 10-byte payloads (stride=20)
    wl.e_dim[1].src_dim_size = 4;       // 5 slices (0-based)
    wl.e_dim[1].src_stride = 20;        // 10-byte payload + 10-byte gap → not contiguous with 10
    wl.e_dim[1].dst_dim_size = 4;       // 5 slices
    wl.e_dim[1].dst_stride = 10 * 256;  // packed slices on dst: each slice is 256 rows × 10 bytes = 2560

    // Logical transfer sizes (strides are irrelevant for byte counts)
    EXPECT_EQ(wl.getAccessedBytes(), 10 * 256 * 5) << "Src logical bytes: 10 * 256 * 5 = 12800";
    EXPECT_EQ(wl.getWrittenBytes(), 10 * 256 * 5) << "Dst logical bytes: 10 * 256 * 5 = 12800";

    // Source contiguity: stride-0 in dim[0] (0 != 10) breaks the chain immediately
    EXPECT_EQ(wl.getContiguousBytesSrc(), 10)
            << "Src: stride=0 in dim[0] breaks contiguity; only the innermost 10 bytes are contiguous";

    // Destination contiguity: stride=10 in dim[0] and stride=2560 in dim[1] are both packed
    EXPECT_EQ(wl.getContiguousBytesDst(), 10 * 256 * 5)
            << "Dst: all dims packed, 10 * 256 * 5 = 12800 bytes fully contiguous";

    // Chunk counts
    EXPECT_EQ(wl.getNumContiguousChunksSrc(), 256 * 5)
            << "Src: 256*5 = 1280 chunks of 10 bytes (one per broadcast row per slice)";
    EXPECT_EQ(wl.getNumContiguousChunksDst(), 1) << "Dst: single fully-contiguous chunk";
}

}  // namespace VPUNN_unit_tests
