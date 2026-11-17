// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_A3_TILING_DATA_H
#define INDEXER_REGION_SCORE_A3_TILING_DATA_H
#include "../indexer_region_score_paged_tiling_data.h"
namespace IndexerRegionScoreA3 {
using IndexerRegionScorePaged::PagedTilingData;
constexpr uint32_t kDimension = 256;
constexpr uint32_t kHeadCapacity = 128;
constexpr uint32_t kRegionTile = 128;
constexpr uint32_t kVectorColumns = 64;
constexpr uint32_t kMaxTopK = 512;
// reservedStride0 is the byte offset of this per-pair bridge from
// GetUserWorkspace(workspace). Two FP32 H=128, N=128 slots are reserved.
// The first implementation uses one synchronous slot; the second is reserved
// for compatible pipelining without changing the host/device tiling ABI.
constexpr uint64_t kBridgeBytesPerCore = 2ULL * kHeadCapacity * kRegionTile * sizeof(float);
} // namespace IndexerRegionScoreA3
#endif
