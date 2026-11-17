// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_A3_H
#define INDEXER_REGION_SCORE_A3_H

#include "kernel_operator.h"
#include "a3_tiling_data.h"

namespace IndexerRegionScoreA3 {
using namespace AscendC;

// A query bundle is owned by one Cube/AIV pair. Only a bounded H-by-128 dot tile
// crosses GM. Scores retain the source's quartet reduction order; no floating
// atomics or cross-core floating reductions are used.
class A3IndexerRegionScore {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR weights, GM_ADDR key,
        GM_ADDR lengths, GM_ADDR table, GM_ADDR cu, GM_ADDR output, GM_ADDR ids,
        GM_ADDR userWorkspace, const PagedTilingData &data, TPipe *pipe)
    {
        data_ = data;
        pipe_ = pipe;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        weightsFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        weightsBits_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(weights));
        weightsHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cu));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        if (data_.batchTokens != 0) {
            scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + data_.scorePoolOffset));
        }
        core_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            core_ /= 2;
            sub_ = GetSubBlockIdx();
        }
        bridge_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            userWorkspace + data_.reservedStride0 + core_ * kBridgeBytesPerCore));
        if ASCEND_IS_AIC {
            InitCube();
        } else {
            InitVector();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            CubeRows();
        } else {
            VectorRows();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            SelectRows();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t kReady = 0;
    static constexpr uint16_t kFree = 1;
    static constexpr uint32_t kSelectionWindow = 4096;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align16(uint32_t n) const { return (n + 15U) / 16U * 16U; }

    __aicore__ inline bool Request(uint32_t token, uint32_t &request, uint32_t &visible) const
    {
        visible = 0;
        for (uint32_t n = 0; n < data_.numRequests; ++n) {
            const int32_t begin = cu_.GetValue(n), end = cu_.GetValue(n + 1);
            if (begin < 0 || end < begin || static_cast<uint32_t>(end) > data_.totalTokens) continue;
            if (token >= static_cast<uint32_t>(begin) && token < static_cast<uint32_t>(end)) {
                request = n;
                const int64_t position = static_cast<int64_t>(lengths_.GetValue(n)) - (end - begin) + token - begin;
                if (position > 0) visible = Min(static_cast<uint32_t>(position / data_.regionSize), data_.seqK);
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint64_t ScoreBase(uint32_t group, uint32_t token) const
    {
        const uint32_t row = data_.batchTokens == 0 ? token : token - data_.batchBegin;
        return static_cast<uint64_t>(group) * data_.outputStrideGroup +
            static_cast<uint64_t>(row) * data_.outputStrideToken;
    }

    __aicore__ inline float Weight(uint32_t group, uint32_t token, uint32_t head) const
    {
        const uint64_t offset = static_cast<uint64_t>(token) * data_.weightStrideToken +
            static_cast<uint64_t>(group) * data_.weightStrideGroup + head;
        float value;
        if (data_.weightDtype == 0) {
            value = weightsFloat_.GetValue(offset);
        } else if (data_.weightDtype == 1) {
            union { uint32_t bits; float number; } converted;
            converted.bits = static_cast<uint32_t>(weightsBits_.GetValue(offset)) << 16U;
            value = converted.number;
        } else {
            value = static_cast<float>(weightsHalf_.GetValue(offset));
        }
        return data_.headScale == 1.0F ? value : value * data_.headScale;
    }

    __aicore__ inline void InitCube()
    {
        pipe_->InitBuffer(qL1Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL1Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(qL0Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL0Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(cL0Buffer_, kHeadCapacity * kRegionTile * sizeof(float));
        qL1_ = qL1Buffer_.Get<bfloat16_t>();
        kL1_ = kL1Buffer_.Get<bfloat16_t>();
        qL0_ = qL0Buffer_.Get<bfloat16_t>();
        kL0_ = kL0Buffer_.Get<bfloat16_t>();
        cL0_ = cL0Buffer_.Get<float>();
        SetFlag<HardEvent::M_MTE2>(0);
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void InitVector()
    {
        pipe_->InitBuffer(dotBuffer_, kHeadCapacity * kVectorColumns * sizeof(float));
        pipe_->InitBuffer(accBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(partBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(weightBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(windowBuffer_, kSelectionWindow * sizeof(float));
        pipe_->InitBuffer(heapScoreBuffer_, kMaxTopK * sizeof(float));
        pipe_->InitBuffer(heapIdBuffer_, kMaxTopK * sizeof(int32_t));
        dots_ = dotBuffer_.Get<float>();
        acc_ = accBuffer_.Get<float>();
        partial_ = partBuffer_.Get<float>();
        weight_ = weightBuffer_.Get<float>();
        window_ = windowBuffer_.Get<float>();
        heapScores_ = heapScoreBuffer_.Get<float>();
        heapIds_ = heapIdBuffer_.Get<int32_t>();
    }

    __aicore__ inline bool ContiguousKey(uint32_t request, uint32_t visible)
    {
        if (data_.summariesPerPage != 1 || data_.cacheStrideRegion != kDimension ||
            data_.cacheStridePage != kDimension) return false;
        if (cachedRequest_ != request) {
            cachedRequest_ = request;
            cachedPrefix_ = 0;
            cachedLinear_ = true;
            cachedFirstPage_ = 0;
        }
        if (!cachedLinear_) return false;
        while (cachedPrefix_ < visible) {
            const int32_t page = table_.GetValue(static_cast<uint64_t>(request) *
                data_.blockTableStrideRequest + cachedPrefix_);
            if (cachedPrefix_ == 0) cachedFirstPage_ = page;
            if (page < 0 || static_cast<uint32_t>(page) >= data_.numPhysicalPages ||
                static_cast<int64_t>(page) != static_cast<int64_t>(cachedFirstPage_) + cachedPrefix_) {
                cachedLinear_ = false;
                return false;
            }
            ++cachedPrefix_;
        }
        return true;
    }

    __aicore__ inline void LoadCubeTile(uint32_t group, uint32_t token,
        uint32_t request, uint32_t regionStart, uint32_t count, uint32_t queries, bool linear)
    {
        WaitFlag<HardEvent::M_MTE2>(0);
        const uint32_t m = Align16(data_.headsPerGroup * queries);
        InitConstValueParams<uint16_t> clearQ(1, static_cast<uint16_t>(m * kDimension / 16), 0, 0);
        InitConstValueParams<uint16_t> clearK(1, static_cast<uint16_t>(kRegionTile * kDimension / 16), 0, 0);
        InitConstValue(qL1_.ReinterpretCast<uint16_t>(), clearQ);
        InitConstValue(kL1_.ReinterpretCast<uint16_t>(), clearK);
        PipeBarrier<PIPE_MTE2>();
        Nd2NzParams qCopy{};
        qCopy.ndNum = 1;
        qCopy.nValue = data_.headsPerGroup;
        qCopy.dValue = kDimension;
        qCopy.srcDValue = kDimension;
        qCopy.dstNzC0Stride = m;
        qCopy.dstNzNStride = 1;
        for (uint32_t q = 0; q < queries; ++q) {
            DataCopy(qL1_[q * data_.headsPerGroup * 16], query_[
                static_cast<uint64_t>(token + q) * data_.qStrideToken +
                static_cast<uint64_t>(group) * data_.qStrideGroup], qCopy);
        }
        // Key rows are paged. Adjacent physical rows share one ND-to-NZ copy.
        Nd2NzParams keyCopy{};
        keyCopy.ndNum = 1;
        keyCopy.nValue = 1;
        keyCopy.dValue = kDimension;
        keyCopy.srcDValue = kDimension;
        keyCopy.dstNzC0Stride = kRegionTile;
        keyCopy.dstNzNStride = 1;
        if (linear) {
            keyCopy.nValue = count;
            DataCopy(kL1_, key_[(static_cast<uint64_t>(cachedFirstPage_) + regionStart) * kDimension], keyCopy);
        }
        for (uint32_t r = linear ? count : 0; r < count;) {
            const uint32_t logical = regionStart + r;
            const int32_t page = table_.GetValue(static_cast<uint64_t>(request) *
                data_.blockTableStrideRequest + logical / data_.summariesPerPage);
            if (page < 0 || static_cast<uint32_t>(page) >= data_.numPhysicalPages) {
                ++r;
                continue;
            }
            const uint64_t offset = static_cast<uint64_t>(page) * data_.cacheStridePage +
                (logical % data_.summariesPerPage) * data_.cacheStrideRegion;
            uint32_t run = 1;
            // Coalesce only rows whose physical offsets are exactly adjacent.
            // Invalid/strided/shuffled pages retain their zero/small-copy path.
            if (data_.cacheStrideRegion == kDimension &&
                data_.cacheStridePage == data_.summariesPerPage * kDimension) {
                while (r + run < count) {
                    const uint32_t next = logical + run;
                    const int32_t nextPage = table_.GetValue(static_cast<uint64_t>(request) *
                        data_.blockTableStrideRequest + next / data_.summariesPerPage);
                    if (nextPage < 0 || static_cast<uint32_t>(nextPage) >= data_.numPhysicalPages) break;
                    const uint64_t nextOffset = static_cast<uint64_t>(nextPage) * data_.cacheStridePage +
                        (next % data_.summariesPerPage) * data_.cacheStrideRegion;
                    if (nextOffset != offset + static_cast<uint64_t>(run) * kDimension) break;
                    ++run;
                }
            }
            keyCopy.nValue = run;
            DataCopy(kL1_[r * 16], key_[offset], keyCopy);
            r += run;
        }
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2DParams loadQ{};
        loadQ.repeatTimes = kDimension / 16;
        loadQ.srcStride = m / 16;
        loadQ.ifTranspose = false;
        for (uint32_t block = 0; block < m / 16; ++block) {
            LoadData(qL0_[block * kDimension * 16], qL1_[block * 256], loadQ);
        }
        LoadData2DParams loadK{};
        loadK.repeatTimes = kRegionTile * kDimension / 256;
        loadK.srcStride = 1;
        loadK.ifTranspose = false;
        LoadData(kL0_, kL1_, loadK);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
    }

    // Fill the otherwise padded M=16 rows for H=4. A bundle is split at
    // request boundaries; each segment shares its request's paged key.
    __aicore__ inline uint32_t BundleWidth() const
    {
        return data_.headsPerGroup == 4 ? 4 : 1;
    }

    __aicore__ inline uint32_t Segment(uint32_t token, uint32_t limit,
        uint32_t request, uint32_t &maxVisible) const
    {
        uint32_t queries = 1;
        while (queries < limit) {
            uint32_t nextRequest = 0, nextVisible = 0;
            if (!Request(token + queries, nextRequest, nextVisible) || nextRequest != request) break;
            if (nextVisible > maxVisible) maxVisible = nextVisible;
            ++queries;
        }
        return queries;
    }

    __aicore__ inline void CubeRows()
    {
        const uint32_t width = BundleWidth();
        const uint32_t bundles = (data_.batchCount + width - 1) / width;
        const uint32_t rows = data_.groups * bundles;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / bundles;
            const uint32_t first = row % bundles * width;
            const uint32_t count = Min(width, data_.batchCount - first);
            for (uint32_t q = 0; q < count;) {
                const uint32_t token = data_.batchBegin + first + q;
                uint32_t request = 0, visible = 0;
                if (!Request(token, request, visible)) { ++q; continue; }
                const uint32_t queries = Segment(token, count - q, request, visible);
                const uint32_t m = Align16(data_.headsPerGroup * queries);
                const bool linear = ContiguousKey(request, visible);
                for (uint32_t r = 0; r < visible; r += kRegionTile) {
                    LoadCubeTile(group, token, request, r, Min(kRegionTile, visible - r), queries, linear);
                    WaitFlag<HardEvent::FIX_M>(0);
                    MmadParams mm{};
                    mm.m = m;
                    mm.n = kRegionTile;
                    mm.k = kDimension;
                    mm.cmatrixInitVal = true;
                    Mmad(cL0_, qL0_, kL0_, mm);
                    SetFlag<HardEvent::M_MTE2>(0);
                    SetFlag<HardEvent::M_FIX>(0);
                    WaitFlag<HardEvent::M_FIX>(0);
                    CrossCoreWaitFlag(kFree);
                    FixpipeParamsV220 fix{};
                    fix.mSize = m;
                    fix.nSize = kRegionTile;
                    fix.srcStride = m;
                    fix.dstStride = kRegionTile;
                    fix.quantPre = QuantMode_t::NoQuant;
                    fix.reluEn = true;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(bridge_, cL0_, fix);
                    CrossCoreSetFlag<2, PIPE_FIX>(kReady);
                    SetFlag<HardEvent::FIX_M>(0);
                }
                q += queries;
            }
        }
        CrossCoreWaitFlag(kFree);
        WaitFlag<HardEvent::M_MTE2>(0);
        WaitFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Reduce(uint32_t group, uint32_t token, uint32_t dotHead)
    {
        for (uint32_t h = 0; h < data_.headsPerGroup; h += 4) {
            auto partial = h == 0 ? acc_ : partial_;
            Muls(partial, dots_[(dotHead + h) * kVectorColumns], Weight(group, token, h), kVectorColumns);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1; j < 4; ++j) {
                Duplicate(weight_, Weight(group, token, h + j), kVectorColumns);
                PipeBarrier<PIPE_V>();
                MulAddDst(partial, dots_[(dotHead + h + j) * kVectorColumns], weight_, kVectorColumns);
                PipeBarrier<PIPE_V>();
            }
            if (h != 0) {
                Add(acc_, acc_, partial_, kVectorColumns);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void VectorRows()
    {
        CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
        const uint32_t width = BundleWidth();
        const uint32_t bundles = (data_.batchCount + width - 1) / width;
        const uint32_t rows = data_.groups * bundles;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / bundles;
            const uint32_t first = row % bundles * width;
            const uint32_t count = Min(width, data_.batchCount - first);
            for (uint32_t q = 0; q < count;) {
                const uint32_t token = data_.batchBegin + first + q;
                uint32_t request = 0, visible = 0;
                if (!Request(token, request, visible)) { ++q; continue; }
                const uint32_t queries = Segment(token, count - q, request, visible);
                for (uint32_t r = 0; r < visible; r += kRegionTile) {
                    CrossCoreWaitFlag(kReady);
                    DataCopyParams copy{};
                    copy.blockCount = data_.headsPerGroup * queries;
                    copy.blockLen = kVectorColumns * sizeof(float) / 32;
                    copy.srcStride = kVectorColumns * sizeof(float) / 32;
                    DataCopy(dots_, bridge_[sub_ * kVectorColumns], copy);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
                    for (uint32_t item = 0; item < queries; ++item) {
                        uint32_t itemRequest = 0, itemVisible = 0;
                        (void)Request(token + item, itemRequest, itemVisible);
                        Reduce(group, token + item, item * data_.headsPerGroup);
                        const uint32_t start = r + sub_ * kVectorColumns;
                        if (start < itemVisible) {
                            const uint32_t length = Min(kVectorColumns, itemVisible - start);
                            SetFlag<HardEvent::V_MTE3>(0);
                            WaitFlag<HardEvent::V_MTE3>(0);
                            DataCopyExtParams store{1, length * 4U, 0, 0, 0};
                            DataCopyPad(scores_[ScoreBase(group, token + item) + start], acc_, store);
                            SetFlag<HardEvent::MTE3_V>(0);
                            WaitFlag<HardEvent::MTE3_V>(0);
                        }
                    }
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }
                q += queries;
            }
        }
    }

    // Better means a higher score or the smaller logical ID at exact ties.
    __aicore__ inline bool Better(float a, int32_t ai, float b, int32_t bi) const
    {
        return a > b || (a == b && ai < bi);
    }

    __aicore__ inline bool HeapBetter(float a, int32_t ai, float b, int32_t bi) const
    {
        return reverseHeap_ ? Better(b, bi, a, ai) : Better(a, ai, b, bi);
    }

    __aicore__ inline void Swap(uint32_t a, uint32_t b)
    {
        const float score = heapScores_.GetValue(a);
        const int32_t id = heapIds_.GetValue(a);
        heapScores_.SetValue(a, heapScores_.GetValue(b));
        heapIds_.SetValue(a, heapIds_.GetValue(b));
        heapScores_.SetValue(b, score);
        heapIds_.SetValue(b, id);
    }

    __aicore__ inline void Push(float score, int32_t id, uint32_t &size)
    {
        uint32_t at = size++;
        heapScores_.SetValue(at, score);
        heapIds_.SetValue(at, id);
        while (at != 0) {
            const uint32_t parent = (at - 1) / 2;
            if (!HeapBetter(heapScores_.GetValue(parent), heapIds_.GetValue(parent), score, id)) break;
            Swap(parent, at);
            at = parent;
        }
    }

    __aicore__ inline void ReplaceRoot(float score, int32_t id, uint32_t size)
    {
        heapScores_.SetValue(0, score);
        heapIds_.SetValue(0, id);
        uint32_t at = 0;
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && HeapBetter(heapScores_.GetValue(child), heapIds_.GetValue(child),
                heapScores_.GetValue(child + 1), heapIds_.GetValue(child + 1))) ++child;
            if (!HeapBetter(heapScores_.GetValue(at), heapIds_.GetValue(at),
                heapScores_.GetValue(child), heapIds_.GetValue(child))) break;
            Swap(at, child);
            at = child;
        }
    }

    __aicore__ inline void SiftDownIds(uint32_t at, uint32_t size)
    {
        const int32_t id = heapIds_.GetValue(at);
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && heapIds_.GetValue(child + 1) > heapIds_.GetValue(child)) ++child;
            const int32_t childId = heapIds_.GetValue(child);
            if (id >= childId) break;
            heapIds_.SetValue(at, childId);
            at = child;
        }
        heapIds_.SetValue(at, id);
    }

    __aicore__ inline void SortIds(uint32_t size)
    {
        // Sort only integer IDs. A max heap yields ascending output with
        // bounded O(K log K) work, including large equal-score tie sets.
        for (uint32_t root = size / 2; root != 0; --root) SiftDownIds(root - 1, size);
        for (uint32_t end = size; end > 1; --end) {
            const int32_t largest = heapIds_.GetValue(0);
            heapIds_.SetValue(0, heapIds_.GetValue(end - 1));
            heapIds_.SetValue(end - 1, largest);
            SiftDownIds(0, end - 1);
        }
    }

    __aicore__ inline void OrderSelectedIds(uint32_t size, uint32_t visible, bool complement)
    {
        if (visible > kSelectionWindow) {
            SortIds(size);
            return;
        }
        // Scoring has completed and Process synchronized all cores. Reuse
        // the first 16 KiB of dots_ for integer membership flags.
        auto flags = dots_.ReinterpretCast<int32_t>();
        const uint32_t aligned = (visible + 7U) / 8U * 8U;
        Duplicate(flags, int32_t(0), aligned);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t i = 0; i < size; ++i) {
            flags.SetValue(static_cast<uint32_t>(heapIds_.GetValue(i)), 1);
        }
        uint32_t out = 0;
        for (uint32_t id = 0; id < visible; ++id) {
            if ((flags.GetValue(id) != 0) != complement) heapIds_.SetValue(out++, static_cast<int32_t>(id));
        }
    }

    __aicore__ inline uint32_t SelectByThreshold(uint32_t group, uint32_t token,
        uint32_t visible, uint32_t k)
    {
        // Selection runs after the global scoring barrier. Reuse 24 KiB of
        // the 32 KiB dot buffer; no additional UB or GM allocation is needed.
        const uint32_t aligned = (visible + 63U) / 64U * 64U;
        auto packed = dots_[0];
        auto scratch = dots_[2048];
        auto ordered = dots_[4096];
        auto index = dots_.ReinterpretCast<uint32_t>()[5120];
        DataCopyExtParams copy{1, visible * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(window_, scores_[ScoreBase(group, token)], copy, pad);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        union { uint32_t bits; float value; } negativeInfinity;
        negativeInfinity.bits = 0xFF800000U;
        for (uint32_t i = visible; i < aligned; ++i) window_.SetValue(i, negativeInfinity.value);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Duplicate(index, uint32_t(0), aligned);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(packed, window_, index, scratch, aligned / 32U);
        PipeBarrier<PIPE_V>();
        Extract(ordered, index, packed, aligned / 32U);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const float threshold = ordered.GetValue(k - 1);
        // Find the first tie by binary search; the hardware sort's tie order
        // is irrelevant because only the threshold value is used.
        uint32_t lo = 0, hi = k;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (ordered.GetValue(mid) > threshold) lo = mid + 1;
            else hi = mid;
        }
        const uint32_t ties = k - lo;
        auto logicalIds = index.ReinterpretCast<int32_t>();
        auto tieIds = scratch.ReinterpretCast<int32_t>();
        auto greater = dots_.ReinterpretCast<uint8_t>()[0];
        auto equal = dots_.ReinterpretCast<uint8_t>()[256];
        auto cutoffMask = dots_.ReinterpretCast<uint8_t>()[512];
        // packed/scratch/ordered storage is dead after reading the threshold.
        CreateVecIndex(logicalIds, int32_t(0), aligned);
        CreateVecIndex(ordered, 0.0F, aligned);
        CompareScalar(greater, window_, threshold, CMPMODE::GT, aligned);
        CompareScalar(equal, window_, threshold, CMPMODE::EQ, aligned);
        PipeBarrier<PIPE_V>();
        uint64_t tieCount = 0;
        GatherMask(tieIds, logicalIds, equal.ReinterpretCast<uint32_t>(),
            true, visible, {1, 1, 8, 1}, tieCount);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const int32_t cutoff = tieIds.GetValue(ties - 1);
        CompareScalar(cutoffMask, ordered, static_cast<float>(cutoff), CMPMODE::LE, aligned);
        PipeBarrier<PIPE_V>();
        auto equalWords = equal.ReinterpretCast<uint16_t>();
        auto greaterWords = greater.ReinterpretCast<uint16_t>();
        And(equalWords, equalWords, cutoffMask.ReinterpretCast<uint16_t>(), aligned / 16U);
        PipeBarrier<PIPE_V>();
        Or(greaterWords, greaterWords, equalWords, aligned / 16U);
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(heapIds_, logicalIds, greater.ReinterpretCast<uint32_t>(),
            true, visible, {1, 1, 8, 1}, selected);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return static_cast<uint32_t>(selected);
    }

    __aicore__ inline void SelectRows()
    {
        const uint32_t k = data_.selectionConfig & 0xFFFFU;
        const uint32_t rows = data_.groups * data_.batchCount;
        const uint32_t vectorCore = GetBlockIdx();
        for (uint32_t row = vectorCore; row < rows; row += data_.usedCubeCores * 2) {
            const uint32_t group = row / data_.batchCount;
            const uint32_t token = data_.batchBegin + row % data_.batchCount;
            uint32_t request = 0, visible = 0;
            (void)Request(token, request, visible);
            uint32_t size = 0;
            if (visible <= k) {
                const uint32_t alignedVisible = (visible + 7U) / 8U * 8U;
                if (alignedVisible != 0) CreateVecIndex(heapIds_, int32_t(0), alignedVisible);
                const uint32_t alignedK = (k + 7U) / 8U * 8U;
                if (alignedK > alignedVisible)
                    Duplicate(heapIds_[alignedVisible], int32_t(-1), alignedK - alignedVisible);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                for (uint32_t i = visible; i < Min(alignedVisible, k); ++i) heapIds_.SetValue(i, -1);
                size = k;
            } else if (visible <= 1024) {
                size = SelectByThreshold(group, token, visible, k);
            } else {
                // A heap of the rejected IDs is smaller when K is near R.
                // Comparator reversal preserves the original score/ID total order.
                const bool complement = visible <= kSelectionWindow && visible - k < k;
                const uint32_t limit = complement ? visible - k : k;
                reverseHeap_ = complement;
                const uint64_t base = ScoreBase(group, token);
                for (uint32_t begin = 0; begin < visible; begin += kSelectionWindow) {
                    const uint32_t count = Min(kSelectionWindow, visible - begin);
                    DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
                    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                    DataCopyPad(window_, scores_[base + begin], copy, pad);
                    SetFlag<HardEvent::MTE2_S>(0);
                    WaitFlag<HardEvent::MTE2_S>(0);
                    for (uint32_t i = 0; i < count; ++i) {
                        const float score = window_.GetValue(i);
                        const int32_t id = static_cast<int32_t>(begin + i);
                        if (size < limit) {
                            Push(score, id, size);
                        } else if (HeapBetter(score, id, heapScores_.GetValue(0), heapIds_.GetValue(0))) {
                            ReplaceRoot(score, id, size);
                        }
                    }
                }
                OrderSelectedIds(size, visible, complement);
                if (complement) size = k;
            }
            for (uint32_t i = size; i < k; ++i) heapIds_.SetValue(i, -1);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + token) * k;
            DataCopyExtParams copy{1, k * 4U, 0, 0, 0};
            DataCopyPad(ids_[out], heapIds_, copy);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }
    }

    PagedTilingData data_{};
    TPipe *pipe_ = nullptr;
    uint32_t core_ = 0, sub_ = 0;
    bool reverseHeap_ = false;
    uint32_t cachedRequest_ = 0xFFFFFFFFU, cachedPrefix_ = 0;
    int32_t cachedFirstPage_ = 0;
    bool cachedLinear_ = true;
    GlobalTensor<bfloat16_t> query_, key_;
    GlobalTensor<float> weightsFloat_, scores_, bridge_;
    GlobalTensor<uint16_t> weightsBits_;
    GlobalTensor<half> weightsHalf_;
    GlobalTensor<int32_t> lengths_, table_, cu_, ids_;
    TBuf<TPosition::A1> qL1Buffer_;
    TBuf<TPosition::B1> kL1Buffer_;
    TBuf<TPosition::A2> qL0Buffer_;
    TBuf<TPosition::B2> kL0Buffer_;
    TBuf<TPosition::CO1> cL0Buffer_;
    TBuf<TPosition::VECCALC> dotBuffer_, accBuffer_, partBuffer_, weightBuffer_;
    TBuf<TPosition::VECCALC> windowBuffer_, heapScoreBuffer_, heapIdBuffer_;
    LocalTensor<bfloat16_t> qL1_, kL1_, qL0_, kL0_;
    LocalTensor<float> cL0_, dots_, acc_, partial_, weight_, window_, heapScores_;
    LocalTensor<int32_t> heapIds_;
};
} // namespace IndexerRegionScoreA3
#endif
