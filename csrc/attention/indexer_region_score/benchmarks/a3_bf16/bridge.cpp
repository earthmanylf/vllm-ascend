// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn/acl_meta.h>
#include <algorithm>
#include <memory>
#include <vector>

extern "C" {
aclnnStatus aclnnIndexerRegionScoreGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, double, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
aclnnStatus aclnnIndexerRegionScore(void *, uint64_t, aclOpExecutor *, aclrtStream);
}

namespace {
struct TensorDeleter {
    void operator()(aclTensor *p) const { if (p) aclDestroyTensor(p); }
};
using TensorHandle = std::unique_ptr<aclTensor, TensorDeleter>;

TensorHandle metadata(const at::Tensor &t) {
    TORCH_CHECK(t.is_contiguous(), "benchmark expects contiguous tensors");
    aclDataType type;
    switch (t.scalar_type()) {
        case at::kBFloat16: type = ACL_BF16; break;
        case at::kFloat: type = ACL_FLOAT; break;
        case at::kInt: type = ACL_INT32; break;
        default: TORCH_CHECK(false, "unsupported benchmark dtype");
    }
    const int64_t storage = t.numel();
    auto *p = aclCreateTensor(t.sizes().data(), t.dim(), type, t.strides().data(), 0,
        ACL_FORMAT_ND, &storage, 1, t.data_ptr());
    TORCH_CHECK(p, "aclCreateTensor failed");
    return TensorHandle(p);
}

int64_t invoke(const at::Tensor &q, const at::Tensor &w, const at::Tensor &key,
    const at::Tensor &length, const at::Tensor &table, const at::Tensor &cu,
    const at::Tensor &scores, const at::Tensor &ids, const at::Tensor &workspace,
    bool full, bool launch) {
    std::vector<TensorHandle> inputs;
    for (const auto *t : {&q, &w, &key, &length, &table, &cu, &scores, &ids})
        inputs.push_back(metadata(*t));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnIndexerRegionScoreGetWorkspaceSize(
        inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get(), inputs[4].get(), inputs[5].get(),
        key.size(0), ids.size(2), 8, 1.0, 1, 3, 1, full ? inputs[6].get() : nullptr,
        inputs[7].get(), &bytes, &executor);
    if (status) {
        if (executor) aclDestroyAclOpExecutor(executor);
        TORCH_CHECK(false, "IndexerRegionScore GetWorkspaceSize failed ", status);
    }
    if (!launch) {
        aclDestroyAclOpExecutor(executor);
        return bytes;
    }
    if (workspace.numel() < static_cast<int64_t>(bytes)) {
        aclDestroyAclOpExecutor(executor);
        TORCH_CHECK(false, "workspace is too small");
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    status = aclnnIndexerRegionScore(workspace.data_ptr(), bytes, executor, stream);
    TORCH_CHECK(status == 0, "IndexerRegionScore launch failed ", status);
    return bytes;
}
}

TORCH_LIBRARY(irs_bench, m) {
    m.def("run(Tensor q, Tensor w, Tensor key, Tensor length, Tensor table, Tensor cu, "
          "Tensor(a!) scores, Tensor(b!) ids, Tensor(c!) workspace, bool full, bool launch) -> int");
}
TORCH_LIBRARY_IMPL(irs_bench, PrivateUse1, m) { m.impl("run", &invoke); }
