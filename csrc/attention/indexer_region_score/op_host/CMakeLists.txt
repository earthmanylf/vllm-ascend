# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE indexer_region_score_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME IndexerRegionScore
    COMPUTE_UNIT ascend950
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
            -mllvm -cce-vf-remove-membar=false
)
add_ops_compile_options(
    OP_NAME IndexerRegionScore
    COMPUTE_UNIT ascend910_93
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)
add_modules_sources(OPTYPE indexer_region_score ACLNNTYPE aclnn_exclude)
