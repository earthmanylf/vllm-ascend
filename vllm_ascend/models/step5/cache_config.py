# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-style KV cache planning for Step5 on the Ascend LBNHC layout.

Upstream vLLM runs Step5 with a block-outermost layout where
``_get_packed_kv_cache_groups`` packs the per-request CSA tail ring
(``CircularBufferSpec``, one state bucket per request) and the 8-token
sparse summary (4 KiB pages) next to the main KV pages, all as views into
one aliased backing whose per-id cost is the widest group. The Ascend
layout is layer-outer (LBNHC), so that packed path never runs, and the
layer-outer fallback relies on ``unify_kv_cache_spec_page_size``: every
group must share one page size, which would grow the summary block to
3,072 tokens and charge every auxiliary cache a full 96 KiB main page per
layer.

This module mirrors the upstream layout on Ascend instead:

* one envelope-sized backing: every group's layers start at byte 0, and
  since all groups draw block ids from one exclusive pool the aliased
  per-id spans never collide -- the pool cost per block id is the widest
  group (the envelope), not the sum of per-page-size backings;
* the sparse summary keeps its real 4 KiB page but shares the DSA
  layer's group and block ids, like upstream ``Step5SparseSummaryCache``
  ("shares the DSA layer's KV cache group and block IDs, and is prefix
  cached and transferred like any other page"): one id per 128 tokens
  carries both the main KV page and the summary page, so a request costs
  one id set instead of two and the two hash chains cannot diverge;
* the CSA tail ring keeps its own ``CircularBufferSpec`` group (one
  scratch bucket per request, not prefix cacheable), sized with the main
  page bytes so it shares the attention groups' stride class and
  free-rides inside the same envelope: per-id byte spans align to one
  grid, so spans of different groups can only intersect at identical
  block ids, which the exclusive pool already forbids.

The planner-side cost model matches the physical allocation through
``get_step5_pool_bytes_per_block``, the same hook the GLM/DSV4 multi-tensor
layouts use.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import accumulate

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import (
    create_kv_cache_group_specs,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

_STEP5_RING_SUFFIX = ".csa_tail_ring"
_STEP5_SUMMARY_SUFFIX = ".sparse_summary_cache"
_STEP5_AUX_SUFFIXES = (_STEP5_RING_SUFFIX, _STEP5_SUMMARY_SUFFIX)


def _layer_specs(group: KVCacheGroupSpec) -> list[tuple[str, KVCacheSpec]]:
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return list(group_spec.kv_cache_specs.items())
    return [(name, group_spec) for name in group.layer_names]


def _page_alignment(spec: KVCacheSpec) -> int:
    alignment = getattr(spec, "get_block_stride_alignment", None)
    if alignment is None:
        return 1
    return alignment()


def _padded_page(spec: KVCacheSpec) -> int:
    page = spec.page_size_bytes
    alignment = _page_alignment(spec)
    return page + (-page % alignment)


def _starts_on_grid(pages: list[int], offset: int, grid: int) -> bool:
    """Whether any page of a block of ``pages`` placed at ``offset`` starts on ``grid``."""
    return any(start % grid == 0 for start in accumulate(pages, initial=offset))


def is_step5_kv_cache_spec(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Whether the model carries the Step5 CSA auxiliary cache layers."""
    return any(name.endswith(_STEP5_AUX_SUFFIXES) for name in kv_cache_spec)


def is_step5_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    return any(name.endswith(_STEP5_AUX_SUFFIXES) for group in kv_cache_groups for name in group.layer_names)


def get_step5_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    orig_get_kv_cache_groups: Callable[[VllmConfig, dict[str, KVCacheSpec]], list[KVCacheGroupSpec]],
) -> list[KVCacheGroupSpec] | None:
    """Group the main layers upstream, then fold in the auxiliary layers.

    Removing the tail-ring and summary specs before upstream grouping keeps
    ``unify_kv_cache_spec_page_size`` from ever seeing the small pages (it
    would grow their block sizes to keep one aliased backing). The
    remaining main layers all share the 96 KiB page, so upstream groups
    them exactly as a model without CSA layers would.

    Each summary layer then rejoins the group holding its DSA parent layer
    (``<parent>.sparse_summary_cache`` -> ``<parent>``): the merged group
    keeps one spec per layer -- the DSA layers keep their 96 KiB page and
    the summary layers keep their 4 KiB page -- while sharing one group id
    and therefore one block-id chain, which is the upstream
    ``Step5SparseSummaryCache`` contract. The tail ring stays a separate
    ``CircularBufferSpec`` group: its one scratch bucket per request must
    not share ids with the full-sequence caches. Finally the sliding-window
    groups are re-balanced toward the widest other group (upstream's
    block-outermost packer lands on 3 x 24 layers via ``_approximate_gcd``;
    the layer-outer fallback splits 4 x 18, one window admission too many).
    """
    if not is_step5_kv_cache_spec(kv_cache_spec):
        return None

    main_specs = {name: spec for name, spec in kv_cache_spec.items() if not name.endswith(_STEP5_AUX_SUFFIXES)}
    groups = orig_get_kv_cache_groups(vllm_config, main_specs)

    # Summary layers -> the group owning their DSA parent layer. Keep the
    # parent layers first in the merged group so the scheduler-side
    # flattening (generate_scheduler_kv_cache_config keeps only the first
    # spec of a UniformTypeKVCacheSpecs) lands on the DSA FullAttentionSpec.
    summary_specs = {name: spec for name, spec in kv_cache_spec.items() if name.endswith(_STEP5_SUMMARY_SUFFIX)}
    unmerged: dict[str, KVCacheSpec] = {}
    if summary_specs:
        per_group_extras: dict[int, dict[str, KVCacheSpec]] = {}
        for name, spec in summary_specs.items():
            parent = name[: -len(_STEP5_SUMMARY_SUFFIX)]
            for idx, group in enumerate(groups):
                if parent in group.layer_names:
                    per_group_extras.setdefault(idx, {})[name] = spec
                    break
            else:
                unmerged[name] = spec
        for idx, extras in per_group_extras.items():
            member = dict(_layer_specs(groups[idx]))
            member.update(extras)
            uniform = UniformTypeKVCacheSpecs.from_specs(member)
            if uniform is None:
                # Mixed draft/target geometry upstream refused to merge:
                # keep the old isolated-group behaviour for these layers.
                unmerged.update(extras)
            else:
                groups[idx] = KVCacheGroupSpec(list(member.keys()), uniform)

    ring_specs = {name: spec for name, spec in kv_cache_spec.items() if name.endswith(_STEP5_RING_SUFFIX)}
    for aux_specs in (unmerged, ring_specs):
        if not aux_specs:
            continue
        uniform = UniformTypeKVCacheSpecs.from_specs(aux_specs)
        if uniform is None:
            # Mixed draft/target auxiliary geometry: fall back to one group
            # per layer so each keeps its own page.
            groups.extend(KVCacheGroupSpec([name], spec) for name, spec in aux_specs.items())
        else:
            groups.append(KVCacheGroupSpec(list(aux_specs.keys()), uniform))

    swa_indices = [
        idx
        for idx, group in enumerate(groups)
        if group.layer_names and all(isinstance(spec, SlidingWindowSpec) for _, spec in _layer_specs(group))
    ]
    swa_names = [name for name, spec in main_specs.items() if isinstance(spec, SlidingWindowSpec)]
    grouped_swa = sorted(name for idx in swa_indices for name in groups[idx].layer_names)
    # Re-split only a bucket upstream grouped on its own: sliding layers that
    # sit in mixed groups (a model-declared grouping) are left alone.
    if len(swa_indices) > 1 and grouped_swa == sorted(swa_names):
        others = [group for idx, group in enumerate(groups) if idx not in swa_indices]
        at = swa_indices[0]  # every group before the first sliding-window one is in `others`
        best = get_step5_max_memory_usage(vllm_config, groups)
        for num_groups in range(1, len(swa_indices)):
            try:
                split = create_kv_cache_group_specs(
                    main_specs,
                    [swa_names[i::num_groups] for i in range(num_groups)],
                )
            except (AssertionError, ValueError):
                continue  # layers the merge refuses cannot share a group
            candidate = others[:at] + split + others[at:]
            usage = get_step5_max_memory_usage(vllm_config, candidate)
            if usage < best:
                groups, best = candidate, usage
    return groups


def _step5_group_width(group: KVCacheGroupSpec) -> int:
    """Bytes one block id spans across this group's layers.

    Layers inside one group share the block id simultaneously, so their
    (padded) pages stack: this is the group's per-id width.
    """
    return sum(_padded_page(spec) for _, spec in _layer_specs(group))


def _step5_envelope(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Per-id width of the single aliased backing: the widest group.

    Every group draws ids from the one exclusive pool, so each group's
    per-id span may start at byte 0 of the same backing; an id is owned by
    at most one group at a time, so the aliased spans never collide. The
    backing therefore only has to be as wide as the widest group.
    """
    return max(_step5_group_width(group) for group in kv_cache_groups)


def get_step5_pool_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Physical bytes one scheduler block occupies in the aliased backing."""
    envelope = _step5_envelope(kv_cache_groups)
    widths = {_step5_group_width(group) for group in kv_cache_groups}
    logger.debug(
        "Step5 KV pool bytes/block (envelope): %d (group widths %s)",
        envelope,
        sorted(widths, reverse=True),
    )
    return envelope


def get_step5_max_memory_usage(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Per-request worst-case KV memory under the single aliased backing.

    Every id a request holds -- in any group -- pins one envelope slot in
    the backing, so the worst case is the sum of the per-group id claims
    (the sliding-window groups saturate at their admission cap, the ring
    holds one bucket, full attention grows to max_model_len) times the
    envelope width.
    """
    envelope = _step5_envelope(kv_cache_groups)
    total_pages = 0
    for group in kv_cache_groups:
        if getattr(group, "host_resident", False):
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_pages += spec.max_memory_usage_pages(vllm_config)
        else:
            total_pages += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )
    return total_pages * envelope


def get_step5_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Like upstream ``get_kv_cache_config_from_groups`` for layer-outer
    layouts, but every tensor aliases one envelope-sized backing (the
    widest group) and ``num_blocks`` is sized by that envelope.
    """
    envelope = _step5_envelope(kv_cache_groups)
    pool_bytes_per_block = envelope
    num_blocks = available_memory // pool_bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    page_sizes = {spec.page_size_bytes for group in kv_cache_groups for _, spec in _layer_specs(group)}
    group_widths = [_step5_group_width(group) for group in kv_cache_groups]
    logger.debug(
        "Step5 KV cache plan: %d groups, page sizes %s, group widths %s (envelope bytes/block %d), num_blocks %d",
        len(kv_cache_groups),
        sorted(page_sizes),
        sorted(group_widths, reverse=True),
        envelope,
        num_blocks,
    )

    wide_page = max(page_sizes)
    wide_stacks = max(
        sum(1 for _, spec in _layer_specs(group) if spec.page_size_bytes == wide_page) for group in kv_cache_groups
    )
    backing_size = envelope * num_blocks
    kv_cache_tensors: list[KVCacheTensor] = []

    def add_layer(layer_name: str, spec: KVCacheSpec, offset: int, block_stride: int) -> None:
        if offset + (num_blocks - 1) * block_stride + spec.page_size_bytes > backing_size:
            raise ValueError(f"Step5 layer {layer_name} does not fit the {envelope}-byte envelope backing.")
        kv_cache_tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=block_stride,
                offset=offset,
            )
        )

    riding: set[int] = set()  # wide stacks already lending their page
    narrow_top = wide_stacks * wide_page * num_blocks  # first byte past every wide stack
    for group in kv_cache_groups:
        wide = [(name, spec) for name, spec in _layer_specs(group) if spec.page_size_bytes == wide_page]
        narrow = [(name, spec) for name, spec in _layer_specs(group) if spec.page_size_bytes < wide_page]
        for stack, (layer_name, spec) in enumerate(wide):
            add_layer(layer_name, spec, stack * wide_page * num_blocks, wide_page)
        if not narrow:
            continue
        narrow_bytes = sum(spec.page_size_bytes for _, spec in narrow)
        # A rider shares its stack with the K and V views of the wide layers
        # there, which start at every block's page start and half-page point,
        # and the zeroer takes one stride per start address. The narrow block
        # therefore sits as near the page centre as a 32-byte boundary (the
        # indexer operator's minimum) allows without any narrow page starting
        # on that half-page grid, so it keeps a segment of its own.
        pages = [spec.page_size_bytes for _, spec in narrow]
        slack = wide_page - narrow_bytes
        in_page = next(
            (
                offset
                for offset in sorted(range(0, slack + 1, 32), key=lambda offset: abs(offset - slack // 2))
                if not _starts_on_grid(pages, offset, wide_page // 2)
            ),
            None,
        )
        ride = None
        if in_page is not None:
            ride = next((stack for stack in range(len(wide), wide_stacks) if stack not in riding), None)
        offset = narrow_top if ride is None else ride * wide_page * num_blocks + in_page
        for layer_name, spec in narrow:
            add_layer(layer_name, spec, offset, wide_page if ride is not None else spec.page_size_bytes)
            offset += spec.page_size_bytes * (1 if ride is not None else num_blocks)
        if ride is None:
            narrow_top = offset
        else:
            riding.add(ride)

    # Upstream's recycled-block zeroer maps one stride per start address;
    # same-stride views may share a start, but two strides at one start
    # would be inexpressible.
    start_strides: dict[int, int] = {}
    for tensor in kv_cache_tensors:
        stride = start_strides.setdefault(tensor.offset, tensor.block_stride)
        if stride != tensor.block_stride:
            raise ValueError(
                f"Step5 envelope aliasing puts a {tensor.block_stride}-byte page stride at a "
                f"start address already held by a {stride}-byte one; stack bases "
                f"must keep start addresses disjoint."
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(vllm_config.cache_config.prefix_cache_retention_interval),
    )
