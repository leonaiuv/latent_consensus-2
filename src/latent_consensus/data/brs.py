"""BRS: Branching Relational Search 数据生成逻辑。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import TypeAlias


Edge = tuple[str, str]


@dataclass(frozen=True)
class BRSConfig:
    entity_count: int
    distractor_count: int
    step_count: int


@dataclass(frozen=True)
class BRSSample:
    entities: list[str]
    facts: list[Edge]
    source: str
    target: str
    query: str
    teacher_steps: list[str]
    answer: str
    dead_end_branch: list[Edge]


BRSBundle: TypeAlias = dict[str, dict[int, list[BRSSample]]]


def _entity_names(entity_count: int) -> list[str]:
    if entity_count > 26:
        raise ValueError("当前 BRS 生成器最多支持 26 个实体")
    return [chr(ord("A") + index) for index in range(entity_count)]


def count_paths(facts: list[Edge], source: str, target: str) -> int:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for left, right in facts:
        adjacency[left].append(right)

    def dfs(node: str, seen: set[str]) -> int:
        if node == target:
            return 1

        total = 0
        for neighbor in adjacency.get(node, []):
            if neighbor in seen:
                continue
            total += dfs(neighbor, seen | {neighbor})
        return total

    return dfs(source, {source})


def _partition_count(total: int, part_count: int, rng: random.Random) -> list[int]:
    if part_count <= 0 or total < part_count:
        raise ValueError("非法的分支划分参数")
    remaining = total
    partitions: list[int] = []
    for part_index in range(part_count - 1):
        min_value = 1
        max_value = remaining - (part_count - part_index - 1)
        value = rng.randint(min_value, max_value)
        partitions.append(value)
        remaining -= value
    partitions.append(remaining)
    return partitions


def _find_unique_main_path(facts: list[Edge], source: str, target: str) -> list[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for left, right in facts:
        adjacency[left].append(right)

    paths: list[list[str]] = []

    def dfs(node: str, current_path: list[str], seen: set[str]) -> None:
        if node == target:
            paths.append(current_path.copy())
            return
        for neighbor in adjacency.get(node, []):
            if neighbor in seen:
                continue
            current_path.append(neighbor)
            dfs(neighbor, current_path, seen | {neighbor})
            current_path.pop()

    dfs(source, [source], {source})
    if len(paths) != 1:
        raise ValueError("BRS 样本不满足唯一路径约束")
    return paths[0]


def _branch_lengths_from(
    node: str,
    adjacency: dict[str, list[str]],
    main_path_nodes: set[str],
) -> list[int]:
    branch_lengths: list[int] = []
    for neighbor in adjacency.get(node, []):
        if neighbor in main_path_nodes:
            continue
        length = 1
        current = neighbor
        while True:
            offchain_neighbors = [
                candidate
                for candidate in adjacency.get(current, [])
                if candidate not in main_path_nodes
            ]
            if not offchain_neighbors:
                break
            current = offchain_neighbors[0]
            length += 1
        branch_lengths.append(length)
    return sorted(branch_lengths)


def brs_template_signature(sample: BRSSample) -> str:
    main_path = _find_unique_main_path(sample.facts, sample.source, sample.target)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for left, right in sample.facts:
        adjacency[left].append(right)

    main_path_nodes = set(main_path)
    branch_shapes: list[tuple[int, int]] = []
    for attach_index, node in enumerate(main_path[:-1]):
        for branch_length in _branch_lengths_from(node, adjacency, main_path_nodes):
            branch_shapes.append((attach_index, branch_length))

    return f"main={len(main_path) - 1}|branches={tuple(sorted(branch_shapes))}"


def generate_brs_sample(config: BRSConfig, seed: int) -> BRSSample:
    if config.entity_count < config.step_count + 2:
        raise ValueError("实体数不足以构成主链与死路分支")
    if config.distractor_count < 1:
        raise ValueError("BRS 至少需要一条死路分支")

    rng = random.Random(seed)
    entities = _entity_names(config.entity_count)
    main_chain = rng.sample(entities, config.step_count + 1)
    source = main_chain[0]
    target = main_chain[-1]
    facts: list[Edge] = list(zip(main_chain, main_chain[1:]))

    remaining_entities = [entity for entity in entities if entity not in main_chain]
    rng.shuffle(remaining_entities)

    branch_count = rng.randint(1, min(config.distractor_count, len(remaining_entities)))
    branch_lengths = _partition_count(len(remaining_entities), branch_count, rng)
    dead_end_branch: list[Edge] = []
    attach_candidates = main_chain[:-1]
    start_index = 0

    for branch_index, branch_length in enumerate(branch_lengths):
        branch_nodes = remaining_entities[start_index : start_index + branch_length]
        start_index += branch_length

        attach_point = source if branch_index == 0 else rng.choice(attach_candidates)
        dead_end_branch.append((attach_point, branch_nodes[0]))
        for left, right in zip(branch_nodes, branch_nodes[1:]):
            dead_end_branch.append((left, right))

    facts.extend(dead_end_branch)
    rng.shuffle(facts)

    teacher_steps = [
        f"[STEP {step_index}] {source} > {node}"
        for step_index, node in enumerate(main_chain[1:], start=1)
    ]

    return BRSSample(
        entities=entities,
        facts=facts,
        source=source,
        target=target,
        query=f"{source} ? {target}",
        teacher_steps=teacher_steps,
        answer=f"{source} > {target}",
        dead_end_branch=dead_end_branch,
    )


def serialize_brs_sample(sample: BRSSample) -> dict[str, object]:
    return {
        "entities": sample.entities,
        "facts": sample.facts,
        "source": sample.source,
        "target": sample.target,
        "query": sample.query,
        "teacher_steps": sample.teacher_steps,
        "answer": sample.answer,
        "dead_end_branch": sample.dead_end_branch,
        "template_signature": brs_template_signature(sample),
    }


def build_brs_dataset_bundle(
    step_counts: tuple[int, ...],
    split_sizes: dict[str, int],
    id_config: BRSConfig,
    ood_config: BRSConfig,
    base_seed: int = 0,
) -> BRSBundle:
    bundle: BRSBundle = {split: {} for split in split_sizes}
    current_seed = base_seed

    for step_count in step_counts:
        used_split_signatures: set[str] = set()
        step_split_budgets = _resolve_step_template_budgets(
            split_sizes=split_sizes,
            id_config=BRSConfig(
                entity_count=id_config.entity_count,
                distractor_count=id_config.distractor_count,
                step_count=step_count,
            ),
            ood_config=BRSConfig(
                entity_count=ood_config.entity_count,
                distractor_count=ood_config.distractor_count,
                step_count=step_count,
            ),
        )
        for split_name, sample_count in split_sizes.items():
            config_template = ood_config if split_name == "ood" else id_config
            config = BRSConfig(
                entity_count=config_template.entity_count,
                distractor_count=config_template.distractor_count,
                step_count=step_count,
            )
            template_budget = step_split_budgets[split_name]
            split_signatures, current_seed = _discover_split_template_pool(
                config=config,
                current_seed=current_seed,
                target_count=template_budget,
                forbidden_signatures=used_split_signatures,
            )
            used_split_signatures.update(split_signatures)
            samples, current_seed = _fill_split_samples_from_pool(
                config=config,
                current_seed=current_seed,
                sample_count=sample_count,
                allowed_signatures=split_signatures,
            )
            bundle[split_name][step_count] = samples

    return bundle


def _resolve_step_template_budgets(
    split_sizes: dict[str, int],
    id_config: BRSConfig,
    ood_config: BRSConfig,
) -> dict[str, int]:
    id_splits = [
        (split_name, sample_count)
        for split_name, sample_count in split_sizes.items()
        if split_name != "ood"
    ]
    ood_splits = [
        (split_name, sample_count)
        for split_name, sample_count in split_sizes.items()
        if split_name == "ood"
    ]

    budgets: dict[str, int] = {}
    if id_splits:
        id_capacity = _estimate_template_capacity(id_config)
        budgets.update(_allocate_template_budgets(id_splits, capacity=id_capacity))
    if ood_splits:
        ood_capacity = _estimate_template_capacity(ood_config)
        budgets.update(_allocate_template_budgets(ood_splits, capacity=ood_capacity))
    return budgets


def _estimate_template_capacity(config: BRSConfig) -> int:
    seen_signatures: set[str] = set()
    current_seed = 0
    stale_count = 0
    max_attempts = 20_000
    patience = 4_000

    while current_seed < max_attempts and stale_count < patience:
        sample = generate_brs_sample(config=config, seed=current_seed)
        current_seed += 1
        signature = brs_template_signature(sample)
        if signature in seen_signatures:
            stale_count += 1
            continue
        seen_signatures.add(signature)
        stale_count = 0

    return len(seen_signatures)


def _allocate_template_budgets(
    split_specs: list[tuple[str, int]],
    capacity: int,
) -> dict[str, int]:
    if not split_specs:
        return {}
    if capacity < len(split_specs):
        raise ValueError("BRS 模板容量不足，无法为所有 split 分配互斥模板池")

    total_samples = sum(sample_count for _split_name, sample_count in split_specs)
    budgets = {split_name: 1 for split_name, _sample_count in split_specs}
    remaining_capacity = capacity - len(split_specs)

    raw_targets = {
        split_name: capacity * sample_count / total_samples
        for split_name, sample_count in split_specs
    }
    remaining_needs = {
        split_name: max(0, min(sample_count, int(raw_targets[split_name])) - 1)
        for split_name, sample_count in split_specs
    }

    while remaining_capacity > 0 and any(need > 0 for need in remaining_needs.values()):
        split_name = max(
            remaining_needs,
            key=lambda name: (remaining_needs[name], raw_targets[name]),
        )
        if remaining_needs[split_name] <= 0:
            break
        budgets[split_name] += 1
        remaining_needs[split_name] -= 1
        remaining_capacity -= 1

    while remaining_capacity > 0:
        allocated = False
        for split_name, sample_count in sorted(
            split_specs,
            key=lambda item: raw_targets[item[0]],
            reverse=True,
        ):
            if budgets[split_name] >= sample_count:
                continue
            budgets[split_name] += 1
            remaining_capacity -= 1
            allocated = True
            if remaining_capacity == 0:
                break
        if not allocated:
            break

    return budgets


def _discover_split_template_pool(
    config: BRSConfig,
    current_seed: int,
    target_count: int,
    forbidden_signatures: set[str],
) -> tuple[set[str], int]:
    split_signatures: set[str] = set()
    attempt_count = 0
    max_attempts = max(target_count * 2000, 2000)

    while len(split_signatures) < target_count:
        sample = generate_brs_sample(config=config, seed=current_seed)
        current_seed += 1
        attempt_count += 1
        signature = brs_template_signature(sample)
        if signature in forbidden_signatures or signature in split_signatures:
            if attempt_count >= max_attempts:
                raise ValueError("BRS 数据生成未能找到足够多的跨 split 唯一模板")
            continue
        split_signatures.add(signature)

    return split_signatures, current_seed


def _fill_split_samples_from_pool(
    config: BRSConfig,
    current_seed: int,
    sample_count: int,
    allowed_signatures: set[str],
) -> tuple[list[BRSSample], int]:
    samples: list[BRSSample] = []
    exact_signatures: set[str] = set()
    attempt_count = 0
    max_attempts = max(sample_count * 1000, 5000)

    while len(samples) < sample_count:
        sample = generate_brs_sample(config=config, seed=current_seed)
        current_seed += 1
        attempt_count += 1
        signature = brs_template_signature(sample)
        if signature not in allowed_signatures:
            if attempt_count >= max_attempts:
                raise ValueError("BRS 数据生成未能稳定复用 split 内模板集合")
            continue

        exact_signature = f"{tuple(sorted(sample.facts))}|{sample.query}|{sample.answer}"
        if exact_signature in exact_signatures:
            if attempt_count >= max_attempts:
                raise ValueError("BRS 数据生成未能产出足够多的 split 内唯一样本")
            continue

        exact_signatures.add(exact_signature)
        samples.append(sample)

    return samples, current_seed
