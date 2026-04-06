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
        for split_name, sample_count in split_sizes.items():
            split_signatures: set[str] = set()
            samples: list[BRSSample] = []
            attempt_count = 0
            target_unique_templates = min(sample_count, 2)
            config_template = ood_config if split_name == "ood" else id_config
            config = BRSConfig(
                entity_count=config_template.entity_count,
                distractor_count=config_template.distractor_count,
                step_count=step_count,
            )

            while len(samples) < sample_count:
                sample = generate_brs_sample(config=config, seed=current_seed)
                current_seed += 1
                attempt_count += 1
                signature = brs_template_signature(sample)

                if signature not in split_signatures and signature in used_split_signatures:
                    if attempt_count > sample_count * 200:
                        raise ValueError("BRS 数据生成未能找到足够多的跨 split 唯一模板")
                    continue

                if signature not in split_signatures:
                    if len(split_signatures) >= target_unique_templates:
                        if attempt_count > sample_count * 200:
                            raise ValueError("BRS 数据生成未能稳定复用 split 内模板集合")
                        continue
                    split_signatures.add(signature)
                    used_split_signatures.add(signature)

                samples.append(sample)

            bundle[split_name][step_count] = samples

    return bundle
