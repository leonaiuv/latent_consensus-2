from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_DIRECTORIES = [
    "configs",
    "data",
    "src",
    "scripts",
    "tests",
    "results",
    "checkpoints",
    "logs",
]

EXPECTED_CONFIGS = [
    "configs/local_base.yaml",
    "configs/arithmetic_debug.yaml",
    "configs/brs_main.yaml",
    "configs/lc2_shared.yaml",
    "configs/lc3_shared.yaml",
    "configs/optional_independent.yaml",
]

EXPECTED_SCRIPT_ENTRYPOINTS = [
    "data/generate_arithmetic_debug.py",
    "data/generate_brs.py",
    "scripts/profile_memory.py",
    "scripts/model_accounting.py",
    "scripts/validate_datasets.py",
    "scripts/run_local_core_ladder.py",
    "scripts/run_arithmetic_debug.py",
    "scripts/run_brs_main.py",
    "scripts/run_brs_promotion.py",
]


def test_expected_directories_exist() -> None:
    for relative_path in EXPECTED_DIRECTORIES:
        path = ROOT / relative_path
        assert path.is_dir(), f"缺少目录：{relative_path}"


def test_expected_configs_exist() -> None:
    for relative_path in EXPECTED_CONFIGS:
        path = ROOT / relative_path
        assert path.is_file(), f"缺少配置文件：{relative_path}"


def test_expected_script_entrypoints_exist() -> None:
    for relative_path in EXPECTED_SCRIPT_ENTRYPOINTS:
        path = ROOT / relative_path
        assert path.is_file(), f"缺少脚本入口：{relative_path}"
