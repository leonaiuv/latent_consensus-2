from pathlib import Path

from latent_consensus.config.loader import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_load_config_resolves_inherits_and_overrides() -> None:
    config = load_config(ROOT / "configs" / "arithmetic_debug.yaml")

    assert config["experiment"]["phase"] == "phase1"
    assert config["experiment"]["name"] == "arithmetic_debug"
    assert config["model"]["base_model"] == "gpt2"
    assert config["training"]["seq_len"] == 192


def test_load_config_raises_for_missing_file() -> None:
    try:
        load_config(ROOT / "configs" / "missing.yaml")
    except FileNotFoundError as error:
        assert "missing.yaml" in str(error)
    else:
        raise AssertionError("缺失配置文件时应直接报错")
