"""Configuration: typed, nested, frozen dataclasses loaded from YAML.

One section per subsystem, so a change to one part of an experiment is a
change to one block of YAML:

    experiment  run name, output directory, training seed
    model       which policy to load and how (LoRA, quantisation)
    task        which dataset/verifier, its version and split
    reward      the weighted reward components
    algorithm   GRPO settings: group size, advantage estimator, loss, clip, KL
    sampling    rollout decoding (also used for pass@k)
    training    steps, optimiser, question schedule
    evaluation  held-out size, pass@k samples, telemetry constants, probes

A YAML file may start with ``extends: other.yaml`` and list only what differs.
Overrides use dotted keys (``algorithm.mu=4``). Unknown keys are errors, so a
misspelled setting can never be silently ignored.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml


class ConfigError(ValueError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


# --------------------------------------------------------------------------- sections
@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "run"                 # run directory name (defaults to the config file's stem)
    output_dir: str = "results"       # runs are written to <output_dir>/<name>/
    seed: int = 42                    # seeds training RNG (sampling, dropout)


@dataclass(frozen=True)
class ModelConfig:
    loader: str = "hf_lora"           # registered model loader: hf_lora | tiny (CPU stand-in)
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    load_in_4bit: bool = True         # NF4 base weights (QLoRA); False loads fp16/fp32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05        # > 0 breaks rho == 1 at mu = 1 (see docs/LESSONS.md)
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")

    def __post_init__(self) -> None:
        _require(self.lora_r >= 1, "model.lora_r must be >= 1")
        _require(0.0 <= self.lora_dropout < 1.0, "model.lora_dropout must be in [0, 1)")


@dataclass(frozen=True)
class TaskConfig:
    name: str = "financial"           # registered task
    version: int = 2                  # financial: 1 = generator used for Run 6, 2 = labels fixed
    size: int = 500                   # questions generated
    train_pool: int = 400             # indices [0, train_pool) train; the rest are held out
    seed: int = 42                    # dataset generator seed
    tolerance: Dict[str, float] = field(default_factory=dict)   # per answer type overrides

    def __post_init__(self) -> None:
        _require(0 < self.train_pool < self.size, "task.train_pool must be inside (0, task.size)")


@dataclass(frozen=True)
class RewardComponentConfig:
    name: str                         # registered reward component
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


def _default_components() -> Tuple[RewardComponentConfig, ...]:
    return (RewardComponentConfig("correct_graded", 1.0, {"lam": 0.3, "kappa": 0.2}),
            RewardComponentConfig("format", 0.1, {}))


@dataclass(frozen=True)
class RewardConfig:
    components: Tuple[RewardComponentConfig, ...] = field(default_factory=_default_components)

    def __post_init__(self) -> None:
        _require(len(self.components) > 0, "reward.components must not be empty")
        names = [c.name for c in self.components]
        _require(len(names) == len(set(names)), f"reward.components has duplicates: {names}")


@dataclass(frozen=True)
class AlgorithmConfig:
    group_size: int = 6               # G: sampled answers per question
    advantage: str = "group_std"      # registered advantage estimator
    loss: str = "grpo"                # registered loss
    epsilon: float = 0.2              # clip range
    beta: float = 0.04                # KL coefficient
    mu: int = 1                       # inner epochs per rollout group

    def __post_init__(self) -> None:
        _require(self.group_size >= 2,
                 "algorithm.group_size must be >= 2 (advantages need a spread)")
        _require(self.mu >= 1, "algorithm.mu must be >= 1")
        _require(0.0 < self.epsilon < 1.0, "algorithm.epsilon must be in (0, 1)")
        _require(self.beta >= 0.0, "algorithm.beta must be >= 0")


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0          # rollout sampling: the only source of exploration
    top_p: float = 0.95
    max_new_tokens: int = 160         # also the evaluation budget

    def __post_init__(self) -> None:
        _require(self.temperature > 0.0, "sampling.temperature must be > 0")
        _require(0.0 < self.top_p <= 1.0, "sampling.top_p must be in (0, 1]")
        _require(self.max_new_tokens >= 1, "sampling.max_new_tokens must be >= 1")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 60                   # one question per step
    lr: float = 1.0e-5
    grad_clip: float = 1.0
    schedule_seed: int = 1234         # fixed question order shared by every run (paired runs)
    window: int = 15                  # log interval and window for trend summaries

    def __post_init__(self) -> None:
        _require(self.steps >= 1, "training.steps must be >= 1")
        _require(self.lr > 0.0, "training.lr must be > 0")
        _require(self.window >= 1, "training.window must be >= 1")


@dataclass(frozen=True)
class EvaluationConfig:
    n: int = 60                       # held-out questions (greedy)
    k: int = 6                        # samples per question for pass@k; 0 disables
    telemetry_lam: float = 0.3        # FIXED partial-credit measurement stick: never vary it
    telemetry_kappa: float = 0.2      # between runs that are compared
    probes: bool = True               # run the base-vs-trained question probe after training
    show: int = 2                     # completions printed per evaluation

    def __post_init__(self) -> None:
        _require(self.n >= 1, "evaluation.n must be >= 1")
        _require(self.k >= 0, "evaluation.k must be >= 0")


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        _require(self.evaluation.n <= self.task.size - self.task.train_pool,
                 "evaluation.n exceeds the held-out pool (task.size - task.train_pool)")

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict:
        return _plain(dataclasses.asdict(self))

    def with_overrides(self, overrides: Union[Mapping[str, Any], Iterable[str]]) -> "Config":
        """A new Config with dotted-key overrides applied (``{"algorithm.mu": 4}``)."""
        return from_dict(apply_overrides(self.to_dict(), overrides))


def fingerprint_of(cfg: "Config", keys: Iterable[str], extra: Optional[dict] = None) -> str:
    """Short hash of selected settings (``"model"`` or ``"model.name"``): the cache
    key of a pipeline stage. A stage is reused only if every setting that can
    change its output is unchanged."""
    d = cfg.to_dict()
    picked: Dict[str, Any] = {}
    for key in keys:
        node: Any = d
        for part in key.split("."):
            node = node[part]
        picked[key] = node
    if extra:
        picked["_extra"] = extra
    blob = json.dumps(picked, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- building
def _plain(obj: Any) -> Any:
    """Tuples -> lists so the dict round-trips through YAML/JSON unchanged."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if origin is Union:                                        # Optional[X]
        if value is None:
            return None
        tp = next(a for a in args if a is not type(None))
        origin, args = get_origin(tp), get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (tuple, Tuple):
        _require(isinstance(value, (list, tuple)), f"{where}: expected a list")
        return tuple(_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (dict, Dict):
        _require(isinstance(value, dict), f"{where}: expected a mapping")
        return dict(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is float and isinstance(value, str):                 # YAML reads "1e-5" as a string
        try:
            return float(value)
        except ValueError:
            pass
    if tp in (int, float, str, bool):
        _require(isinstance(value, tp) and not (tp is int and isinstance(value, bool)),
                 f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _build(cls: Any, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    _require(isinstance(data, dict), f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    _require(not unknown,
             f"{where}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(names)}")
    hints = get_type_hints(cls)
    kwargs = {k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as e:                                     # e.g. a required field missing
        raise ConfigError(f"{where}: {e}") from None


def from_dict(data: Mapping[str, Any]) -> Config:
    return _build(Config, dict(data), "config")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path, seen: Tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    _require(path not in seen, f"circular 'extends' at {path}")
    _require(path.exists(), f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    _require(isinstance(data, dict), f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent:
        return _deep_merge(_read_yaml(path.parent / parent, seen + (path,)), data)
    return data


def parse_override(item: str) -> Tuple[str, Any]:
    _require("=" in item, f"override {item!r} must look like section.key=value")
    key, value = item.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def apply_overrides(data: dict, overrides: Union[Mapping[str, Any], Iterable[str], None]) -> dict:
    data = copy.deepcopy(data)
    items = overrides.items() if isinstance(overrides, Mapping) else \
        (parse_override(o) for o in (overrides or []))
    for key, value in items:
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            _require(isinstance(node, dict), f"override {key!r}: {p!r} is not a section")
        node[parts[-1]] = value
    return data


def load_config(path: Union[str, Path, None] = None, overrides=None) -> Config:
    """Read a YAML config (following ``extends``), apply overrides, validate.

    ``experiment.name`` defaults to the config file's stem, so every config
    file gets its own run directory unless it names one.
    """
    raw = _read_yaml(Path(path)) if path else {}
    if path and "name" not in (raw.get("experiment") or {}):
        raw.setdefault("experiment", {})["name"] = Path(path).stem
    cfg = from_dict(apply_overrides(raw, overrides))
    check_registered(cfg)
    return cfg


def check_registered(cfg: Config) -> None:
    """Every named component must exist before any GPU time is spent."""
    from .algorithm import ADVANTAGES, LOSSES
    from .data import TASKS
    from .model import MODEL_LOADERS
    from .rewards import REWARD_COMPONENTS

    checks = [(TASKS, cfg.task.name, "task.name"),
              (ADVANTAGES, cfg.algorithm.advantage, "algorithm.advantage"),
              (LOSSES, cfg.algorithm.loss, "algorithm.loss"),
              (MODEL_LOADERS, cfg.model.loader, "model.loader")]
    checks += [(REWARD_COMPONENTS, c.name, "reward.components") for c in cfg.reward.components]
    for reg, name, where in checks:
        _require(name in reg, f"{where}: unknown {reg.kind} {name!r}; registered: {reg.names()}")
