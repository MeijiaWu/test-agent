"""Configuration: ``configs/default.yaml`` plus ``--set a.b=value`` overrides.
Unknown keys are errors -- a silently ignored setting is worse than none."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .llm import LLMConfig

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class AgentConfig:
    # ablation switches (each one is an experiment)
    use_rag: bool = True
    use_review: bool = True
    use_exec: bool = True
    use_fix_check: bool = True        # confirm a reproduction against a self-written candidate fix
    # budgets
    max_exec_rounds: int = 5          # write -> run -> triage cycles
    max_design_rounds: int = 2        # design <-> review cycles (initial + revisions)
    max_fix_attempts: int = 3         # tries at a candidate fix that applies and runs
    max_tokens: int = 400_000         # per task; exceeded -> stop and report
    # retrieval
    context_chars: int = 24_000
    search_k: int = 8
    embed_model: str | None = None    # e.g. BAAI/bge-small-en-v1.5 (needs the [embed] extra)


@dataclass
class SandboxConfig:
    timeout: int = 600
    reruns: int = 1                   # extra runs of a failing test (flakiness check)
    swt_dir: str | None = None


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


def _fill(cls, data: dict, where: str):
    names = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown config keys in {where}: {sorted(unknown)}")
    return cls(**data)


def _set(d: dict, dotted: str):
    key, _, raw = dotted.partition("=")
    *parents, leaf = key.strip().split(".")
    cur = d
    for p in parents:
        if p not in cur:
            raise KeyError(f"unknown config section {p!r} in {dotted!r}")
        cur = cur[p]
    if leaf not in cur:
        raise KeyError(f"unknown config key {key!r}")
    cur[leaf] = yaml.safe_load(raw)


def load(path: str | Path | None = None, overrides=()) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT).read_text()) or {}
    raw = copy.deepcopy(raw)
    for sec, cls in (("llm", LLMConfig), ("agent", AgentConfig), ("sandbox", SandboxConfig)):
        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        base.update(raw.get(sec) or {})
        raw[sec] = base
    for o in overrides:
        _set(raw, o)
    return Config(llm=_fill(LLMConfig, raw["llm"], "llm"),
                  agent=_fill(AgentConfig, raw["agent"], "agent"),
                  sandbox=_fill(SandboxConfig, raw["sandbox"], "sandbox"))
