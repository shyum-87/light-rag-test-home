"""src/config.py — config.yaml 로딩 + 환경변수 치환."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatabaseConfig:
    driver: str
    host: str
    port: int
    database: str
    username: str
    password: str


@dataclass
class SafetyConfig:
    allowed_statements: list[str]
    blocked_keywords: list[str]
    max_rows: int
    table_whitelist: list[str]


@dataclass
class ExecutionConfig:
    mode: str  # "approval" or "auto"
    log_all_sql: bool
    log_path: str


@dataclass
class LightRAGConfig:
    working_dir: str
    domain_docs_dir: str
    embed_model_path: str


@dataclass
class AppConfig:
    database: DatabaseConfig
    safety: SafetyConfig
    execution: ExecutionConfig
    lightrag: LightRAGConfig


def _substitute_env_vars(value: str) -> str:
    """${VAR_NAME} 패턴을 환경변수 값으로 치환한다."""
    if not isinstance(value, str):
        return value
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _substitute_recursive(obj):
    """dict/list를 재귀적으로 순회하며 환경변수를 치환한다."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_recursive(v) for v in obj]
    return obj


def load_config(path: str = "config.yaml") -> AppConfig:
    """YAML 파일을 읽어 AppConfig를 반환한다."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _substitute_recursive(raw)

    return AppConfig(
        database=DatabaseConfig(**raw["database"]),
        safety=SafetyConfig(**raw["safety"]),
        execution=ExecutionConfig(**raw["execution"]),
        lightrag=LightRAGConfig(**raw["lightrag"]),
    )
