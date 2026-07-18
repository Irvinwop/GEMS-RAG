from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, RetrieverConfig

PROFILE_SCHEMA_VERSION = 1
_SUBCOMMANDS = {"check", "prepare", "init", "index", "query"}


def load_retriever_profile(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("retriever profile must be a JSON object")
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"retriever profile schema_version must be {PROFILE_SCHEMA_VERSION}"
        )
    if not str(raw.get("name") or "").strip():
        raise ValueError("retriever profile requires a name")
    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("retriever profile requires a non-empty rules list")
    return raw


def apply_retriever_profile(
    config: ExperimentConfig,
    profile: dict[str, Any],
) -> tuple[ExperimentConfig, dict[str, Any]]:
    available = {retriever.name for retriever in config.retrievers}
    configured: set[str] = set()
    commands_modified = 0
    retrievers = list(config.retrievers)

    for rule_index, raw_rule in enumerate(profile["rules"]):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"retriever profile rule {rule_index} must be an object")
        names = _retriever_names(raw_rule.get("retrievers"), rule_index)
        overlap = configured.intersection(names)
        if overlap:
            raise ValueError(
                f"retriever profile configures retrievers more than once: {sorted(overlap)}"
            )
        missing = set(names) - available
        if missing and bool(raw_rule.get("required", True)):
            raise ValueError(
                f"retriever profile rule {rule_index} requires missing retrievers: {sorted(missing)}"
            )
        selected = set(names).intersection(available)
        configured.update(selected)
        if not selected:
            continue

        global_options = _profile_options(
            raw_rule.get("global_options", {}),
            label=f"rule {rule_index} global_options",
        )
        subcommand_options = _profile_options(
            raw_rule.get("subcommand_options", {}),
            label=f"rule {rule_index} subcommand_options",
        )
        if not global_options and not subcommand_options:
            raise ValueError(f"retriever profile rule {rule_index} has no command options")
        command_keys = _command_keys(raw_rule.get("command_keys"), rule_index)

        updated = []
        for retriever in retrievers:
            if retriever.name not in selected:
                updated.append(retriever)
                continue
            updated_retriever, modified = _apply_rule(
                retriever,
                command_keys=command_keys,
                global_options=global_options,
                subcommand_options=subcommand_options,
                rule_index=rule_index,
            )
            updated.append(updated_retriever)
            commands_modified += modified
        retrievers = updated

    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": str(profile["name"]),
        "retrievers_modified": sorted(configured),
        "retriever_count": len(configured),
        "commands_modified": commands_modified,
    }
    return replace(config, retrievers=retrievers), report


def _apply_rule(
    retriever: RetrieverConfig,
    *,
    command_keys: tuple[str, ...],
    global_options: dict[str, Any],
    subcommand_options: dict[str, Any],
    rule_index: int,
) -> tuple[RetrieverConfig, int]:
    options = dict(retriever.options)
    modified = 0
    for key in command_keys:
        command = options.get(key)
        if not isinstance(command, list | tuple) or not command:
            raise ValueError(
                f"retriever profile rule {rule_index} expected {retriever.name}.options.{key}"
            )
        original = [str(part) for part in command]
        profiled = _profile_command(
            original,
            global_options=global_options,
            subcommand_options=subcommand_options,
            label=f"{retriever.name}.options.{key}",
        )
        options[key] = profiled
        modified += int(profiled != original)
    return replace(retriever, options=options), modified


def _profile_command(
    command: list[str],
    *,
    global_options: dict[str, Any],
    subcommand_options: dict[str, Any],
    label: str,
) -> list[str]:
    subcommand_index = next(
        (index for index, part in enumerate(command) if part in _SUBCOMMANDS),
        None,
    )
    if subcommand_index is None:
        raise ValueError(f"{label} has no supported adapter subcommand")

    global_parts = _strip_options(command[:subcommand_index], global_options)
    subcommand_parts = _strip_options(command[subcommand_index + 1 :], subcommand_options)
    return [
        *global_parts,
        *_render_options(global_options),
        command[subcommand_index],
        *subcommand_parts,
        *_render_options(subcommand_options),
    ]


def _strip_options(parts: list[str], replacements: dict[str, Any]) -> list[str]:
    stripped = []
    index = 0
    while index < len(parts):
        part = parts[index]
        flag = next(
            (
                candidate
                for candidate in replacements
                if part == candidate or part.startswith(f"{candidate}=")
            ),
            None,
        )
        if flag is None:
            stripped.append(part)
            index += 1
            continue
        expects_value = not isinstance(replacements[flag], bool)
        if part == flag and expects_value and index + 1 < len(parts):
            index += 2
        else:
            index += 1
    return stripped


def _render_options(options: dict[str, Any]) -> list[str]:
    rendered = []
    for flag, value in options.items():
        if value is False:
            continue
        rendered.append(flag)
        if value is not True:
            rendered.append(str(value))
    return rendered


def _profile_options(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    options: dict[str, Any] = {}
    for raw_flag, raw_value in value.items():
        flag = str(raw_flag)
        if not flag.startswith("--") or len(flag) < 3:
            raise ValueError(f"{label} contains an invalid flag: {flag!r}")
        if raw_value is None or isinstance(raw_value, dict | list):
            raise ValueError(f"{label}.{flag} must be a scalar or boolean")
        options[flag] = raw_value
    return options


def _retriever_names(value: Any, rule_index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"retriever profile rule {rule_index} requires a non-empty retrievers list"
        )
    names = tuple(str(item).strip() for item in value)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"retriever profile rule {rule_index} has invalid retriever names")
    return names


def _command_keys(value: Any, rule_index: int) -> tuple[str, ...]:
    if value is None:
        return ("command", "check_command")
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"retriever profile rule {rule_index} command_keys must be a non-empty list"
        )
    keys = tuple(str(item) for item in value)
    if any(key not in {"command", "check_command"} for key in keys):
        raise ValueError(
            f"retriever profile rule {rule_index} command_keys may only name command or check_command"
        )
    return keys
