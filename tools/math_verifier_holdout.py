#!/usr/bin/env python3
"""Offline holdout harness for strict canonical nonnegative integer addition.

The freeze and prediction artifacts are immutable-by-convention and created
with exclusive file creation. Ground truth is intentionally computed only by
``score`` and only after predictions have been sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "input.json"
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
PREDICTIONS_NAME = "predictions.json"
SEAL_NAME = "predictions.seal.json"
REPORT_NAME = "score-report.json"
SCHEMA_VERSION = "math-verifier-holdout-v1"
# Duplicated intentionally: freeze records this identifier without importing or
# constructing the label-producing callback. score checks it against the module.
ORACLE_VERSION = "decimal-integer-addition-v1"
SOURCE_PATHS = (
    "core/constitutional_brain.py",
    "core/math_result_verifier.py",
    "tools/math_verifier_holdout.py",
    "requirements-ci.lock",
)
_HOLDOUT_ADDITION = re.compile(
    r"([1-9][0-9]{9,11}) \+ ([1-9][0-9]{9,11})", re.ASCII
)


class HarnessError(RuntimeError):
    """An artifact is missing, malformed, or fails an integrity check."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_exclusive(path: Path, data: bytes) -> None:
    """Create a file atomically with O_EXCL semantics; never replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def _read_json(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8")), raw
    except FileNotFoundError as exc:
        raise HarnessError(f"required artifact missing: {path.name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid JSON artifact: {path.name}") from exc


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError("unable to determine current git revision") from exc


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        try:
            hashes[relative] = _sha256(path.read_bytes())
        except OSError as exc:
            raise HarnessError(f"unable to read source {relative}") from exc
    return hashes


def _source_bundle_hash(hashes: dict[str, str]) -> str:
    """Digest the canonical source-path/hash mapping stored in the manifest."""
    return _sha256(_json_bytes(hashes))


def _random_operand(rng: random.Random) -> int:
    digits = rng.randint(10, 12)
    return rng.randrange(10 ** (digits - 1), 10 ** digits)


def freeze(output: str | Path, count: int = 128) -> dict[str, Any]:
    """Freeze a random, label-free set of unique operand pairs."""
    if type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("count must be an integer in [1, 10000]")
    directory = Path(output).expanduser().resolve()
    # Directory-level exclusive creation prevents accidentally reusing a set.
    directory.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(256)
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    cases = []
    while len(cases) < count:
        pair = (_random_operand(rng), _random_operand(rng))
        if pair in seen:
            continue
        seen.add(pair)
        cases.append({
            "id": f"case-{len(cases) + 1:05d}",
            "expression": f"{pair[0]} + {pair[1]}",
        })
    input_payload = {"schema": SCHEMA_VERSION, "cases": cases}
    input_bytes = _json_bytes(input_payload)
    sources = _source_hashes()
    manifest = {
        "schema": SCHEMA_VERSION,
        "seed_hex": f"{seed:064x}",
        "count": count,
        "input_sha256": _sha256(input_bytes),
        "sources": sources,
        "source_bundle_sha256": _source_bundle_hash(sources),
        "oracle_version": ORACLE_VERSION,
        "harness_version": SCHEMA_VERSION,
        "git_revision": _git_revision(),
        "python_version": sys.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_bytes = _json_bytes(manifest)
    _write_exclusive(directory / INPUT_NAME, input_bytes)
    _write_exclusive(directory / MANIFEST_NAME, manifest_bytes)
    _write_exclusive(directory / MANIFEST_HASH_NAME,
                     (_sha256(manifest_bytes) + "\n").encode("ascii"))
    return manifest


def _validate_cases(payload: Any, expected_count: int) -> list[dict[str, str]]:
    if type(payload) is not dict or payload.get("schema") != SCHEMA_VERSION:
        raise HarnessError("unsupported or malformed input schema")
    cases = payload.get("cases")
    if type(cases) is not list or len(cases) != expected_count:
        raise HarnessError("input case count does not match manifest")
    ids: set[str] = set()
    expressions: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if type(case) is not dict or set(case) != {"id", "expression"}:
            raise HarnessError("each input case must contain only id and expression")
        identifier, expression = case["id"], case["expression"]
        if type(identifier) is not str or identifier != f"case-{index:05d}" or identifier in ids:
            raise HarnessError("case ids must be unique canonical sequence identifiers")
        if type(expression) is not str or _HOLDOUT_ADDITION.fullmatch(expression) is None:
            raise HarnessError("input contains an unsupported addition expression")
        if expression in expressions:
            raise HarnessError("input expressions must be unique")
        ids.add(identifier)
        expressions.add(expression)
    return cases


def verify_frozen(directory: str | Path) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    """Verify manifest, detached manifest digest, input and frozen sources."""
    directory = Path(directory).expanduser().resolve()
    manifest, manifest_bytes = _read_json(directory / MANIFEST_NAME)
    try:
        expected_manifest_hash = (directory / MANIFEST_HASH_NAME).read_text("ascii").strip()
    except OSError as exc:
        raise HarnessError("manifest digest is missing") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_hash) or _sha256(manifest_bytes) != expected_manifest_hash:
        raise HarnessError("manifest digest mismatch")
    required_manifest = {
        "schema", "seed_hex", "count", "input_sha256", "sources",
        "source_bundle_sha256", "oracle_version", "harness_version",
        "git_revision", "python_version", "created_utc",
    }
    if (type(manifest) is not dict or set(manifest) != required_manifest
            or manifest.get("schema") != SCHEMA_VERSION):
        raise HarnessError("unsupported or malformed manifest")
    count = manifest.get("count")
    if type(count) is not int or not 1 <= count <= 10_000:
        raise HarnessError("manifest count is invalid")
    input_payload, input_bytes = _read_json(directory / INPUT_NAME)
    input_hash = manifest.get("input_sha256")
    if type(input_hash) is not str or _sha256(input_bytes) != input_hash:
        raise HarnessError("input SHA256 does not match manifest")
    cases = _validate_cases(input_payload, count)
    stored_sources = manifest.get("sources")
    current_sources = _source_hashes()
    if type(stored_sources) is not dict or stored_sources != current_sources:
        raise HarnessError("frozen source hashes differ from current source files")
    if (manifest.get("source_bundle_sha256") != _source_bundle_hash(stored_sources)
            or manifest.get("oracle_version") != ORACLE_VERSION
            or manifest.get("harness_version") != SCHEMA_VERSION):
        raise HarnessError("frozen source/version commitment is invalid")
    if type(manifest.get("seed_hex")) is not str or not re.fullmatch(r"[0-9a-f]{64}", manifest["seed_hex"]):
        raise HarnessError("manifest seed is invalid")
    if type(manifest.get("git_revision")) is not str or not manifest["git_revision"]:
        raise HarnessError("manifest git revision is missing")
    if type(manifest.get("python_version")) is not str or not manifest["python_version"]:
        raise HarnessError("manifest Python version is missing")
    if manifest["python_version"] != sys.version:
        raise HarnessError("frozen Python version differs from current interpreter")
    return manifest, cases, _sha256(manifest_bytes)


class _QuietLog:
    """Minimal logger interface used by RealToolUse; no external side effects."""
    def log_event(self, event_type: str, event_data: dict[str, Any]) -> str:
        return ""


def _primitive(value: Any, field: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise HarnessError(f"tool returned a non-primitive or non-finite {field}")


def _prediction_record(case: dict[str, str], result: Any) -> dict[str, Any]:
    from core.constitutional_brain import ToolExecutionResult

    if type(result) is not ToolExecutionResult:
        raise HarnessError("execute_tool did not return ToolExecutionResult")
    if type(result.tool_name) is not str or type(result.success) is not bool:
        raise HarnessError("tool result has invalid tool_name/success types")
    if result.error is not None and type(result.error) is not str:
        raise HarnessError("tool result error must be a string or null")
    elapsed = _primitive(result.execution_time, "execution_time")
    timestamp = _primitive(result.timestamp, "timestamp")
    if type(elapsed) not in (int, float) or type(timestamp) not in (int, float):
        raise HarnessError("tool timing fields must be numeric primitives")
    if type(result.cached) is not bool:
        raise HarnessError("tool cached field must be a bool")
    return {
        "id": case["id"],
        "expression": case["expression"],
        "tool_name": result.tool_name,
        "success": result.success,
        "result": _primitive(result.result, "result"),
        "error": result.error,
        "execution_time": elapsed,
        "cached": result.cached,
        "timestamp": timestamp,
    }


def _predict_with_tool(directory: str | Path, tool_use: Any) -> dict[str, Any]:
    """Internal executor seam for synthetic tests; production uses ``predict``."""
    manifest, cases, manifest_hash = verify_frozen(directory)
    directory = Path(directory).expanduser().resolve()
    if (directory / PREDICTIONS_NAME).exists() or (directory / SEAL_NAME).exists():
        raise FileExistsError("predictions or seal already exist; refusing to overwrite")
    predictions = []
    for case in cases:
        try:
            result = tool_use.execute_tool("math_calculate", {"expression": case["expression"]})
        except Exception as exc:
            # Preserve failures as predictions; do not retry or compute a label.
            from core.constitutional_brain import ToolExecutionResult
            result = ToolExecutionResult(
                tool_name="math_calculate", success=False, result=None,
                error=f"{type(exc).__name__}: {exc}", execution_time=0.0,
                cached=False, timestamp=time.time(),
            )
        predictions.append(_prediction_record(case, result))
    input_hash = manifest["input_sha256"]
    payload = {
        "schema": SCHEMA_VERSION,
        "input_sha256": input_hash,
        "manifest_sha256": manifest_hash,
        "predictions": predictions,
    }
    prediction_bytes = _json_bytes(payload)
    # Exclusive creation is performed separately for prediction data and seal.
    _write_exclusive(directory / PREDICTIONS_NAME, prediction_bytes)
    seal = {
        "schema": SCHEMA_VERSION,
        "predictions_sha256": _sha256(prediction_bytes),
        "input_sha256": input_hash,
        "manifest_sha256": manifest_hash,
    }
    _write_exclusive(directory / SEAL_NAME, _json_bytes(seal))
    return payload


def predict(directory: str | Path) -> dict[str, Any]:
    """Run the real executor and seal outputs without importing or invoking oracle."""
    from core.constitutional_brain import RealToolUse

    tool_use = RealToolUse(constitutional_log=_QuietLog())
    return _predict_with_tool(directory, tool_use)


def _validate_pairing(cases: list[dict[str, str]], predictions: Any) -> list[dict[str, Any]]:
    if type(predictions) is not list or len(predictions) != len(cases):
        raise HarnessError("predictions must pair one-to-one with frozen inputs")
    expected = {case["id"]: case["expression"] for case in cases}
    seen: set[str] = set()
    required = {"id", "expression", "tool_name", "success", "result", "error",
                "execution_time", "cached", "timestamp"}
    for prediction in predictions:
        if type(prediction) is not dict or set(prediction) != required:
            raise HarnessError("prediction has missing or unexpected fields")
        identifier = prediction["id"]
        expression = prediction["expression"]
        if type(identifier) is not str or identifier not in expected or identifier in seen:
            raise HarnessError("prediction ids must exactly pair one-to-one with inputs")
        if type(expression) is not str or expression != expected[identifier]:
            raise HarnessError("prediction expression does not match its input id")
        if type(prediction["tool_name"]) is not str or type(prediction["success"]) is not bool:
            raise HarnessError("prediction tool_name/success types are invalid")
        if prediction["error"] is not None and type(prediction["error"]) is not str:
            raise HarnessError("prediction error must be a string or null")
        _primitive(prediction["result"], "result")
        for key in ("execution_time", "timestamp"):
            value = _primitive(prediction[key], key)
            if type(value) not in (int, float):
                raise HarnessError(f"prediction {key} must be numeric")
        if type(prediction["cached"]) is not bool:
            raise HarnessError("prediction cached must be bool")
        seen.add(identifier)
    if seen != set(expected):
        raise HarnessError("prediction ids are incomplete")
    return predictions


def _load_sealed_predictions(directory: Path, manifest: dict[str, Any], manifest_hash: str,
                             cases: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str]:
    payload, raw = _read_json(directory / PREDICTIONS_NAME)
    seal, _ = _read_json(directory / SEAL_NAME)
    digest = _sha256(raw)
    if (type(payload) is not dict
            or set(payload) != {"schema", "input_sha256", "manifest_sha256", "predictions"}
            or payload.get("schema") != SCHEMA_VERSION):
        raise HarnessError("prediction payload schema is invalid")
    if (type(seal) is not dict
            or set(seal) != {"schema", "predictions_sha256", "input_sha256", "manifest_sha256"}
            or seal.get("schema") != SCHEMA_VERSION):
        raise HarnessError("prediction seal is invalid")
    if seal.get("predictions_sha256") != digest:
        raise HarnessError("prediction seal hash mismatch")
    if seal.get("input_sha256") != manifest["input_sha256"] or payload.get("input_sha256") != manifest["input_sha256"]:
        raise HarnessError("predictions are not bound to the frozen input")
    if seal.get("manifest_sha256") != manifest_hash or payload.get("manifest_sha256") != manifest_hash:
        raise HarnessError("predictions are not bound to the verified manifest")
    predictions = _validate_pairing(cases, payload.get("predictions"))
    return predictions, digest


def score(directory: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    """Score only sealed predictions; replay through one fresh metacognition system."""
    manifest, cases, manifest_hash = verify_frozen(directory)
    directory = Path(directory).expanduser().resolve()
    predictions, predictions_hash = _load_sealed_predictions(
        directory, manifest, manifest_hash, cases,
    )
    # Import and construct the label-producing oracle only after the seal passes.
    from core.constitutional_brain import (
        METACOGNITIVE_SCORE_EVALUATOR, RealMetacognitionSystem, ToolExecutionResult,
    )
    from core.math_result_verifier import ORACLE_VERSION, TASK_TYPE, make_result_evaluator

    if manifest["oracle_version"] != ORACLE_VERSION:
        raise HarnessError("frozen oracle version differs from the scoring callback")
    evaluators: dict[str, Callable[[object], float]] = {}
    for case in cases:
        evaluator = make_result_evaluator(case["expression"])
        if evaluator is None:
            raise HarnessError("frozen input falls outside the verifier domain")
        evaluators[case["id"]] = evaluator

    metacognition = RealMetacognitionSystem()
    initial_updates = sum(s.usage_count for s in metacognition.strategy_selector.strategies.values())
    n_correct = 0
    for prediction in predictions:
        result = ToolExecutionResult(
            tool_name=prediction["tool_name"],
            success=prediction["success"],
            result=prediction["result"],
            error=prediction["error"],
            execution_time=prediction["execution_time"],
            cached=prediction["cached"],
            timestamp=prediction["timestamp"],
        )
        result_evaluator = evaluators[prediction["id"]]
        _, _assessment = metacognition.execute_with_monitoring(
            process_type=TASK_TYPE,
            strategy="analytical",
            execution_func=lambda saved_result=result: saved_result,
            evaluator=result_evaluator,
        )
        monitored = metacognition.monitor.completed_processes[-1]
        if monitored.score_source != METACOGNITIVE_SCORE_EVALUATOR:
            raise HarnessError("metacognitive replay did not record evaluator provenance")
        if monitored.performance == 1.0:
            n_correct += 1
    final_updates = sum(s.usage_count for s in metacognition.strategy_selector.strategies.values())
    strategy_updates = final_updates - initial_updates
    if strategy_updates != 0:
        raise HarnessError("incompatible analytical strategy unexpectedly received credit")
    report = {
        "schema": SCHEMA_VERSION,
        "oracle_version": ORACLE_VERSION,
        "n_matched": len(predictions),
        "n_correct": n_correct,
        "accuracy": n_correct / len(predictions) if predictions else 0.0,
        "strategy_updates": strategy_updates,
        "production_enabled": False,
        "generalization_claim": False,
        "labels_exposed": True,
        "hashes": {
            "input_sha256": manifest["input_sha256"],
            "manifest_sha256": manifest_hash,
            "predictions_sha256": predictions_hash,
            "source_bundle_sha256": manifest["source_bundle_sha256"],
        },
    }
    destination = Path(report_path).expanduser().resolve() if report_path else directory / REPORT_NAME
    _write_exclusive(destination, _json_bytes(report))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="create a label-free random holdout")
    freeze_parser.add_argument("--output", required=True, help="new output directory (must not exist)")
    freeze_parser.add_argument("--count", type=int, default=128)
    predict_parser = commands.add_parser("predict", help="run the tool and seal predictions")
    predict_parser.add_argument("--input", required=True, help="frozen holdout directory")
    score_parser = commands.add_parser("score", help="score sealed predictions")
    score_parser.add_argument("--input", required=True, help="frozen holdout directory")
    score_parser.add_argument("--report", help="report path; exclusive-create, defaults to score-report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            outcome = freeze(args.output, args.count)
        elif args.command == "predict":
            outcome = predict(args.input)
        else:
            outcome = score(args.input, args.report)
    except (HarnessError, FileExistsError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
