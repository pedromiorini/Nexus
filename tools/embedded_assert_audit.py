"""Audita a localização dos asserts B101 do demo integrado.

O auditor não declara os asserts seguros nem os suprime. Ele apenas evita que
novos asserts de demonstração apareçam fora do bloco ``if __name__ == '__main__'``
no corpo direto do módulo e fornece um inventário reproduzível para a migração
incremental. Guardas aninhadas não criam faixas de exceção. O número
de linha citado só entra numa faixa se for ``int`` exato: ``bool``, float,
``str`` e ``None`` não são convertidos.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _main_block_ranges(source: str) -> list[tuple[int, int]]:
    tree = ast.parse(source)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            end = max((getattr(child, "end_lineno", child.lineno) for child in node.body), default=node.lineno)
            ranges.append((node.lineno, end))
    return ranges


def _assert_lines_outside_main(source: str, ranges: list[tuple[int, int]]) -> list[int]:
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and not any(start <= node.lineno <= end for start, end in ranges)
    ]


def _cited_line_is_inside(line_number: object, ranges: list[tuple[int, int]]) -> bool:
    """Faixa isenta só para int exato. bool não conta; nada é convertido."""
    if type(line_number) is not int:
        return False
    return any(start <= line_number <= end for start, end in ranges)


def audit(payload: dict[str, Any], source: str, filename: str) -> dict[str, Any]:
    ranges = _main_block_ranges(source)
    assert_lines_outside_main = _assert_lines_outside_main(source, ranges)
    findings = [item for item in payload.get("results", []) if item.get("test_id") == "B101"]
    outside = [
        item
        for item in findings
        if item.get("filename") == filename
        and not _cited_line_is_inside(item.get("line_number", 0), ranges)
    ]
    return {
        "filename": filename,
        "main_block_ranges": ranges,
        "b101_count": len(findings),
        "outside_main_block": [
            {"line_number": item.get("line_number"), "code": item.get("code", "")}
            for item in outside
        ],
        "assert_lines_outside_main": assert_lines_outside_main,
        "ok": bool(ranges) and not outside and not assert_lines_outside_main,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bandit_json")
    parser.add_argument("source")
    parser.add_argument("--filename", default="core/constitutional_brain.py")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.bandit_json).read_text(encoding="utf-8"))
    result = audit(payload, Path(args.source).read_text(encoding="utf-8"), args.filename)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
