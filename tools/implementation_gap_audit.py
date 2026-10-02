"""Inventário conservador de lacunas explícitas de implementação em Python."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def audit(source: str, filename: str) -> dict[str, Any]:
    tree = ast.parse(source, filename=filename)
    pass_only: list[dict[str, Any]] = []
    not_implemented: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [item for item in node.body if not isinstance(item, ast.Expr) or not isinstance(getattr(item, "value", None), ast.Constant)]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                pass_only.append({"name": node.name, "line": node.lineno})
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                not_implemented.append({"line": node.lineno})
            elif isinstance(func, ast.Attribute) and func.attr == "NotImplementedError":
                not_implemented.append({"line": node.lineno})
    return {
        "filename": filename,
        "pass_only": pass_only,
        "not_implemented": not_implemented,
        "pass_only_count": len(pass_only),
        "not_implemented_count": len(not_implemented),
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit(Path(args.source).read_text(encoding="utf-8"), args.source)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
