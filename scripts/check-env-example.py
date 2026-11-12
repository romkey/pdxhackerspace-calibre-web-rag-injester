#!/usr/bin/env python3
"""Pre-commit check: .env.example lists every variable the app reads.

    scripts/check-env-example.py [PROJECT_DIR]

Finds the environment variables read in src/ (env.get("X"), env["X"],
os.environ.get("X"), os.environ["X"], os.getenv("X")), including through
helpers that read a parameter (def required(name): env.get(name), called
as required("X")), and fails, naming each one, if .env.example has no line
for it. A commented-out line
("# X=...") counts, for variables that are optional or set elsewhere.

.env.example may list more than the app reads: variables for compose
services (POSTGRES_PASSWORD) or tools (UV_PUBLISH_TOKEN).
"""

import ast
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
ENV_OBJECTS = {"env", "environ"}  # env (a Mapping parameter) or os.environ


def is_env(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in ENV_OBJECTS) or (
        isinstance(node, ast.Attribute) and node.attr == "environ"
    )


def first_name(args: list[ast.expr]) -> str | None:
    if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        return args[0].value if NAME.match(args[0].value) else None
    return None


def env_key(node: ast.AST) -> ast.expr | None:
    """The key expression if node reads an environment variable."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (node.func.attr == "get" and is_env(node.func.value)) or node.func.attr == "getenv":
            return node.args[0] if node.args else None
    elif isinstance(node, ast.Subscript) and is_env(node.value):
        return node.slice
    return None


def env_helpers(tree: ast.AST) -> set[str]:
    """Functions that read the variable named by their first parameter."""
    helpers = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef) or not func.args.args:
            continue
        param = func.args.args[0].arg
        for node in ast.walk(func):
            key = env_key(node)
            if isinstance(key, ast.Name) and key.id == param:
                helpers.add(func.name)
    return helpers


def variables_read(source: str) -> set[str]:
    tree = ast.parse(source)
    helpers = env_helpers(tree)
    names = set()
    for node in ast.walk(tree):
        key = env_key(node)
        name = first_name([key]) if key is not None else None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helpers
        ):
            name = first_name(node.args)
        if name:
            names.add(name)
    return names


def listed(example: Path) -> set[str]:
    names = set()
    for line in example.read_text().splitlines():
        line = line.lstrip("# ").removeprefix("export ")
        if "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path()
    example = root / ".env.example"
    read = set()
    for path in sorted((root / "src").rglob("*.py")):
        # Skip hidden folders (e.g. a .claude/worktrees checkout of another branch).
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        read |= variables_read(path.read_text())
    missing = sorted(read - (listed(example) if example.exists() else set()))
    for name in missing:
        print(f"{example}: no line for {name}, which the app reads")
    if missing:
        print("Add each with a dummy value and a comment; see docs/configuration.md.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
