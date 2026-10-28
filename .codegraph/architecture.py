"""Draw dependency figures of glide from the CodeGraph index.

The figures are Graphviz diagrams built from the import edges stored in ``.codegraph/codegraph.db``: an overview of
the packages, and with ``--modules`` a zoom on the modules of the estimators, engines and monitors packages. Only the
standard library is used; the ``dot`` executable (Graphviz) is optional and only needed to render the images, the
``.dot`` source is always written.

Run it through ``make architecture`` or ``make architecture-modules``, which refresh the index first. Files are
written to ``architecture/`` at the repository root.
"""

import argparse
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

CODEGRAPH_DIRECTORY = Path(__file__).resolve().parent
DATABASE_PATH = CODEGRAPH_DIRECTORY / "codegraph.db"
OUTPUT_DIRECTORY = CODEGRAPH_DIRECTORY.parent / "architecture"
ROOT_PACKAGE = "glide"
MODULE_VIEW_PACKAGES = ("estimators", "engines", "monitors")
FONT = "Helvetica"
DEFAULT_FILL_COLOR = "#e6e6e6"
FILL_COLORS = {
    "estimators": "#cfe8ff",
    "engines": "#ffe3b3",
    "monitors": "#d9f2d0",
    "samplers": "#f6d5ff",
    "confidence_intervals": "#fff3b0",
    "confidence_sequences": "#fff3b0",
    "mean_inference_results": "#ffd6d6",
    "mean_monitoring_results": "#ffd6d6",
}


def _package_name(path: str) -> Optional[str]:
    parts = path.split("/")
    name = parts[1].removesuffix(".py") if parts[0] == ROOT_PACKAGE and len(parts) > 1 else None
    return name


def _load_import_edges() -> List[Tuple[str, str]]:
    query = """
        SELECT source.file_path, target.file_path
        FROM edges
        JOIN nodes AS source ON source.id = edges.source
        JOIN nodes AS target ON target.id = edges.target
        WHERE edges.kind = 'imports' AND source.kind = 'file'
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        edges = connection.execute(query).fetchall()
    return edges


def _load_module_counts() -> Counter:
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute("SELECT file_path FROM nodes WHERE kind = 'file'").fetchall()
    module_counts = Counter(_package_name(path) for (path,) in rows if not path.endswith("__init__.py"))
    return module_counts


def _build_packages_dot(edges: List[Tuple[str, str]], module_counts: Counter) -> str:
    import_counts: Counter = Counter()
    for source, target in edges:
        source_package, target_package = _package_name(source), _package_name(target)
        if source_package and target_package and source_package != target_package:
            import_counts[(source_package, target_package)] += 1

    lines = [
        "digraph architecture {",
        f'  rankdir=LR; bgcolor="white"; labelloc="t"; fontname="{FONT}"; fontsize=12;',
        '  label="glide packages: an arrow A to B means A imports from B (label: number of imports)";',
        f'  node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=12];',
        f'  edge [color="#777777", fontname="{FONT}", fontsize=9];',
    ]
    for package in sorted({package for pair in import_counts for package in pair}):
        modules = module_counts[package]
        label = f"{package}\\n{modules} module{'s' if modules > 1 else ''}"
        color = FILL_COLORS.get(package, DEFAULT_FILL_COLOR)
        lines.append(f'  "{package}" [label="{label}", fillcolor="{color}"];')
    for (source, target), count in sorted(import_counts.items()):
        penwidth = 1 + min(count, 40) / 8
        lines.append(f'  "{source}" -> "{target}" [label="{count}", penwidth={penwidth:.1f}];')
    lines.append("}")
    source_code = "\n".join(lines) + "\n"
    return source_code


def _build_modules_dot(edges: List[Tuple[str, str]]) -> str:
    def is_shown(path: str) -> bool:
        shown = _package_name(path) in MODULE_VIEW_PACKAGES and not path.endswith("__init__.py")
        return shown

    shown_edges = sorted(
        {(source, target) for source, target in edges if source != target and is_shown(source) and is_shown(target)}
    )
    shown_modules = sorted({path for edge in shown_edges for path in edge})

    lines = [
        "digraph architecture {",
        f'  rankdir=LR; bgcolor="white"; labelloc="t"; fontname="{FONT}"; fontsize=12; concentrate=true; ranksep=1.0;',
        '  label="glide modules: an arrow A to B means A imports from B";',
        f'  node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=11];',
        '  edge [color="#888888", arrowsize=0.6];',
    ]
    for package in MODULE_VIEW_PACKAGES:
        color = FILL_COLORS[package]
        lines.append(
            f'  subgraph cluster_{package} {{ label="glide/{package}"; fontname="{FONT}-Bold"; '
            f'style="rounded,filled"; fillcolor="{color}33"; color="{color}";'
        )
        for path in shown_modules:
            if _package_name(path) == package:
                label = path.split("/")[-1].removesuffix(".py")
                lines.append(f'    "{path}" [label="{label}", fillcolor="{color}"];')
        lines.append("  }")
    for source, target in shown_edges:
        lines.append(f'  "{source}" -> "{target}";')
    lines.append("}")
    source_code = "\n".join(lines) + "\n"
    return source_code


def _render(dot_path: Path) -> List[Path]:
    if shutil.which("dot") is None:
        print("Graphviz 'dot' not found: wrote the .dot source only (install Graphviz to render images).")
        return []
    image_paths = [dot_path.with_suffix(".svg"), dot_path.with_suffix(".png")]
    for image_path in image_paths:
        command = ["dot", f"-T{image_path.suffix[1:]}", "-Gdpi=130", str(dot_path), "-o", str(image_path)]
        subprocess.run(command, check=True)
    return image_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modules", action="store_true", help="Draw the modules of the main packages instead.")
    arguments = parser.parse_args()

    if not DATABASE_PATH.exists():
        print(f"No CodeGraph index at {DATABASE_PATH}; run 'codegraph init' first.", file=sys.stderr)
        return 1

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    edges = _load_import_edges()
    if arguments.modules:
        dot_path = OUTPUT_DIRECTORY / "modules.dot"
        dot_path.write_text(_build_modules_dot(edges))
    else:
        dot_path = OUTPUT_DIRECTORY / "packages.dot"
        dot_path.write_text(_build_packages_dot(edges, _load_module_counts()))
    for path in [dot_path, *_render(dot_path)]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
