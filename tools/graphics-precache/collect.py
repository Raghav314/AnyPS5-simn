import argparse
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from session import fields, pipeline_modules


HEADER = "# kyty pipeline manifest v1"
CAPTURE = re.compile(r"\[kyty-queued-recipe\] ([0-9a-f]{16}) (.*)")


def collect_queued_recipes(log, counts):
    pending = {}
    for line in log:
        match = CAPTURE.search(line)
        if match is None:
            continue
        variant, record = match.groups()
        record = record.rstrip()
        kind, _, _ = record.partition(" ")
        if not line.endswith("\n"):
            counts["incomplete"] += variant not in pending
        elif kind == "missing":
            counts["missing"] += 1
        elif record == HEADER:
            counts["incomplete"] += variant in pending
            pending[variant] = {}
        elif variant not in pending:
            counts["incomplete"] += 1
        elif kind == "M":
            parts = record.split(maxsplit=3)
            if len(parts) == 4 and parts[1] not in pending[variant]:
                pending[variant][parts[1]] = record
            else:
                del pending[variant]
                counts["incomplete"] += 1
        elif kind == "P":
            modules = pending.pop(variant)
            if record.startswith("P v=") and set(pipeline_modules(record)) == modules.keys():
                counts["complete"] += 1
                yield from modules.values()
                yield record
            else:
                counts["incomplete"] += 1
        else:
            del pending[variant]
            counts["incomplete"] += 1
    counts["incomplete"] += len(pending)


def merge_recipe_records(records, modules, pipelines, layouts):
    for record in records:
        parts = record.split(maxsplit=3)
        if not parts or parts[0].startswith("#"):
            continue
        kind = parts[0]
        if kind == "P" and "v" in fields(record.split()[1:]):
            pipelines.add(record)
        elif (kind == "M" and len(parts) == 4) or (kind == "T" and len(parts) >= 2):
            ident = parts[1]
            target = modules if kind == "M" else layouts
            if target.setdefault(ident, record) != record:
                raise ValueError(f"conflicting {kind} record {ident}")
        else:
            raise ValueError(f"invalid {kind} record")


def write_recipe_manifest(path, modules, pipelines, layouts):
    if any(ident not in modules for row in pipelines for ident in pipeline_modules(row)):
        raise ValueError("pipeline refers to a missing module")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            for line in [HEADER, *sorted(modules.values()), *sorted(pipelines), *sorted(layouts.values())]:
                file.write(line + "\n")
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect queued Kyty graphics recipes from runtime logs")
    parser.add_argument("logs", nargs="+", type=Path, help="runtime logs from the same title and version")
    parser.add_argument("--manifest", action="append", default=[], type=Path, help="existing .kpm to merge")
    parser.add_argument("--output", required=True, type=Path, help="destination .kpm")
    args = parser.parse_args(argv)
    modules = {}
    pipelines = set()
    layouts = {}
    counts = Counter()
    try:
        if any(args.output.resolve() == path.resolve() for path in args.logs):
            raise ValueError("output must not overwrite an input log")
        for path in args.manifest:
            with path.open(encoding="utf-8") as manifest:
                if manifest.readline().rstrip("\r\n") != HEADER:
                    raise ValueError(f"{path}: unsupported manifest header")
                if any(not line.endswith("\n") for line in manifest):
                    raise ValueError(f"{path}: incomplete manifest record")
                manifest.seek(0)
                merge_recipe_records((line.rstrip("\r\n") for line in manifest), modules, pipelines, layouts)
        previous = len(pipelines)
        for path in args.logs:
            with path.open(encoding="utf-8", errors="replace") as log:
                merge_recipe_records(collect_queued_recipes(log, counts), modules, pipelines, layouts)
        print(f"Queued captures: {counts['complete']} complete, {counts['missing']} missing dependencies, "
              f"{counts['incomplete']} incomplete records", file=sys.stderr)
        if not counts["complete"]:
            raise ValueError("no complete queued recipe was found")
        write_recipe_manifest(args.output, modules, pipelines, layouts)
        print(f"Manifest: {len(modules)} modules, {len(pipelines)} graphics records "
              f"({len(pipelines) - previous} added), {len(layouts)} tessellation layouts", file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f"collect: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
