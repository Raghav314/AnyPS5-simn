import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


HEX64 = re.compile(r"[0-9a-f]{16}")
FIELDS = re.compile(r"\b([a-z_]+)=(\S+)")


def valid_hash(value):
    return HEX64.fullmatch(value or "") is not None


def read_log(lines):
    guest_entry = None
    programs = defaultdict(set)
    origins = {}
    demanded = {}
    for line in lines:
        values = dict(FIELDS.findall(line))
        if "[kyty-startup] guest_entry " in line:
            if guest_entry is not None:
                raise ValueError("expected a single guest-entry marker")
            guest_entry = int(values["monotonic_ns"])
        elif ("[kyty-variant] build stage=CS " in line or "[kyty-startup] runtime_pipeline kind=compute " in line
              or "[kyty-compute-origin] " in line):
            variant = values.get("variant", values.get("module"))
            shader = values.get("shader")
            if valid_hash(variant) and valid_hash(shader):
                programs[variant].add(shader)
                if "[kyty-compute-origin] " in line:
                    origins.setdefault(variant, values)
        elif ("[kyty-driver-pipeline] kind=compute " in line and values.get("mode") == "create"
              and values.get("event") == "compile_required"):
            variant = values.get("variant")
            if valid_hash(variant):
                timestamp = int(values["monotonic_ns"])
                demanded.setdefault(variant, []).append(timestamp)
    if guest_entry is None:
        raise ValueError("no guest-entry marker in the log")
    rows = []
    for variant, attempts in demanded.items():
        timestamps = [time for time in attempts if time >= guest_entry]
        if not timestamps:
            continue
        shaders = programs[variant]
        shader = next(iter(shaders)) if len(shaders) == 1 else ""
        origin = origins.get(variant, {}) if shader else {}
        rows.append({"variant": variant, "shader": shader, "first_request_ns": min(timestamps) - guest_entry,
                     "route": origin.get("route", ""), "submit": origin.get("submit", ""),
                     "packet": origin.get("packet", ""), "caller_return": origin.get("caller_return", "")})
    return sorted(rows, key=lambda row: (row["first_request_ns"], row["variant"]))


def recipe_programs(path):
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "kyty compute recipes 1":
            raise ValueError("unsupported compute recipe version")
        programs = set()
        for line in stream:
            fields = line.split()
            if len(fields) != 8 or not valid_hash(fields[0]):
                raise ValueError("invalid compute recipe record")
            programs.add(fields[0])
        return programs


def manifest_compute_programs(path):
    programs = set()
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "# kyty pipeline manifest v1":
            raise ValueError("unsupported pipeline manifest version")
        for line in stream:
            fields = line.split()
            if len(fields) < 4 or fields[0] != "M" or fields[2] != "cs":
                continue
            for field in fields[3:]:
                if field.startswith("code=") and valid_hash(field[5:]):
                    programs.add(field[5:])
                    break
    return programs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find first-use compute compile requests and their guest dispatch origins.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--recipes", type=Path, help="only select programs for which offline creation recipes exist")
    parser.add_argument("--exclude", type=Path, help="exclude a list of previously prepared program hashes")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="exclude programs present in a compute manifest")
    parser.add_argument("--seed", type=Path, help="write selected program hashes for precache prepare --select")
    parser.add_argument("--report", type=Path, help="write the private module, packet and caller report as TSV")
    args = parser.parse_args(argv)
    with args.log.open(encoding="utf-8", errors="replace") as stream:
        rows = read_log(stream)
    shaders = {row["shader"] for row in rows if row["shader"]}
    eligible = shaders & recipe_programs(args.recipes) if args.recipes else shaders
    excluded = set(args.exclude.read_text(encoding="ascii").split()) if args.exclude else set()
    if not all(valid_hash(value) for value in excluded):
        parser.error("--exclude requires a list of 16-digit lowercase hex hashes")
    for manifest in args.exclude_manifest:
        excluded.update(manifest_compute_programs(manifest))
    seed = sorted(eligible - excluded)
    if args.seed:
        with args.seed.open("x", encoding="ascii") as stream:
            stream.writelines(f"{shader}\n" for shader in seed)
    if args.report:
        with args.report.open("x", encoding="ascii", newline="") as stream:
            fields = ("variant", "shader", "first_request_ns", "route", "submit", "packet", "caller_return")
            writer = csv.DictWriter(stream, fields, dialect="excel-tab")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"variants": len(rows), "programs": len(shaders), "eligible_programs": len(eligible),
                      "selected_programs": len(seed), "variants_with_packet": sum(bool(row["packet"]) for row in rows),
                      "variants_with_caller": sum(row["caller_return"] not in ("", "0000000000000000") for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
