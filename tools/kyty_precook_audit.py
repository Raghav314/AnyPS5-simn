import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


GUEST_ENTRY = re.compile(r"\[kyty-startup\] guest_entry monotonic_ns=(\d+)$")
GUEST_WORK = re.compile(r"\[kyty-startup\] first_(?:draw|dispatch) monotonic_ns=(\d+)$")
PREPARATION_RELEASE = re.compile(r"\[kyty-startup\] preparation_released monotonic_ns=(\d+)$")
MANIFEST_RELEASE = re.compile(
    r"\[kyty-startup\] manifest_release monotonic_ns=(\d+) result=(\w+) .* jobs_finished=(\d+) jobs_total=(\d+)$"
)
AUDIT_ENABLED = "[kyty-startup] precook_audit shader_variants=1 driver_pipeline=1"
SHADER_BUILD = re.compile(r"\[kyty-variant\] build .* @(\d+)$")
PIPELINE_MISS = re.compile(r"\[kyty-startup\] runtime_pipeline .* route=(.+?) variant=\S+ shader=\S+ monotonic_ns=(\d+)$")
DRIVER_AUDIT = re.compile(
    r"\[kyty-driver-pipeline\] kind=(\w+) variant=([0-9a-f]{16}) mode=(create|lookup) "
    r"event=(no_compile|compile_required|unverified) result=\S+ monotonic_ns=(\d+)$"
)


def summarize(path):
    guest_entry = None
    audit_enabled = False
    preparation_release = None
    manifest_releases = []
    work = []
    shaders = []
    misses = []
    driver = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if match := GUEST_ENTRY.search(line):
                if guest_entry is not None:
                    raise ValueError("multiple guest entries; supply one run")
                guest_entry = int(match[1])
            elif line.endswith(AUDIT_ENABLED):
                audit_enabled = True
            elif match := PREPARATION_RELEASE.search(line):
                preparation_release = int(match[1])
            elif match := MANIFEST_RELEASE.search(line):
                time, result, finished, total = match.groups()
                manifest_releases.append(dict(monotonic_ns=int(time), result=result, finished=int(finished), total=int(total)))
            elif match := GUEST_WORK.search(line):
                work.append(int(match[1]))
            elif match := SHADER_BUILD.search(line):
                shaders.append(int(match[1]))
            elif match := PIPELINE_MISS.search(line):
                misses.append((match[1], int(match[2])))
            elif "[kyty-driver-pipeline] " in line:
                match = DRIVER_AUDIT.search(line)
                if match is None:
                    raise ValueError(f"line {number}: malformed driver audit event")
                kind, variant, mode, result, time = match.groups()
                driver.append((kind, variant, mode, result, int(time)))

    if guest_entry is None:
        raise ValueError("guest entry is missing")
    game_driver = [event for event in driver if event[4] >= guest_entry]
    required = [event for event in game_driver if event[2:4] == ("create", "compile_required")]
    unverified = [event for event in game_driver if event[3] == "unverified"]
    driver_counts = Counter((kind, mode, result) for kind, _, mode, result, _ in game_driver)
    miss_counts = Counter(route for route, time in misses if time >= guest_entry)
    translated = sum(time >= guest_entry for time in shaders)
    guest_work_observed = any(time >= guest_entry for time in work)
    audit_observed = any(event[3] != "unverified" for event in driver)
    startup = max((event for event in manifest_releases if event["monotonic_ns"] < guest_entry),
                  key=lambda event: event["monotonic_ns"], default=None)
    return {
        "guest_work_observed": guest_work_observed,
        "audit_enabled": audit_enabled,
        "preparation_released_before_guest": preparation_release is not None and preparation_release <= guest_entry,
        "manifest_release_before_guest": startup,
        "driver_audit_observed": audit_observed,
        "guest_shader_builds": translated,
        "guest_pipeline_miss_routes": dict(sorted(miss_counts.items())),
        "guest_driver_probes": {"/".join(key): count for key, count in sorted(driver_counts.items())},
        "guest_creates_requiring_compilation": len(required),
        "guest_compile_required_variants": sorted({f"{kind}/{variant}" for kind, variant, _, _, _ in required}),
        "guest_unverified_driver_probes": len(unverified),
        "zero_compilation_observed": (
            guest_work_observed and audit_enabled and audit_observed
            and not (translated or required or unverified or miss_counts["skipped"])
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure precook misses after guest entry from one Kyty stderr log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--require-zero", action="store_true", help="fail if the log lacks zero-compilation evidence")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.log)
    except (OSError, ValueError) as error:
        print(f"kyty_precook_audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(args.require_zero and not report["zero_compilation_observed"])


if __name__ == "__main__":
    raise SystemExit(main())
