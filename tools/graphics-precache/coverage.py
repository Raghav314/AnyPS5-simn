import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from session import fields, pipeline_modules, read_manifest


def pipeline_key(line):
    options = fields(line.split()[1:])
    for key in ("fv", "fh", "fd", "fp"):
        options.pop(key, None)
    return tuple(sorted(options.items()))


def observations(path, module_count, pipeline_count):
    modules = {}
    pipelines = {}
    for line in path.read_text(errors="replace").splitlines():
        if match := re.fullmatch(r"module (\d+) ([01])", line):
            index = int(match[1])
            if index in modules:
                raise ValueError("duplicate module observation")
            modules[index] = match[2] == "1"
        if match := re.fullmatch(r"pipeline (\d+) (ready 1 0|failed [01] -?\d+|ineligible|layout-failed)", line):
            index = int(match[1])
            if index in pipelines:
                raise ValueError("duplicate pipeline observation")
            pipelines[index] = match[2]
    if set(modules) != set(range(module_count)) or set(pipelines) != set(range(pipeline_count)):
        raise ValueError("incomplete probe output")
    return modules, pipelines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("evaluated", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    modules, pipes = read_manifest(args.target)
    tested, recipes = read_manifest(args.evaluated)
    seen, results = observations(args.log, len(tested), len(recipes))
    module_hits = {ident for i, ident in enumerate(tested) if seen[i]} & modules.keys()
    ready = {pipeline_key(line) for i, line in enumerate(recipes) if results[i] == "ready 1 0"}
    candidates = defaultdict(set)
    for line in pipes:
        opts = fields(line.split()[1:])
        if "p" in opts:
            candidates[modules[opts["p"]][1]["code"]].add(pipeline_key(line))
    stage_hits = Counter(modules[i][0] for i in module_hits)
    code_hits = Counter(stage for stage, _ in {(modules[i][0], modules[i][1]["code"]) for i in module_hits})
    target_keys = {pipeline_key(line) for line in pipes}
    joined = Counter()
    for line in pipes:
        stages = pipeline_modules(line)
        matched = sum(i in module_hits for i in stages)
        joined[f"{matched}/{len(stages)}"] += 1
    print(json.dumps(dict(module_hits=stage_hits, codes_with_hit=code_hits, matched_stages=joined,
                          probe_results=Counter(results.values()), candidate_states=len(target_keys),
                          ready_target_states=len(ready & target_keys),
                          pixel_programs_all_states_verified=sum(keys <= ready for keys in candidates.values())),
                     sort_keys=True))


if __name__ == "__main__":
    main()
