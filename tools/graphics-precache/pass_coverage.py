from collections import Counter, defaultdict

from session import fields, read_manifest


FILL_MODES = {2: "Line", 3: "Fill"}
STATE_NAMES = ("rasterizer", "depth_stencil", "blend")
COMPARED_FIELDS = ("polygon_mode", "depth_clip", "alpha_to_coverage", "blend_enable", "color_write_mask")


def index_pass_states(bundles):
    by_stage = defaultdict(list)
    for bundle in bundles:
        for effect in bundle["effects"]:
            for row in effect["passes"]:
                shaders = row["shaders"]
                if any(stage not in ("vs", "ps") for stage in shaders) or not shaders:
                    continue
                pair = tuple(shaders.get(stage, {}).get("code") for stage in ("vs", "ps"))
                by_stage[pair].append({name: effect[name][index] for name, index in row["states"].items()})
    return by_stage


def compare_color_targets(blend, attachments, options):
    if not attachments:
        return {}
    if blend["independent_blend"] == 1:
        targets = [blend["targets"][index] for index in attachments]
    elif attachments == [0]:
        targets = [blend["targets"][0]]
    else:
        return {}
    return {
        "blend_enable": all(target["enable"] == int(str(options.get(f"b{index}", "0")).split(".")[0])
                            for index, target in zip(attachments, targets)),
        "color_write_mask": all(target["write_mask"] == int(options.get(f"m{index}", "0"), 16)
                                for index, target in zip(attachments, targets)),
    }


def compare_selected_fields(states, options, attachments):
    compared = {}
    if "rasterizer" in states:
        raster = states["rasterizer"]
        if raster["fill_mode"] in FILL_MODES:
            compared["polygon_mode"] = FILL_MODES[raster["fill_mode"]] == options.get("poly", "Fill")
        if raster["depth_clip_enable"] in (0, 1):
            compared["depth_clip"] = bool(raster["depth_clip_enable"]) == ("noclip" not in options)
    if "blend" in states:
        blend = states["blend"]
        if blend["alpha_to_coverage"] in (0, 1) and "p" in options:
            compared["alpha_to_coverage"] = bool(blend["alpha_to_coverage"]) == ("a2c" in options)
        compared.update(compare_color_targets(blend, attachments, options))
    return compared


def compare_manifest_states(path, stage_states):
    modules, pipelines = read_manifest(path)
    availability = Counter()
    results = {field: Counter() for field in COMPARED_FIELDS}
    for row in pipelines:
        options = fields(row.split()[1:])
        pair = tuple(modules[options[stage]][1]["code"] if stage in options else None for stage in ("v", "p"))
        candidates = () if "h" in options or "d" in options else stage_states.get(pair, ())
        available = {name for states in candidates for name in states}
        availability.update(available)
        availability["stage_pair"] += bool(candidates)
        attachments = [index for index in range(8) if f"c{index}" in options]
        flags_by_field = defaultdict(set)
        for states in candidates:
            for field, agrees in compare_selected_fields(states, options, attachments).items():
                flags_by_field[field].add(agrees)
        for field, histogram in results.items():
            flags = flags_by_field[field]
            if len(flags) == 2:
                histogram["mixed"] += 1
            elif flags:
                histogram["agree" if True in flags else "differ"] += 1
            else:
                histogram["unobserved"] += 1

    return dict(graphics_records=len(pipelines), with_pair_candidates=availability["stage_pair"],
                with_explicit_selection={name: availability[name] for name in STATE_NAMES},
                fields={field: dict(counts) for field, counts in results.items()},
                from_runtime_key=["attachment_formats", "samples", "topology", "vertex_input", "shader_specialization"],
                not_compared=["blend_factors", "blend_operations", "depth_stencil_testing", "cull_mode", "front_face"])
