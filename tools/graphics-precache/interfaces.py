import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from recipes import pixel_interface
from session import fields


EXPORT_FORMATS = ("ZERO", "32_R", "32_GR", "32_AR", "FP16_ABGR", "UNORM16_ABGR", "SNORM16_ABGR",
                  "UINT16_ABGR", "SINT16_ABGR", "32_ABGR")
RESOURCE_OPS = ("BUFFER_", "TBUFFER_", "S_BUFFER_", "IMAGE_", "S_LOAD", "FLAT_", "GLOBAL_", "SCRATCH_")
STAGE_CURSOR = {"ps": 19, "vs": 33, "mesh": 33}


def decoded_interfaces(path):
    decoded = {}
    invalid = set()
    for line in path.open():
        row = json.loads(line)
        isa = row.get("decoded")
        if not isa:
            continue
        code = row["code"]
        ops = isa["isa"]
        if isa["unsupported"] or isa["calls"] or isa["swap_pc"] or "S_ENDPGM" not in ops:
            invalid.add(code)
            continue
        summary = (isa["exp_mrt"], isa["exp_pos"], tuple(sorted(op for op in ops if op.startswith(RESOURCE_OPS))))
        if decoded.setdefault(code, summary) != summary:
            invalid.add(code)
    return {code: value for code, value in decoded.items() if code not in invalid}


def semantic(word):
    return dict(id=word & 255, vgpr=word >> 8 & 255, elements=word >> 16 & 15,
                half=word >> 20 & 3, flat=bool(word & 1 << 22), linear=bool(word & 1 << 23),
                custom=bool(word & 1 << 24), static_buffer=bool(word & 1 << 25),
                static_attribute=bool(word & 1 << 26), default=word >> 28 & 3, default_hi=word >> 30)


def vertex_fetch(direct, inputs):
    if direct is None or len(direct) > 11:
        return None
    slots = [None if value == 0xffff else value for value in direct] + [None] * (11 - len(direct))
    buffer, attrib = slots[8], slots[10]
    if any(reg is not None and reg + 1 >= 32 for reg in (attrib, buffer)):
        return None
    if attrib is not None and (buffer is None or not 0 < len(inputs) <= 32 or
                               any(s["static_buffer"] or s["static_attribute"] for s in inputs)):
        return None
    return {"attribute_sgpr": attrib, "buffer_sgpr": buffer, "fetch_shader_sgpr": slots[2]}


def mesh_shapes(cx, sh, enable, special, wave, scratch):
    if len(special) < 12 or special[0] != 0x25b or special[2] != 0x2d5 or special[8] != 0x29b:
        return []
    fast = enable >> 19 & 3
    merged = bool(enable & 32)
    group = special[1]
    prim_group, vertex_group = group & 511, group >> 9 & 511
    output_size, max_vertices = special[9] + 1, cx[0x1ff]
    gs_max, amp = cx[0x2ce], cx[0x2d3] & 511
    lds = (sh[0x8b] >> 19 & 255) * 128
    gs_components = sh[0x8a] >> 29 & 3
    es_components = sh[0x8b] >> 16 & 3
    if (not 1 <= output_size <= 3 or max_vertices == 0 or fast > 1 or
        (merged and gs_max < output_size) or
        (fast and (gs_components != 0 or es_components != 1 or prim_group != 1 or vertex_group != 1
                   or special[10] != 0x262 or special[11] & 7 or max_vertices != gs_max)) or
        (not fast and merged and (gs_components != 3 or es_components != 3))):
        return []
    shapes = []
    for prim in (1,) if fast else (1, 2, 4, 5, 6):
        size = min(prim, 3)
        step = 1 if prim in (5, 6) else size
        def primitives(vertices):
            return max(0, (vertices - size) // step + 1)
        count = min(prim_group, primitives(vertex_group),
                    max_vertices // gs_max if merged else primitives(max_vertices))
        if not count:
            continue
        vertices = (count - 1) * step + size
        max_prims = count * amp
        threads = max(vertices if not merged and not fast else max_vertices, max_prims)
        workgroup = ((threads + wave - 1) // wave) * wave
        shape = {"mt": f"{workgroup},1,1", "mprim": str(prim), "mppg": str(count), "mvpg": str(vertices),
                 "mmaxv": str(max_vertices), "mmaxp": str(max_prims)}
        if output_size != 3:
            shape["mops"] = str(output_size)
        if fast:
            shape["mfast"] = ""
        if lds:
            shape["mlds"] = str(lds)
        if scratch:
            shape["mscratch"] = str(scratch)
        if wave != 64:
            shape["mwave"] = str(wave)
        shapes.append(shape)
    return shapes


def graphics_recipe(row, isa):
    code, kind, cx, sh, enable, scratch, inputs, outputs, direct, special = row
    if kind not in (1, 2):
        return None
    cx, sh = dict(cx), dict(sh)
    rsrc = sh.get(11 if kind == 1 else 139)
    if rsrc is None:
        return None
    user = (rsrc >> 1 & 31) | ((rsrc >> 27 & 1) << 5)
    meta = dict(version=1, code=code, inputs=list(map(semantic, inputs)), outputs=list(map(semantic, outputs)))
    mrt, pos, ops = isa
    meta["isa"] = dict(mrt_exports=mrt, position_exports=pos, resource_operations=ops)
    if kind == 1:
        ps = pixel_interface(cx, rsrc, scratch)
        if ps is None or 0x1c4 not in cx:
            return None
        before, after, count = ps
        if count != len(inputs):
            return None
        fixed = fields([*before, *after])
        custom = sum(1 << i for i, sem in enumerate(meta["inputs"]) if sem["custom"] and not sem["half"])
        if custom:
            fixed["custom"] = f"{custom:x}"
        modes = [cx[0x1c5] >> (4 * i) & 15 for i in range(8)]
        if any(mode >= len(EXPORT_FORMATS) for mode in modes):
            return None
        slots = [i for i in range(8) if cx[0x8f] >> (4 * i) & 15]
        meta["exports"] = [dict(index=i, format=EXPORT_FORMATS[mode],
                                target=slots[i] if i < len(slots) else None)
                           for i, mode in enumerate(modes) if mode]
        meta["system_input_enable"] = cx[0x1b3]
        meta["system_input_address"] = cx[0x1b4]
        meta["barycentric_control"] = cx[0x1b8]
        meta["depth_export_format"] = cx[0x1c4]
        meta["allow_alpha_to_coverage"] = not bool(cx[0x203] & 1 << 11)
        stage = "ps"
    else:
        if enable is None or 0x207 not in cx:
            return None
        mesh = bool(enable & 32 or (rsrc >> 19 & 255))
        stage = "mesh" if mesh else "vs"
        fixed = {}
        wave = 32 if enable & 0x400000 else 64
        if cx[0x207]:
            fixed["vsout"] = f"{cx[0x207]:x}"
        if mesh:
            if not {0x1ff, 0x2ce, 0x2d3} <= cx.keys() or 0x8a not in sh:
                return None
            meta["assembly"] = mesh_shapes(cx, sh, enable, special, wave, scratch)
            if not meta["assembly"]:
                return None
            user += 8
        else:
            fetch = vertex_fetch(direct, meta["inputs"])
            if fetch is None:
                return None
            meta["fetch"] = fetch
            if fetch["attribute_sgpr"] is not None:
                fixed["dyn"] = ""
                for name, key in (("fa", "attribute_sgpr"), ("fb", "buffer_sgpr")):
                    if fetch[key]:
                        fixed[name] = str(fetch[key])
            if wave != 64:
                fixed["wave"] = str(wave)
            if scratch:
                fixed["scratch"] = str(scratch)
    if user:
        fixed["ud"] = str(user)
    meta.update(stage=stage, fixed=fixed, cursor=STAGE_CURSOR[stage])
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("headers", type=Path)
    parser.add_argument("decoded", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decoded = decoded_interfaces(args.decoded)
    programs = set()
    recipes = defaultdict(set)
    for line in args.headers.open():
        row = json.loads(line)
        code, kind = row[:2]
        if kind not in (1, 2, 4, 6, 8):
            continue
        programs.add(code)
        if code in decoded and (recipe := graphics_recipe(row, decoded[code])) is not None:
            recipes[recipe["stage"], code].add(json.dumps(recipe, sort_keys=True, separators=(",", ":")))
    with args.output.open("x") as output:
        for key in sorted(recipes):
            for line in sorted(recipes[key]):
                print(line, file=output)
    print("graphics programs", len(programs), "recipe programs", len({c for _, c in recipes}),
          "by stage", dict(Counter(stage for stage, _ in recipes)), "interfaces", sum(map(len, recipes.values())))


if __name__ == "__main__":
    main()
