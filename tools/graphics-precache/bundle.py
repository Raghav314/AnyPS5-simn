import argparse
import json
import mmap
import struct
import sys
from collections import Counter
from pathlib import Path

from catalog import xxhash
from pass_coverage import compare_manifest_states, index_pass_states
from resource_coverage import (compare_manifest_resources, index_declared_dimensions, reflected_resources,
                               reflected_texture_count)
from session import fields, read_manifest


STAGES = ("vs", "ps", "gs", "ds", "hs", "cs")
ALIGNMENT = 4096
NATIVE_SHADER_MAGIC = b"sgaProg\0"
EMBEDDED_SHADER_METADATA = 144
EMBEDDED_SHADER_MIN_SIZE = 324


class BundleReader:
    def __init__(self, data, start, include_resources=False):
        self.data = data
        self.position = start
        self.include_resources = include_resources

    def skip(self, size):
        start = self.position
        self.position += size
        if self.position > len(self.data):
            raise ValueError(f"record at {start:#x} exceeds the input")

    def take(self, size):
        start = self.position
        self.skip(size)
        return self.data[start:self.position]

    def word(self):
        return struct.unpack("<I", self.take(4))[0]

    def sized(self):
        return self.take(self.word())

    def counted(self, width):
        return struct.iter_unpack(f"<{width}s", self.take(self.word() * width))

    def name(self):
        name = self.take(self.take(1)[0])
        if not name or name[-1] or b"\0" in name[:-1]:
            raise ValueError("invalid length-prefixed name")

    def reflection(self, size=None, *, shader=False):
        table = self.sized() if size is None else self.take(size)
        resources, members, auxiliary, declared = struct.unpack_from("<4H", table)
        if declared != len(table) or 8 + 24 * (resources + members) + 8 * auxiliary > len(table):
            raise ValueError("invalid reflection table size")
        if shader:
            result = {"reflected_texture_count": reflected_texture_count(table, resources)}
            if self.include_resources:
                result["resources"] = reflected_resources(table, resources)
            return result

    def shader(self, hash64, *, metadata_bytes=152):
        self.name()
        version = self.take(1)[0]
        archive = self.sized()
        magic, header_size, code_size = struct.unpack_from("<8sII", archive)
        if magic != NATIVE_SHADER_MAGIC or header_size < 96 or code_size < 48 or 16 + header_size + code_size != len(archive):
            raise ValueError("unsupported native shader wrapper")
        if struct.unpack_from("<2I", archive, 16) != (0x34333231, 24):
            raise ValueError("invalid native shader creation header")
        if struct.unpack_from("<2I", archive, 16 + 64) != (header_size, code_size):
            raise ValueError("native shader creation sizes disagree with its wrapper")
        if archive[-48:-40] != b"barefoot":
            raise ValueError("invalid native shader footer")
        footer_version, instructions, constants, metadata = struct.unpack_from("<4I", archive, len(archive) - 32)
        length = align16(instructions) + constants
        if footer_version != 1 or not instructions or length % 4 or align16(length) + align16(metadata) != code_size - 48:
            raise ValueError("invalid native shader program extent")
        program = archive[16 + header_size:16 + header_size + length]
        self.skip(metadata_bytes)
        return {"code": f"{hash64(program, length):016x}", "native_kind": archive[16 + 90], "version": version,
                **self.reflection(shader=True)}


def align16(size):
    return (size + 15) & ~15


def rasterizer_state(record):
    fill, cull, winding, bias, clamp, slope, clip, _, multisample, flags = struct.unpack("<BBBbIIBBBB", record)
    return dict(fill_mode=fill, cull_mode=cull, front_counter_clockwise=winding, depth_bias=bias,
                depth_bias_clamp_bits=clamp, slope_scaled_depth_bias_bits=slope,
                depth_clip_enable=clip, multisample_enable=multisample, flags=flags)


def depth_stencil_state(record):
    return dict(depth_enable=record[0], depth_write_mask=record[1], depth_comparison=record[2],
                stencil_enable=record[3], stencil_read_mask=record[4], stencil_write_mask=record[5],
                front_face=list(record[6:10]), back_face=list(record[10:14]), depth_bounds_enable=record[15])


def blend_state(record):
    keys = ("enable", "source", "destination", "op", "source_alpha", "destination_alpha", "op_alpha", "write_mask")
    targets = [dict(zip(keys, target)) for target in struct.iter_unpack("<8B", record[4:])]
    return dict(alpha_to_coverage=record[0], independent_blend=record[1], targets=targets)


def read_effect(reader, stages):
    reader.name()
    reader.word()
    variants = []
    for library in stages:
        indices = struct.iter_unpack("<I", reader.take(reader.word() * 4))
        selected = [library[index] for index, in indices]
        variants.append(selected)
    return read_effect_passes(reader, variants)


def read_effect_passes(reader, variants):
    reader.skip(80)
    rasterizer = [rasterizer_state(record) for record, in reader.counted(16)]
    depth_stencil = [depth_stencil_state(record) for record, in reader.counted(24)]
    blend = [blend_state(record) for record, in reader.counted(68)]
    sampler_count = reader.word()
    reader.skip(36 * sampler_count)
    techniques = list(struct.iter_unpack("<I", reader.take(reader.word() * 4)))
    pass_start = reader.position + 4
    passes = list(reader.counted(24))
    lookup_start = reader.position + 4
    lookup_size = reader.word() * 4
    reader.skip(lookup_size)

    reflection_source = reader.word()
    if reflection_source not in (0, 0x80000000):
        reader.reflection(reflection_source)
    reader.reflection()
    reader.reflection()
    binding_count = reader.word() & 0xFFFFFF
    for _ in range(binding_count):
        marker = reader.word()
        reader.skip(4 if marker == 0x80000000 else marker)
    names = reader.sized()

    for packed, in techniques:
        name_offset = packed & 0x7FFF
        first_pass = (packed >> 15) & 0x1FF
        if name_offset >= len(names) or names.find(b"\0", name_offset) < 0 or first_pass + (packed >> 24) > len(passes):
            raise ValueError("invalid technique name or pass range")

    state_tables = (rasterizer, depth_stencil, blend)
    result = []
    for index, (record,) in enumerate(passes):
        if any(slot > len(library) for slot, library in zip(record[:6], variants)):
            raise ValueError("pass references a missing shader variant")
        shaders = {stage: library[slot - 1] for stage, slot, library in zip(STAGES, record[:6], variants) if slot}
        flags = record[10]
        selections = (record[6], record[7] + ((flags & 0x40) << 2), record[8] + ((flags & 0x80) << 1))
        states = {}
        for bit, (name, choice, table) in enumerate(zip(("rasterizer", "depth_stencil", "blend"), selections, state_tables)):
            if flags & (1 << bit):
                if choice >= len(table):
                    raise ValueError(f"pass references a missing {name} state")
                states[name] = choice
        bindings = [None if value == 255 else value for value in record[18:22]]
        if any(value is not None and value >= binding_count for value in bindings):
            raise ValueError("pass references a missing binding layout")
        usage = struct.unpack_from("<H", record, 16)[0]
        target = pass_start + index * 24 + usage + 4 - lookup_start if usage else None
        if target is not None and not 0 <= target < lookup_size:
            raise ValueError("pass usage link exceeds its lookup table")
        result.append(dict(shaders=shaders, states=states, bindings=bindings, usage_offset=target,
                           stencil_reference=record[9] if flags & 8 else None))
    return dict(technique_count=len(techniques), binding_count=binding_count, sampler_count=sampler_count,
                rasterizer=rasterizer, depth_stencil=depth_stencil, blend=blend, passes=result)


def read_bundles(path, hash64, include_resources=False):
    if not path.stat().st_size:
        raise ValueError("empty shader bundle input")
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        end = 0
        for offset in range(0, len(data) - 3, ALIGNMENT):
            if offset < end or data[offset:offset + 4] != b"SGD5":
                continue
            reader = BundleReader(data, offset + 12, include_resources)
            try:
                binding_count = reader.word()
                for _ in range(binding_count):
                    reader.skip(reader.word())
                reader.reflection()
                stages = [[reader.shader(hash64) for _ in range(reader.word())] for _ in STAGES]
                effects = [read_effect(reader, stages) for _ in range(reader.word())]
            except (IndexError, ValueError, struct.error) as error:
                raise ValueError(f"SGD5 bundle at {offset:#x}, near {reader.position:#x}: {error}") from error
            end = reader.position
            yield dict(offset=offset, end=end, shared_binding_count=binding_count,
                       stages=dict(zip(STAGES, stages)), effects=effects)


def embedded_shader_starts(data, wrapper, floor):
    size, = struct.unpack_from("<I", data, wrapper - 4)
    if size < 176 or size > len(data) - wrapper:
        return
    name_end = wrapper - 5
    for start in range(max(floor + 4, name_end - 256), name_end - 1):
        name_size = data[start]
        if name_size != name_end - start - 1 or data[name_end - 1] != 0:
            continue
        name = data[start + 1:name_end - 1]
        if not name or b"\0" in name:
            continue
        count, = struct.unpack_from("<I", data, start - 4)
        if 0 < count <= (len(data) - start) // EMBEDDED_SHADER_MIN_SIZE:
            yield start - 4


def read_embedded_bundles(path, hash64, *, include_resources=False):
    if not path.stat().st_size:
        raise ValueError("empty embedded shader bundle input")
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        position = 12
        end = 0
        while (wrapper := data.find(NATIVE_SHADER_MAGIC, position)) >= 0:
            position = wrapper + len(NATIVE_SHADER_MAGIC)
            for start in embedded_shader_starts(data, wrapper, end):
                reader = BundleReader(data, start, include_resources=include_resources)
                stages = []
                try:
                    for _ in STAGES:
                        count = reader.word()
                        if count > (len(data) - reader.position) // EMBEDDED_SHADER_MIN_SIZE:
                            raise ValueError("invalid embedded shader array size")
                        stages.append([reader.shader(hash64, metadata_bytes=EMBEDDED_SHADER_METADATA) for _ in range(count)])
                    effect = read_effect_passes(reader, stages)
                except (IndexError, ValueError, struct.error):
                    continue
                if not effect["technique_count"] or not any("vs" in row["shaders"] for row in effect["passes"]):
                    continue
                end = position = reader.position
                yield dict(offset=start, end=end, shared_binding_count=0,
                           stages=dict(zip(STAGES, stages)), effects=[effect], embedded=True)
                break


def summarize(bundles):
    stages = Counter()
    totals = Counter()
    pairs = set()
    for bundle in bundles:
        stages.update({stage: len(records) for stage, records in bundle["stages"].items()})
        totals.update(effects=len(bundle["effects"]), shared_bindings=bundle["shared_binding_count"])
        for effect in bundle["effects"]:
            totals.update(techniques=effect["technique_count"], passes=len(effect["passes"]))
            for row in effect["passes"]:
                shaders = row["shaders"]
                if "cs" in shaders:
                    totals["compute_passes"] += 1
                elif shaders:
                    totals["graphics_passes"] += 1
                    pairs.add(tuple(shaders.get(stage, {}).get("code") for stage in ("vs", "ps")))
                else:
                    totals["shaderless_passes"] += 1
    return dict(totals, bundles=len(bundles), stages=dict(stages), graphics_stage_pairs=len(pairs)), pairs


def compare_manifest(path, stage_pairs):
    modules, pipelines = read_manifest(path)
    matches = 0
    for row in pipelines:
        opts = fields(row.split()[1:])
        pair = tuple(modules[opts[stage]][1]["code"] if stage in opts else None for stage in ("v", "p"))
        matches += pair in stage_pairs
    return dict(graphics_records=len(pipelines), with_shipped_stage_pair=matches)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect shipped graphics recipes in readable SGD5 shader bundles")
    parser.add_argument("asset", nargs="?", type=Path, help="loose bundle or archive with aligned loose bundle data")
    parser.add_argument("--embedded-image", action="append", default=[], type=Path,
                        help="find validated native effect records embedded in an executable")
    parser.add_argument("--output", type=Path, help="write detailed private inventory JSON to a new file")
    parser.add_argument("--reference", action="append", default=[], type=Path, help="Kyty .kpm for shader pair comparison")
    parser.add_argument("--state-coverage", action="store_true", help="compare explicit raster and blend selections with references")
    parser.add_argument("--resource-coverage", action="store_true", help="compare reflected image dimensions with references")
    args = parser.parse_args(argv)
    if args.asset is None and not args.embedded_image:
        parser.error("provide an asset or at least one --embedded-image")
    if args.state_coverage and not args.reference:
        parser.error("--state-coverage requires at least one --reference")
    if args.resource_coverage and not args.reference:
        parser.error("--resource-coverage requires at least one --reference")
    try:
        hash64 = xxhash()
        bundles = (list(read_bundles(args.asset, hash64, include_resources=args.resource_coverage))
                   if args.asset is not None else [])
        for image in args.embedded_image:
            bundles.extend(read_embedded_bundles(image, hash64, include_resources=args.resource_coverage))
        if not bundles:
            raise ValueError("no readable shader bundle or embedded effect found")
        summary, stage_pairs = summarize(bundles)
        if args.reference:
            summary["references"] = [compare_manifest(path, stage_pairs) for path in args.reference]
        if args.state_coverage:
            pass_states = index_pass_states(bundles)
            for result, path in zip(summary["references"], args.reference):
                result["state_coverage"] = compare_manifest_states(path, pass_states)
        if args.resource_coverage:
            declarations = index_declared_dimensions(bundles)
            for result, path in zip(summary["references"], args.reference):
                result["resource_coverage"] = compare_manifest_resources(path, declarations)
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(dict(summary=summary, bundles=bundles), output, separators=(",", ":"))
                output.write("\n")
        print(json.dumps(summary, sort_keys=True))
    except (KeyError, OSError, ValueError) as error:
        print(f"bundle: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
