import struct
from collections import Counter, defaultdict

from session import pipeline_modules, read_manifest


TEXTURE_DIMENSIONS = {
    (0, False): "1D",
    (0, True): "1DArray",
    (1, False): "2D",
    (1, True): "2DArray",
    (2, False): "Cube",
    (2, True): "CubeArray",
    (3, False): "3D",
}
TEXTURE_TYPES = {(2, 0), (12, 1)}


def resource_type(word):
    return word & 15, (word >> 4) & 3


def reflected_texture_count(table, count):
    return sum(resource_type(struct.unpack_from("<H", table, 8 + 24 * index)[0]) in TEXTURE_TYPES
               for index in range(count))


def reflected_resources(table, count):
    result = []
    for index in range(count):
        kind_and_shape, slot, _, _, flags = struct.unpack_from("<H4B", table, 8 + 24 * index)
        kind, category = resource_type(kind_and_shape)
        shape = (kind_and_shape >> 8) & 15
        array = bool(kind_and_shape & 64)
        dimension = TEXTURE_DIMENSIONS.get((shape, array)) if (kind, category) in TEXTURE_TYPES else None
        result.append(dict(serialized_type=kind_and_shape, serialized_slot=slot, flags=flags,
                           kind=kind, category=category, shape=shape, array=array, texture_dimension=dimension))
    return result


def index_declared_dimensions(bundles):
    result = defaultdict(set)
    for bundle in bundles:
        for shaders in bundle["stages"].values():
            for shader in shaders:
                dimensions = {row["texture_dimension"] for row in shader["resources"] if row["texture_dimension"]}
                result[shader["code"]].add(frozenset(dimensions))
    return result


def runtime_image_descriptors(values):
    result = []
    for item in values.get("si", "").split(","):
        if not item or item == "-":
            continue
        fields = {name: value for name, _, value in (token.partition(":") for token in item.split("/"))}
        dimension = fields.get("d")
        if "cube" in fields:
            dimension = "Cube" if dimension == "2DArray" else None
        dynamic_index = any(field in fields for field in ("ir", "t", "heap"))
        result.append((dimension, fields.get("n", "Unknown"), dynamic_index))
    return result


def compare_manifest_resources(path, declarations):
    modules, pipelines = read_manifest(path)
    used = {ident for row in pipelines for ident in pipeline_modules(row)}
    counts = Counter()
    runtime_numeric_classes = Counter()
    runtime_dimensions = Counter()
    undeclared_dimensions = Counter()
    signatures = defaultdict(set)
    for ident in used:
        _, values, _ = modules[ident]
        code = values["code"]
        images = runtime_image_descriptors(values)
        observed = {dimension for dimension, _, _ in images if dimension}
        signature = Counter((dimension, numeric_class) for dimension, numeric_class, _ in images)
        signatures[code].add(frozenset(signature.items()))
        counts["modules_with_runtime_image_descriptors"] += bool(images)
        counts["modules_with_indirect_table_or_heap_images"] += any(indirect for _, _, indirect in images)
        runtime_numeric_classes.update(numeric_class for _, numeric_class, _ in images)
        runtime_dimensions.update(dimension for dimension, _, _ in images if dimension)
        choices = declarations.get(code)
        if "back" in values or choices is None:
            counts["modules_without_comparable_shader_record"] += 1
            continue
        counts["modules_with_comparable_shader_record"] += 1
        combined = set().union(*choices)
        counts["modules_with_reflected_textures"] += bool(combined)
        if observed:
            counts["image_modules_without_reflected_textures"] += not combined
            counts["image_modules_with_any_dimension_agreement"] += bool(observed & combined)
            counts["image_modules_with_all_dimensions_in_one_declaration"] += any(observed <= choice for choice in choices)
            counts["image_modules_with_equal_dimension_set"] += observed in choices
            undeclared_dimensions.update(observed - combined)

    return dict(counts, shader_modules=len(used), runtime_dimensions=dict(runtime_dimensions),
                undeclared_dimensions=dict(undeclared_dimensions), runtime_numeric_classes=dict(runtime_numeric_classes),
                code_digests_with_varying_image_signatures=sum(len(shapes) > 1 for shapes in signatures.values()))
