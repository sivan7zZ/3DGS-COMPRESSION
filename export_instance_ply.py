"""Append aligned instance labels to a standard 3DGS PLY without altering it.

Labels are class-free regions, not semantic ground truth.  The training sidecar
binds its rows to the current Gaussian centers.  Existing output paths are never
overwritten; publication of the completed file is atomic.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement, PlyProperty

from utils.instance_pruning import InstancePruningState, xyz_sha256


LABEL_PROPERTIES = (
    ("instance_id", "i4"),
    ("instance_confidence", "f4"),
    ("instance_views", "u2"),
    ("instance_source", "u1"),
    ("original_id", "u4"),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_array(saved, key, count, minimum, maximum):
    value = saved[key]
    if value.shape != (count,) or value.dtype.kind not in "iu":
        raise ValueError("%s must be a one-dimensional integer array matching PLY rows" % key)
    if np.any(value < minimum) or np.any(value > maximum):
        raise ValueError("%s contains out-of-range values" % key)


def _validate_raw_metadata(path, count, xyz):
    # Validate before conversion: casting uint64 IDs or negative view counts can
    # otherwise silently wrap to the compact output type.
    required = ("format", "original_ids", "instance_ids", "instance_confidence",
                "instance_views", "instance_source", "source_xyz_sha256",
                "current_xyz_sha256", "original_count", "config_json", "policy_json")
    with np.load(str(path), allow_pickle=False) as saved:
        missing = sorted(set(required) - set(saved.files))
        if missing:
            raise ValueError("Instance sidecar is missing: %s" % ", ".join(missing))
        if saved["format"].shape != () or str(saved["format"].item()) != "pwsr-instance-v1":
            raise ValueError("Unsupported instance sidecar format")
        for key in ("source_xyz_sha256", "current_xyz_sha256"):
            if saved[key].shape != () or not re.match(r"^[0-9a-f]{64}$", str(saved[key].item())):
                raise ValueError("%s must be a SHA-256 hex string" % key)
        if str(saved["current_xyz_sha256"].item()) != xyz_sha256(xyz):
            raise ValueError("Instance sidecar current_xyz_sha256 mismatch: Gaussian rows are not aligned")
        original_count = saved["original_count"]
        if original_count.shape != () or original_count.dtype.kind not in "iu" or int(original_count) < count:
            raise ValueError("Invalid original Gaussian count")
        _integer_array(saved, "original_ids", count, 0, min(int(original_count) - 1, np.iinfo(np.uint32).max))
        _integer_array(saved, "instance_ids", count, 0, np.iinfo(np.int32).max)
        _integer_array(saved, "instance_views", count, 0, np.iinfo(np.uint16).max)
        _integer_array(saved, "instance_source", count, 0, 3)
        confidence = saved["instance_confidence"]
        if (confidence.shape != (count,) or confidence.dtype.kind not in "fiu" or
                not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError("Invalid instance confidence")
    state = InstancePruningState.load_resume(path, xyz)
    known_policy_ids = set(int(row["instance_id"]) for row in state.policy)
    if not set(int(value) for value in np.unique(state.instance_ids) if value > 0).issubset(known_policy_ids):
        raise ValueError("Instance IDs are missing from the source instance policy")
    return state


def _atomic_publish(path, writer):
    """Publish a fully written file without replacing any existing path."""
    path = os.path.abspath(str(path))
    if os.path.lexists(path):
        raise FileExistsError("Refusing to overwrite existing output: %s" % path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".instance-export-", dir=parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        # Both names are on the same filesystem.  link is atomic and fails if a
        # competing writer created the destination after the check above.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_instance_ply(ply_path, metadata_path, output_path, report_path=None):
    ply_path = os.path.abspath(str(ply_path))
    metadata_path = os.path.abspath(str(metadata_path))
    output_path = os.path.abspath(str(output_path))
    report_path = os.path.abspath(str(report_path)) if report_path is not None else None
    if output_path in (ply_path, metadata_path):
        raise ValueError("Output must differ from input PLY and metadata")
    if os.path.lexists(output_path):
        raise FileExistsError("Refusing to overwrite existing output: %s" % output_path)
    if report_path is not None:
        if report_path in (ply_path, metadata_path, output_path):
            raise ValueError("Report must have its own output path")
        if os.path.lexists(report_path):
            raise FileExistsError("Refusing to overwrite existing report: %s" % report_path)
    model = PlyData.read(ply_path)
    if "vertex" not in model:
        raise ValueError("PLY has no vertex element")
    vertex = model["vertex"]
    names = vertex.data.dtype.names
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY vertices require x, y, z")
    if set(names).intersection(name for name, _ in LABEL_PROPERTIES):
        raise ValueError("Input PLY already contains an instance export property")
    for name in names:
        values = vertex.data[name]
        if values.dtype.kind in "fc" and not np.isfinite(values).all():
            raise ValueError("Non-finite PLY property: %s" % name)
    xyz = np.column_stack([vertex.data[axis] for axis in ("x", "y", "z")])
    state = _validate_raw_metadata(metadata_path, len(vertex.data), xyz)
    combined = np.empty(len(vertex.data), dtype=vertex.data.dtype.descr + list(LABEL_PROPERTIES))
    for name in names:
        combined[name] = vertex.data[name]
    for target, source in (("instance_id", "instance_ids"),
                           ("instance_confidence", "instance_confidence"),
                           ("instance_views", "instance_views"),
                           ("instance_source", "instance_source"),
                           ("original_id", "original_ids")):
        combined[target] = getattr(state, source)
    labeled_vertex = PlyElement("vertex", list(vertex.properties) +
                                [PlyProperty(name, dtype) for name, dtype in LABEL_PROPERTIES],
                                len(combined), comments=list(vertex.comments))
    labeled_vertex.data = combined
    result = PlyData([labeled_vertex if element.name == "vertex" else element for element in model.elements],
                     text=model.text, byte_order=model.byte_order,
                     comments=list(model.comments), obj_info=list(model.obj_info))
    _atomic_publish(output_path, result.write)
    ids, counts = np.unique(state.instance_ids, return_counts=True)
    report = {
        "format": "pwsr-instance-ply-export-v1",
        "input_ply": ply_path,
        "metadata": metadata_path,
        "output_ply": output_path,
        "gaussian_count": len(combined),
        "source_gaussian_count": state.original_count,
        "labeled_gaussians": int(np.count_nonzero(state.instance_ids)),
        "instance_count": int(np.count_nonzero(ids)),
        "unknown_id": 0,
        "original_properties": list(names),
        "added_properties": [name for name, _ in LABEL_PROPERTIES],
        "source_xyz_sha256": state.source_xyz_sha256,
        "current_xyz_sha256": xyz_sha256(xyz),
        "input_ply_sha256": file_sha256(ply_path),
        "metadata_sha256": file_sha256(metadata_path),
        "output_ply_sha256": file_sha256(output_path),
        "output_bytes": os.path.getsize(output_path),
        "instance_counts": {str(int(key)): int(value) for key, value in zip(ids, counts)},
        "label_meaning": "Class-free automatically associated regions; IDs are not semantic classes.",
        "source_codes": {"0": "unknown", "1": "direct multiview evidence", "2": "spatial propagation", "3": "reserved"},
    }
    if report_path is not None:
        def write_report(temporary):
            with open(temporary, "w") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        _atomic_publish(report_path, write_report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, help="Standard compressed 3DGS PLY")
    parser.add_argument("--metadata", required=True, help="Aligned pwsr-instance-v1 NPZ sidecar")
    parser.add_argument("--output", required=True, help="New instance-labeled PLY; must not already exist")
    parser.add_argument("--report", help="Optional new JSON report with hashes and instance counts")
    args = parser.parse_args()
    report = export_instance_ply(args.ply, args.metadata, args.output, args.report)
    print(json.dumps({key: report[key] for key in ("output_ply", "gaussian_count", "instance_count", "output_bytes")}, indent=2))


if __name__ == "__main__":
    main()
