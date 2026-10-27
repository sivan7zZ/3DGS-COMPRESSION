import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from export_instance_ply import export_instance_ply, _atomic_publish, LABEL_PROPERTIES
from utils.instance_pruning import InstancePruningState, xyz_sha256


class InstancePlyExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ply = self.root / "standard.ply"
        self.metadata = self.root / "instances.npz"
        self.output = self.root / "labeled.ply"
        self.report = self.root / "report.json"

    def tearDown(self):
        self.directory.cleanup()

    def make_inputs(self, text=False, byte_order="<"):
        xyz = np.arange(60, dtype=np.float32).reshape(20, 3) / 7
        ids = np.r_[np.zeros(4), np.ones(8), np.full(8, 2)].astype(np.int32)
        priors = self.root / "raw.npz"
        np.savez(priors, instance_ids=ids,
                 instance_confidence=np.where(ids > 0, .8, 0).astype(np.float32),
                 instance_views=np.where(ids > 0, 3, 0).astype(np.uint16),
                 instance_source=np.where(ids > 0, 1, 0).astype(np.uint8),
                 source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
        state = InstancePruningState.load(priors, xyz)
        keep = np.isin(np.arange(20), [0, 3, 5, 9, 12, 15, 18])
        state.subset_(keep)
        xyz = xyz[keep]
        data = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("f_dc_0", "f4"), ("f_rest_0", "f4"),
                                         ("opacity", "f4"), ("custom_int", "i2"),
                                         ("custom_double", "f8")])
        for axis, values in zip(("x", "y", "z"), xyz.T):
            data[axis] = values
        data["opacity"] = [-10., -1., -0., 0., 1., 3., 9.]
        data["custom_int"] = np.arange(7) - 3
        data["custom_double"] = np.pi + np.arange(7) * 1.e-8
        faces = np.empty(1, dtype=[("vertex_indices", "O")])
        faces["vertex_indices"][0] = np.asarray([0, 1, 2], dtype=np.int32)
        PlyData([PlyElement.describe(data, "vertex", comments=["vertex comment"]),
                 PlyElement.describe(faces, "face")], text=text, byte_order=byte_order,
                comments=["original model"], obj_info=["camera source"]).write(str(self.ply))
        state.save(self.metadata, xyz)
        return data, state

    def mutate_metadata(self, **updates):
        with np.load(str(self.metadata), allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        values.update(updates)
        np.savez(str(self.metadata), **values)

    def test_preserves_every_original_property_and_other_elements(self):
        for text, byte_order in ((False, "<"), (False, ">"), (True, "=")):
            with self.subTest(text=text, byte_order=byte_order):
                data, state = self.make_inputs(text, byte_order)
                report = export_instance_ply(self.ply, self.metadata, self.output, self.report)
                before, after = PlyData.read(str(self.ply)), PlyData.read(str(self.output))
                self.assertEqual(after["vertex"].data.dtype.names,
                                 data.dtype.names + tuple(name for name, _ in LABEL_PROPERTIES))
                for name in data.dtype.names:
                    np.testing.assert_array_equal(after["vertex"].data[name], before["vertex"].data[name])
                    self.assertEqual(after["vertex"].data[name].dtype, before["vertex"].data[name].dtype)
                    self.assertEqual(after["vertex"].data[name].tobytes(), before["vertex"].data[name].tobytes())
                np.testing.assert_array_equal(after["vertex"].data["instance_id"], state.instance_ids)
                np.testing.assert_array_equal(after["vertex"].data["instance_confidence"], state.instance_confidence)
                np.testing.assert_array_equal(after["vertex"].data["instance_views"], state.instance_views)
                np.testing.assert_array_equal(after["vertex"].data["instance_source"], state.instance_source)
                np.testing.assert_array_equal(after["vertex"].data["original_id"], state.original_ids)
                np.testing.assert_array_equal(after["face"].data["vertex_indices"][0], [0, 1, 2])
                self.assertEqual(after.comments, before.comments)
                self.assertEqual(after.obj_info, before.obj_info)
                self.assertEqual(after["vertex"].comments, before["vertex"].comments)
                self.assertEqual(after.text, before.text)
                self.assertEqual(after.byte_order, before.byte_order)
                self.assertEqual(report["gaussian_count"], 7)
                self.assertEqual(report["source_gaussian_count"], 20)
                self.assertEqual(report["instance_count"], 2)
                self.assertEqual(report["labeled_gaussians"], 5)
                self.assertEqual(len(report["output_ply_sha256"]), 64)
                self.assertEqual(json.loads(self.report.read_text()), report)
                self.output.unlink()
                self.report.unlink()

    def test_row_permutation_or_center_change_is_rejected(self):
        self.make_inputs()
        model = PlyData.read(str(self.ply), mmap=False)
        model["vertex"].data = model["vertex"].data[::-1].copy()
        model.write(str(self.ply))
        with self.assertRaisesRegex(ValueError, "current_xyz_sha256 mismatch"):
            export_instance_ply(self.ply, self.metadata, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_malformed_or_overflowing_metadata_before_casting(self):
        changes = (
            {"instance_ids": np.full(7, 2 ** 32 + 1, dtype=np.uint64)},
            {"instance_ids": np.ones(7, dtype=np.float32)},
            {"instance_views": np.full(7, -1, dtype=np.int32)},
            {"instance_views": np.full(7, 65536, dtype=np.uint32)},
            {"instance_source": np.full(7, 4, dtype=np.uint8)},
            {"instance_confidence": np.full(7, np.nan, dtype=np.float32)},
            {"instance_confidence": np.full(7, 1.1, dtype=np.float32)},
            {"instance_ids": np.ones(6, dtype=np.int32)},
            {"original_ids": np.zeros(7, dtype=np.int64)},
            {"original_ids": np.arange(7, dtype=np.int64) + 20},
            {"current_xyz_sha256": np.asarray("not-a-sha")},
            {"format": np.asarray("unversioned")},
            {"instance_ids": np.full(7, 99, dtype=np.int32)},
        )
        for update in changes:
            with self.subTest(update=list(update)):
                self.make_inputs()
                self.mutate_metadata(**update)
                with self.assertRaises(ValueError):
                    export_instance_ply(self.ply, self.metadata, self.output)
                self.assertFalse(self.output.exists())

    def test_refuses_non_finite_original_render_parameters(self):
        self.make_inputs()
        model = PlyData.read(str(self.ply), mmap=False)
        model["vertex"].data["opacity"][2] = np.inf
        model.write(str(self.ply))
        with self.assertRaisesRegex(ValueError, "Non-finite PLY property: opacity"):
            export_instance_ply(self.ply, self.metadata, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_report_and_input_are_never_overwritten(self):
        self.make_inputs()
        original = self.ply.read_bytes()
        with self.assertRaises(ValueError):
            export_instance_ply(self.ply, self.metadata, self.ply)
        self.assertEqual(self.ply.read_bytes(), original)
        self.output.write_bytes(b"previous deliverable")
        with self.assertRaises(FileExistsError):
            export_instance_ply(self.ply, self.metadata, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous deliverable")
        self.output.unlink()
        self.report.write_text("previous report")
        with self.assertRaises(FileExistsError):
            export_instance_ply(self.ply, self.metadata, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.report.read_text(), "previous report")

    def test_refuses_duplicate_instance_properties(self):
        self.make_inputs()
        export_instance_ply(self.ply, self.metadata, self.output)
        with self.assertRaisesRegex(ValueError, "already contains"):
            export_instance_ply(self.output, self.metadata, self.root / "twice.ply")

    def test_interrupted_writer_and_competing_writer_never_publish_partial_or_replace(self):
        def failing_writer(path):
            Path(path).write_bytes(b"incomplete")
            raise RuntimeError("simulated failure")
        with self.assertRaises(RuntimeError):
            _atomic_publish(self.output, failing_writer)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".instance-export-*")))

        def competing_writer(path):
            Path(path).write_bytes(b"complete new file")
            self.output.write_bytes(b"concurrent deliverable")
        with self.assertRaises(FileExistsError):
            _atomic_publish(self.output, competing_writer)
        self.assertEqual(self.output.read_bytes(), b"concurrent deliverable")
        self.assertFalse(list(self.root.glob(".instance-export-*")))


if __name__ == "__main__":
    unittest.main()
