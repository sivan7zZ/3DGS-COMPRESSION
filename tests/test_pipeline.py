import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_graph_uses_pruned_student_and_original_teacher_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data with spaces"
            data.mkdir()
            ply = root / "initial model.ply"
            ply.touch()
            teacher = root / "original chkpnt30000.pth"
            teacher.touch()
            with patch.object(run_pipeline, "ROOT", root):
                output, plan = run_pipeline.build_plan({"source_path": str(data), "input_ply": str(ply),
                                                        "teacher_checkpoint": str(teacher)})
            self.assertEqual([item["name"] for item in plan], ["prune", "distill", "render_stage3", "render_sh2", "metrics"])
            command = plan[1]["command"]
            checkpoint = str(output / "stage3" / "chkpnt25199.pth")
            self.assertEqual(command[command.index("--start_checkpoint")+1], checkpoint)
            self.assertEqual(command[command.index("--teacher_model")+1], str(teacher))
            self.assertNotEqual(command[command.index("--teacher_model")+1], checkpoint)
            self.assertEqual(command[command.index("--distill_steps")+1], "5000")
            self.assertNotIn("--start_pointcloud", command)
            self.assertIn(str(data), command)
            self.assertFalse(output.exists())

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "initial.ply"
            ply.touch()
            teacher = root / "teacher.pth"
            teacher.touch()
            config = root / "config.json"
            with config.open("w") as stream:
                json.dump({"source_path": str(root), "input_ply": str(ply),
                           "teacher_checkpoint": str(teacher)}, stream)
            with patch.object(run_pipeline, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                run_pipeline.main(["--config", str(config), "--dry_run"])
            self.assertFalse((root / "output").exists())

    def test_reject_external_and_historical_output_and_ambiguous_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"source_path": str(root), "input_ply": __file__}
            with patch.object(run_pipeline, "ROOT", root):
                for path in ("/tmp", "output", "output/city2_stage3", "output/city2_stage3_sh2/nested"):
                    with self.assertRaises(ValueError):
                        run_pipeline.build_plan(dict(config, output_root=path))
                with self.assertRaises(ValueError):
                    run_pipeline.build_plan(dict(config, input_checkpoint=__file__))

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "point_cloud.ply").touch()
            with self.assertRaises(FileExistsError):
                run_pipeline.check_output_collisions([{"name": "distill", "output": directory}])

    def test_distillation_requires_explicit_teacher_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "initial.ply"
            ply.touch()
            with patch.object(run_pipeline, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "teacher_checkpoint"):
                    run_pipeline.build_plan({"source_path": str(root), "input_ply": str(ply)})

if __name__ == "__main__":
    unittest.main()
