import copy
import collections
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from evaluate_boundary_weighted_semantics import (
    NON_TEST, TEST, build_protocol, inspect_annotation, validate_folds,
    validated_models, verify_metadata, verify_predictions,
)
from render_boundary_semantic_views import select_auxiliary_cameras
from finalize_boundary_weighted_results import validate_aggregation
from utils.semantic_comparison_metrics import evaluate_semantics


class BoundarySemanticProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'frames': [{'frame': frame} for frame in sorted(NON_TEST + TEST)]}
        self.historical = dict(checkpoint='/frozen/model.pt', checkpoint_sha256='fixed', selected_step=1500,
            metadata=dict(train_frames=NON_TEST, test_frames_used_for_training_or_selection=[]))

    def test_out_of_fold_coverage_and_independent_test_isolation(self):
        protocol = build_protocol(self.manifest, self.historical)
        folds = protocol['folds']
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertFalse(set(fold['train_frames']) & set(TEST))
            self.assertFalse(set(fold['train_frames']) & set(fold['predict_frames']))
            self.assertEqual(fold['validation_frames'], [])
        self.assertEqual(sorted(frame for fold in folds for frame in fold['predict_frames']), sorted(NON_TEST + TEST))
        self.assertEqual(len(protocol['aggregate_groups']['auxiliary_training_views_8']), 8)
        self.assertEqual(protocol['aggregate_groups']['independent_test_2'], TEST)

    def test_auxiliary_fit_uses_six_without_evaluation_selection(self):
        folds = build_protocol(self.manifest, self.historical)['folds'][1:]
        for fold in folds:
            self.assertEqual(len(fold['train_frames']), 6)
            self.assertEqual(len(fold['predict_frames']), 2)
            self.assertEqual(fold['steps'], 1500)
            self.assertEqual(fold['fitting'], 'fixed_steps_no_evaluation_frame_selection')

    def test_heldout_mode_requires_no_new_fit(self):
        protocol = build_protocol(self.manifest, self.historical, 'city_heldout')
        self.assertEqual(len(protocol['folds']), 1)
        self.assertEqual(protocol['aggregate_groups'], {'independent_test_2': TEST})

    def test_rejects_historical_evaluator_that_fitted_test_labels(self):
        self.historical['metadata']['train_frames'] = NON_TEST[:-1] + TEST[:1]
        with self.assertRaises(ValueError):
            build_protocol(self.manifest, self.historical)

    def test_rejects_duplicate_missing_or_leaked_fold_frames(self):
        folds = build_protocol(self.manifest, self.historical)['folds']
        broken = copy.deepcopy(folds)
        broken[1]['predict_frames'] = broken[2]['predict_frames']
        with self.assertRaises(ValueError):
            validate_folds(broken, 'city_oof')
        broken = copy.deepcopy(folds)
        broken[1]['train_frames'][0] = broken[1]['predict_frames'][0]
        with self.assertRaises(ValueError):
            validate_folds(broken, 'city_oof')

    def test_same_basename_is_not_image_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'frame_000'
            folder.mkdir()
            Image.new('RGB', (7, 5), 'red').save(folder / 'frame.png')
            Image.new('L', (7, 5), 255).save(folder / 'Building.png')
            Image.new('L', (7, 5), 255).save(folder / 'mask.png')
            (folder / 'frame.json').write_text(json.dumps(dict(shapes=[{'label': 'Building'}], imageWidth=7, imageHeight=5)))
            row = inspect_annotation(folder, {'frame_000': dict(path='other', size_wh=[7, 5], sha256='different', rgb_pixel_sha256='different')})
            self.assertEqual(row['matching_model_input_frames'], [])
            self.assertFalse(row['same_name_pixels_identical'])
            self.assertFalse(row['same_name_file_identical'])

    def test_metadata_cannot_use_original_manifest_for_new_fold(self):
        protocol = build_protocol(self.manifest, self.historical)
        protocol.update(data_manifest_sha256='new', historical_data_manifest_sha256='old')
        fold = protocol['folds'][1]
        metadata = dict(data_manifest_sha256='new', train_frames=fold['train_frames'],
                        validation_frames=[], selected_step=1500)
        verify_metadata(metadata, fold, protocol)
        metadata['data_manifest_sha256'] = 'old'
        with self.assertRaises(ValueError):
            verify_metadata(metadata, fold, protocol)

    def test_metadata_rejects_evaluation_frame_model_selection(self):
        protocol = build_protocol(self.manifest, self.historical)
        protocol.update(data_manifest_sha256='new', historical_data_manifest_sha256='old')
        fold = protocol['folds'][1]
        metadata = dict(data_manifest_sha256='new', train_frames=fold['train_frames'],
                        validation_frames=fold['predict_frames'], selected_step=1500)
        with self.assertRaises(ValueError):
            verify_metadata(metadata, fold, protocol)

    def test_model_registry_rejects_unsafe_and_duplicate_ids(self):
        for models in ([{'id': '../outside'}], [{'id': 'one'}, {'id': 'one'}], [{'id': 'GT'}]):
            with self.assertRaises(ValueError):
                validated_models({'models': models})

    def test_prediction_ids_and_checkpoint_cannot_be_silently_swapped(self):
        inputs = {'images': [{'id': 'one/frame_000'}]}
        for prediction in [dict(checkpoint_sha256='wrong', predictions=[]),
                           dict(checkpoint_sha256='correct', predictions=[]),
                           dict(checkpoint_sha256='correct', predictions=[{'id': 'one/frame_000'}, {'id': 'one/frame_000'}])]:
            with self.assertRaises(ValueError):
                verify_predictions(prediction, inputs, 'correct')

    def test_auxiliary_camera_selection_uses_names_and_rejects_test_views(self):
        Camera = collections.namedtuple('Camera', 'image_name is_test')
        cameras = [Camera(name + '.png', False) for name in reversed(NON_TEST)]
        selected = select_auxiliary_cameras(cameras)
        self.assertEqual([Path(camera.image_name).stem for camera in selected], NON_TEST)
        cameras[0] = Camera(cameras[0].image_name, True)
        with self.assertRaises(ValueError):
            select_auxiliary_cameras(cameras)


class BoundarySemanticAggregationTests(unittest.TestCase):
    def setUp(self):
        first = np.ones((24, 32), dtype=np.uint8)
        first[:, 16:] = 2
        second = np.ones((24, 32), dtype=np.uint8)
        second[8:, :] = 2
        prediction = second.copy()
        prediction[8:12, :12] = 1
        self.semantic = {'scopes': {
            'independent_test_2': {'results': {'model': {'metrics': evaluate_semantics([first], [first])}}},
            'auxiliary_training_views_8': {'results': {'model': {'metrics': evaluate_semantics([prediction], [second])}}},
            'all_annotated_10': {'results': {'model': {'metrics': evaluate_semantics([first, prediction], [first, second])}}},
        }}

    def test_disjoint_scopes_recompute_from_counts(self):
        self.assertEqual(validate_aggregation(self.semantic), 3)
        # Main and auxiliary have different class supports: averaging their
        # already-computed scores must not replace the count-based aggregate.
        total = self.semantic['scopes']['all_annotated_10']['results']['model']['metrics']
        total['miou'] = -1
        with self.assertRaises(ValueError):
            validate_aggregation(self.semantic)

    def test_consistent_but_wrong_total_still_fails_disjoint_identity(self):
        # Replacing total by a self-consistent main-only metric would pass
        # internal score checks, but must fail the independent 2+8 identity.
        self.semantic['scopes']['all_annotated_10']['results']['model']['metrics'] = copy.deepcopy(
            self.semantic['scopes']['independent_test_2']['results']['model']['metrics'])
        with self.assertRaises(ValueError):
            validate_aggregation(self.semantic)


if __name__ == '__main__':
    unittest.main()
