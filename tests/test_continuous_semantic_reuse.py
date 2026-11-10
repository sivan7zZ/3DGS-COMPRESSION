import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import evaluate_continuous_boundary_semantics as reuse
from evaluate_boundary_weighted_semantics import NON_TEST, TEST, build_protocol
from utils.city_semantic_data import sha256_file


class FrozenContinuousSemanticProtocolTests(unittest.TestCase):
    def setUp(self):
        manifest = dict(frames=[dict(frame=name) for name in sorted(NON_TEST + TEST)])
        historical = dict(checkpoint='/historical/main.pt', checkpoint_sha256='main-checkpoint', selected_step=1500,
            metadata=dict(train_frames=NON_TEST, test_frames_used_for_training_or_selection=[]))
        self.protocol = build_protocol(manifest, historical)
        self.protocol.update(data_manifest_sha256='exact-old-auxiliary-manifest',
                             historical_data_manifest_sha256='original-main-manifest')
        for index, fold in enumerate(self.protocol['folds']):
            fold.update(checkpoint='/frozen/%d.pt' % index,
                expected_data_manifest_sha256=('original-main-manifest' if index == 0 else 'exact-old-auxiliary-manifest'))

    def metadata(self, fold):
        return dict(data_manifest_sha256=fold['expected_data_manifest_sha256'],
                    train_frames=list(fold['train_frames']), validation_frames=[], selected_step=1500,
                    test_frames_used_for_training_or_selection=[])

    def test_each_frozen_fold_uses_its_original_registered_manifest(self):
        for index, fold in enumerate(self.protocol['folds']):
            with self.subTest(fold=fold['id']):
                metadata = self.metadata(fold)
                reuse.verify_metadata(metadata, fold, self.protocol)
                metadata['data_manifest_sha256'] = ('exact-old-auxiliary-manifest' if index == 0 else 'original-main-manifest')
                with self.assertRaises(ValueError):
                    reuse.verify_metadata(metadata, fold, self.protocol)

    def test_missing_expected_manifest_or_test_label_selection_is_rejected(self):
        fold = self.protocol['folds'][1]
        metadata = self.metadata(fold)
        broken = dict(fold)
        broken.pop('expected_data_manifest_sha256')
        with self.assertRaises(ValueError):
            reuse.verify_metadata(metadata, broken, self.protocol)
        for mode in ('predicted_frame_fit', 'main_frame_fit', 'evaluation_selection', 'test_selection_record'):
            metadata = self.metadata(fold)
            if mode == 'predicted_frame_fit':
                metadata['train_frames'][0] = fold['predict_frames'][0]
            elif mode == 'main_frame_fit':
                metadata['train_frames'][0] = TEST[0]
            elif mode == 'evaluation_selection':
                metadata['validation_frames'] = list(fold['predict_frames'])
            else:
                metadata['test_frames_used_for_training_or_selection'] = TEST[:1]
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    reuse.verify_metadata(metadata, fold, self.protocol)


class FrozenContinuousPredictionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch = mock.patch.object(reuse, 'PREVIOUS', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def write_fold(self, name, checkpoint_bytes=b'fixed-weight-A', image_bytes=b'input-RGB',
                   prediction_bytes=b'prediction-A', image_id='old-name/frame_000'):
        folder = self.root / 'comparison/semantic/folds' / name
        folder.mkdir(parents=True)
        checkpoint = folder / 'model.pt'
        checkpoint.write_bytes(checkpoint_bytes)
        image = folder / 'frame_000.png'
        image.write_bytes(image_bytes)
        prediction = folder / 'prediction.png'
        prediction.write_bytes(prediction_bytes)
        row = dict(id=image_id, image_path=str(image), image_sha256=sha256_file(image),
                   prediction_path=str(prediction), prediction_sha256=sha256_file(prediction))
        manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), predictions=[row])
        path = folder / 'predictions_manifest.json'
        path.write_text(json.dumps(manifest))
        semantic = self.root / 'comparison/semantic'
        protocol_path, metrics_path = semantic / 'protocol.json', semantic / 'metrics.json'
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else dict(folds=[])
        protocol['folds'].append(dict(id=name, checkpoint_sha256=manifest['checkpoint_sha256'],
            original_predictions_manifest=str(path), original_predictions_manifest_sha256=sha256_file(path)))
        metrics_path.write_text(json.dumps(dict(status='complete', folds={
            fold['id']: dict(predictions_manifest_sha256=fold['original_predictions_manifest_sha256'])
            for fold in protocol['folds']})))
        protocol.update(reused_metrics=str(metrics_path), reused_metrics_sha256=sha256_file(metrics_path))
        protocol_path.write_text(json.dumps(protocol))
        return manifest, row, path

    def test_cache_requires_checkpoint_and_input_content_together(self):
        first, a, _ = self.write_fold('first')
        second, b, _ = self.write_fold('second', checkpoint_bytes=b'fixed-weight-B', prediction_bytes=b'prediction-B')
        self.assertEqual(a['image_sha256'], b['image_sha256'])
        self.assertNotEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])
        a_cache = reuse.prediction_cache(first['checkpoint_sha256'], self.root)
        b_cache = reuse.prediction_cache(second['checkpoint_sha256'], self.root)
        self.assertEqual(a_cache[a['image_sha256']]['prediction_sha256'], a['prediction_sha256'])
        self.assertEqual(b_cache[b['image_sha256']]['prediction_sha256'], b['prediction_sha256'])
        self.assertEqual(reuse.prediction_cache('unrelated-checkpoint', self.root), {})

    def test_shared_frame_basename_does_not_imply_shared_rgb_identity(self):
        first, a, _ = self.write_fold('first')
        _, b, _ = self.write_fold('second', image_bytes=b'different-RGB', prediction_bytes=b'different-prediction')
        cache = reuse.prediction_cache(first['checkpoint_sha256'], self.root)
        self.assertEqual(len(cache), 2)
        self.assertEqual(Path(a['image_path']).name, Path(b['image_path']).name)
        self.assertNotEqual(a['image_sha256'], b['image_sha256'])

    def test_cached_checkpoint_rgb_or_prediction_tampering_is_rejected(self):
        manifest, row, _ = self.write_fold('first')
        paths = [Path(manifest['checkpoint']), Path(row['image_path']), Path(row['prediction_path'])]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(b'changed-after-evaluation')
            with self.subTest(path=path.name):
                with self.assertRaises(ValueError):
                    reuse.prediction_cache(manifest['checkpoint_sha256'], self.root)
            path.write_bytes(original)

    def test_conflicting_predictions_for_identical_checkpoint_and_rgb_are_rejected(self):
        manifest, _, _ = self.write_fold('first')
        self.write_fold('second', prediction_bytes=b'conflicting-prediction')
        with self.assertRaises(ValueError):
            reuse.prediction_cache(manifest['checkpoint_sha256'], self.root)

    def test_identical_duplicates_allow_reuse_without_a_model_name_key(self):
        manifest, row, _ = self.write_fold('first')
        _, second, _ = self.write_fold('second', image_id='renamed-model/frame_000')
        cache = reuse.prediction_cache(manifest['checkpoint_sha256'], self.root)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache[row['image_sha256']]['prediction_sha256'], second['prediction_sha256'])

    def test_joint_prediction_and_manifest_tampering_cannot_rewrite_frozen_history(self):
        manifest, row, path = self.write_fold('first')
        prediction = Path(row['prediction_path'])
        prediction.write_bytes(b'changed-prediction-with-updated-self-reported-hash')
        manifest['predictions'][0]['prediction_sha256'] = sha256_file(prediction)
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            reuse.prediction_cache(manifest['checkpoint_sha256'], self.root)

    def test_original_completed_metrics_receipt_is_pinned(self):
        manifest, _, _ = self.write_fold('first')
        (self.root / 'comparison/semantic/metrics.json').write_text('{"status":"complete","changed":true}')
        with self.assertRaises(ValueError):
            reuse.prediction_cache(manifest['checkpoint_sha256'], self.root)


if __name__ == '__main__':
    unittest.main()
