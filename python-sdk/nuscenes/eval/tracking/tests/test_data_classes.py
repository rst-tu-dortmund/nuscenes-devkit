import unittest

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig


class TestTrackingDataClasses(unittest.TestCase):

    def setUp(self):
        config_factory('tracking_nips_2019')

    def test_tracking_box_backward_compat_without_covariance(self):
        payload = {
            'sample_token': 'sample',
            'translation': [0.0, 0.0, 0.0],
            'size': [1.0, 1.0, 1.0],
            'rotation': [1.0, 0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0],
            'tracking_id': 'track-1',
            'tracking_name': 'car',
            'tracking_score': 0.5,
        }

        box = TrackingBox.deserialize(payload)
        self.assertIsNone(box.covariance)
        self.assertIsNone(box.state_dim)

    def test_tracking_box_covariance_roundtrip(self):
        payload = {
            'sample_token': 'sample',
            'translation': [0.0, 0.0, 0.0],
            'size': [1.0, 1.0, 1.0],
            'rotation': [1.0, 0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0],
            'tracking_id': 'track-1',
            'tracking_name': 'car',
            'tracking_score': 0.5,
            'covariance': [[1.0, 0.0], [0.0, 1.0]],
            'state_dim': 2,
        }

        box = TrackingBox.deserialize(payload)
        self.assertEqual(box.covariance, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(box.state_dim, 2)
        self.assertEqual(box.serialize()['covariance'], [[1.0, 0.0], [0.0, 1.0]])

    def test_tracking_config_nees_state_indices_roundtrip(self):
        cfg = config_factory('tracking_nips_2019')
        serialized = cfg.serialize()
        serialized['nees_state_indices'] = [0, 1, 8]

        restored = TrackingConfig.deserialize(serialized)
        self.assertEqual(restored.nees_state_indices, [0, 1, 8])


if __name__ == '__main__':
    unittest.main()
