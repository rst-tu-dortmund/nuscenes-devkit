# Modified by Timo Osterburg and Stefan Schütte (TU Dortmund University), 2026, for CANMOT.

import copy
import unittest
import warnings
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.algo import TrackingEvaluation, categorize_mahalanobis_distances, pearson_chi2_two_category
from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingBox
from nuscenes.eval.tracking.loaders import interpolate_tracks
from nuscenes.eval.tracking.tests.scenarios import get_scenarios


class TestAlgo(unittest.TestCase):

    def test_mahalanobis_threshold_categorization(self):
        values = np.array([1.0, 2.0, 5.0, 7.0], dtype=float)
        stats = categorize_mahalanobis_distances(values, dof=2, alpha_maha=0.05)

        self.assertAlmostEqual(stats['maha_threshold'], 5.991464547107979)
        self.assertEqual(stats['count_inside'], 3.0)
        self.assertEqual(stats['count_outside'], 1.0)
        self.assertEqual(stats['pct_inside'], 75.0)
        self.assertEqual(stats['pct_outside'], 25.0)

    def test_pearson_chi2_two_category_decision(self):
        non_significant = pearson_chi2_two_category(90.0, 10.0, alpha_maha=0.05, alpha_chi2=0.01)
        significant = pearson_chi2_two_category(80.0, 20.0, alpha_maha=0.05, alpha_chi2=0.01)

        self.assertAlmostEqual(non_significant['chi2_statistic'], 5.263157894736842)
        self.assertAlmostEqual(non_significant['chi2_critical'], 6.6348966010212145)
        self.assertEqual(non_significant['chi2_significant'], 0.0)

        self.assertAlmostEqual(significant['chi2_statistic'], 47.368421052631575)
        self.assertAlmostEqual(significant['chi2_critical'], 6.6348966010212145)
        self.assertEqual(significant['chi2_significant'], 1.0)

    @staticmethod
    def single_scene() -> Tuple[str, Dict[str, Dict[int, List[TrackingBox]]]]:
        class_name = 'car'
        box = TrackingBox(translation=(0, 0, 0), tracking_id='ta', tracking_name=class_name,
                          tracking_score=0.5)
        timestamp_boxes_gt = {
            0: [copy.deepcopy(box)],
            1: [copy.deepcopy(box)],
            2: [copy.deepcopy(box)],
            3: [copy.deepcopy(box)]
        }
        timestamp_boxes_gt[0][0].sample_token = 'a'
        timestamp_boxes_gt[1][0].sample_token = 'b'
        timestamp_boxes_gt[2][0].sample_token = 'c'
        timestamp_boxes_gt[3][0].sample_token = 'd'
        tracks_gt = {'scene-1': timestamp_boxes_gt}

        return class_name, tracks_gt

    def test_gt_submission(self):
        """ Test with GT submission. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one prediction.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fn == 0)
        assert np.all(md.fp == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_empty_submission(self):
        """ Test a submission with no predictions. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove all predictions.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        for timestamp, box in timestamp_boxes_pred.items():
            timestamp_boxes_pred[timestamp] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.mota == 0)
        assert np.all(md.motar == 0)
        assert np.all(np.isnan(md.recall_hypo))
        assert np.all(md.tp == 0)
        assert np.all(md.fn == 4)
        assert np.all(np.isnan(md.fp))  # FP/Frag/IDS are nan as we there were no predictions.
        assert np.all(md.lgd == 20)
        assert np.all(md.tid == 20)
        assert np.all(np.isnan(md.frag))
        assert np.all(np.isnan(md.ids))

    def test_drop_prediction(self):
        """ Drop one prediction from the GT submission. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        timestamp_boxes_pred[1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # Recall values above 0.75 (3/4 correct) are not achieved and therefore nan.
        first_achieved = np.where(md.recall_hypo <= 0.75)[0][0]
        assert np.all(np.isnan(md.confidence[:first_achieved]))
        assert md.tp[first_achieved] == 3
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 1
        assert md.lgd[first_achieved] == 0.5
        assert md.tid[first_achieved] == 0
        assert md.frag[first_achieved] == 1
        assert md.ids[first_achieved] == 0

    def test_drop_prediction_multiple(self):
        """  Drop the first three predictions from the GT submission. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        timestamp_boxes_pred[0] = []
        timestamp_boxes_pred[1] = []
        timestamp_boxes_pred[2] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        # Recall values above 0.75 (3/4 correct) are not achieved and therefore nan.
        first_achieved = np.where(md.recall_hypo <= 0.25)[0][0]
        assert np.all(np.isnan(md.confidence[:first_achieved]))
        assert md.tp[first_achieved] == 1
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 3
        assert md.lgd[first_achieved] == 3 * 0.5
        assert md.tid[first_achieved] == 3 * 0.5
        assert md.frag[first_achieved] == 0
        assert md.ids[first_achieved] == 0

    def test_identity_switch(self):
        """ Change the tracking_id of one frame from the GT submission. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one predicted box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        timestamp_boxes_pred[2][0].tracking_id = 'tb'
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        first_achieved = np.where(md.recall_hypo <= 0.5)[0][0]
        assert md.tp[first_achieved] == 2
        assert md.fp[first_achieved] == 0
        assert md.fn[first_achieved] == 0
        assert md.lgd[first_achieved] == 0
        assert md.tid[first_achieved] == 0
        assert md.frag[first_achieved] == 0
        assert md.ids[first_achieved] == 2  # One wrong id leads to 2 identity switches.

    def test_drop_gt(self):
        """ Drop one box from the GT. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one GT box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_gt['scene-1'][1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 3)
        assert np.all(md.fp == 1)
        assert np.all(md.fn == 0)
        assert np.all(md.lgd == 0.5)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_drop_gt_interpolate(self):
        """ Drop one box from the GT and interpolate the results to fill in that box. """

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        # Define inputs.
        class_name, tracks_gt = TestAlgo.single_scene()
        verbose = False

        # Remove one GT box.
        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        tracks_gt['scene-1'][1] = []
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        # Interpolate to "restore" dropped GT.
        tracks_gt['scene-1'] = interpolate_tracks(defaultdict(list, tracks_gt['scene-1']))

        # Accumulate metrics.
        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose)
        md = ev.accumulate()

        # Check outputs.
        assert np.all(md.tp == 4)
        assert np.all(md.fp == 0)
        assert np.all(md.fn == 0)
        assert np.all(md.lgd == 0)
        assert np.all(md.tid == 0)
        assert np.all(md.frag == 0)
        assert np.all(md.ids == 0)

    def test_tp_error_and_nees_single_match(self):
        cfg = config_factory('tracking_nips_2019')
        class_name, tracks_gt = TestAlgo.single_scene()

        for _, boxes in tracks_gt['scene-1'].items():
            for box in boxes:
                box.size = (1.0, 1.0, 1.0)

        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        for timestamp, boxes in timestamp_boxes_pred.items():
            for box in boxes:
                box.rotation = Quaternion(axis=(0, 0, 1), radians=0.0).elements
                box.covariance = np.eye(9).tolist()
                box.state_dim = 9
        timestamp_boxes_pred[0][0].translation = (1.0, 0.0, 0.0)
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=False)
        md = ev.accumulate()

        self.assertAlmostEqual(float(np.nanmin(md.tp_translation_error_mean)), 0.25)
        self.assertAlmostEqual(float(np.nanmin(md.tp_scale_error_mean)), 0.0)
        self.assertAlmostEqual(float(np.nanmin(md.tp_velocity_error_mean)), 0.0)
        self.assertAlmostEqual(float(np.nanmin(md.tp_orientation_error_mean)), 0.0)
        self.assertAlmostEqual(float(np.nanmin(md.nees_mean)), 0.25)
        self.assertAlmostEqual(float(np.nanmin(md.count_inside)), 4.0)
        self.assertAlmostEqual(float(np.nanmin(md.count_outside)), 0.0)
        self.assertAlmostEqual(float(np.nanmin(md.pct_inside)), 100.0)
        self.assertAlmostEqual(float(np.nanmin(md.pct_outside)), 0.0)
        self.assertAlmostEqual(float(np.nanmin(md.maha_threshold)), 16.918977604620448)
        self.assertAlmostEqual(float(np.nanmin(md.chi2_critical)), 6.6348966010212145)
        self.assertAlmostEqual(float(np.nanmin(md.chi2_significant)), 0.0)

    def test_nees_warns_on_non_symmetric_covariance(self):
        cfg = config_factory('tracking_nips_2019')
        class_name, tracks_gt = TestAlgo.single_scene()

        timestamp_boxes_pred = copy.deepcopy(tracks_gt['scene-1'])
        bad_cov = np.eye(9)
        bad_cov[0, 1] = 0.2
        bad_cov[1, 0] = 0.0
        for _, boxes in timestamp_boxes_pred.items():
            for box in boxes:
                box.covariance = bad_cov.tolist()
                box.state_dim = 9
        tracks_pred = {'scene-1': timestamp_boxes_pred}

        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            md = ev.accumulate()

        self.assertTrue(any('not symmetric' in str(w.message) for w in caught))
        self.assertTrue(np.isfinite(float(np.nanmin(md.nees_mean))))

    def test_nees_nan_with_missing_covariance(self):
        cfg = config_factory('tracking_nips_2019')
        class_name, tracks_gt = TestAlgo.single_scene()
        tracks_pred = {'scene-1': copy.deepcopy(tracks_gt['scene-1'])}

        ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            md = ev.accumulate()

        self.assertTrue(any('NEES unavailable' in str(w.message) for w in caught))
        self.assertTrue(np.all(np.isnan(md.nees_mean)))

    def test_scenarios(self):
        """ More flexible scenario test structure. """

        def create_tracks(_scenario, tag=None):
            tracks = {}
            for entry_id, entry in enumerate(_scenario['input']['pos_'+tag]):
                tracking_id = 'tag_{}'.format(entry_id)
                for timestamp, pos in enumerate(entry):
                    if timestamp not in tracks.keys():
                        tracks[timestamp] = []
                    box = TrackingBox(translation=(pos[0], pos[1], 0.0), tracking_id=tracking_id, tracking_name='car',
                                      tracking_score=0.5)
                    tracks[timestamp].append(box)

            return tracks

        # Get config.
        cfg = config_factory('tracking_nips_2019')

        for scenario in get_scenarios():
            tracks_gt = {'scene-1': create_tracks(scenario, tag='gt')}
            tracks_pred = {'scene-1': create_tracks(scenario, tag='pred')}

            # Accumulate metrics.
            ev = TrackingEvaluation(tracks_gt, tracks_pred, 'car', cfg.dist_fcn_callable,
                                    cfg.dist_th_tp, cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                    metric_worst=cfg.metric_worst, verbose=False)
            md = ev.accumulate()

            for key, value in scenario['output'].items():
                metric_values = getattr(md, key)
                metric_values = metric_values[np.logical_not(np.isnan(metric_values))]
                assert np.all(metric_values == value)


if __name__ == '__main__':
    unittest.main()
