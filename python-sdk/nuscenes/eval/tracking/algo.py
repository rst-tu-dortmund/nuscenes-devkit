"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
import warnings
from typing import List, Dict, Callable, Tuple
import unittest

import numpy as np
import sklearn
import tqdm

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

from nuscenes.eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricData
from nuscenes.eval.common.utils import center_distance, scale_iou, velocity_l2, yaw_diff, quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.eval.tracking.render import TrackingRenderer
from nuscenes.eval.tracking.utils import print_threshold_metrics, create_motmetrics


TP_ERROR_METRIC_MAP = {
    'tp_translation_error_mean': 'tp_translation_error_mean',
    'tp_scale_error_mean': 'tp_scale_error_mean',
    'tp_velocity_error_mean': 'tp_velocity_error_mean',
    'tp_orientation_error_mean': 'tp_orientation_error_mean',
}


def _normalize_angle_diff(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _safe_yaw_from_rotation(rotation) -> float:
    try:
        quaternion = Quaternion(rotation)
        if quaternion.norm == 0:
            return 0.0
        return float(quaternion_yaw(quaternion))
    except Exception:
        return 0.0


def compute_tp_errors(gt_box: TrackingBox, pred_box: TrackingBox) -> Dict[str, float]:
    try:
        scale_error = float(1.0 - scale_iou(gt_box, pred_box))
    except Exception:
        scale_error = np.nan

    try:
        orientation_error = float(yaw_diff(gt_box, pred_box, period=2 * np.pi))
    except Exception:
        orientation_error = np.nan

    return {
        'tp_translation_error_mean': float(center_distance(gt_box, pred_box)),
        'tp_scale_error_mean': scale_error,
        'tp_velocity_error_mean': float(velocity_l2(gt_box, pred_box)),
        'tp_orientation_error_mean': orientation_error,
    }


def _default_state_vector(box: TrackingBox) -> np.ndarray:
    yaw = _safe_yaw_from_rotation(box.rotation)
    return np.array([
        box.translation[0], box.translation[1], box.translation[2],
        box.size[0], box.size[1], box.size[2],
        box.velocity[0], box.velocity[1],
        yaw,
    ], dtype=float)


def parse_covariance(box: TrackingBox, expected_dim: int, warn_prefix: str) -> np.ndarray:
    if box.covariance is None:
        return None
    try:
        covariance = np.array(box.covariance, dtype=float)
    except Exception:
        warnings.warn(f'{warn_prefix}: invalid covariance format for track {box.tracking_id}.', RuntimeWarning)
        return None

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        warnings.warn(f'{warn_prefix}: covariance must be square for track {box.tracking_id}.', RuntimeWarning)
        return None

    if box.state_dim is not None and int(box.state_dim) != covariance.shape[0]:
        warnings.warn(
            f'{warn_prefix}: state_dim ({box.state_dim}) does not match covariance shape '
            f'({covariance.shape[0]}x{covariance.shape[1]}) for track {box.tracking_id}.',
            RuntimeWarning
        )

    if expected_dim != covariance.shape[0]:
        warnings.warn(
            f'{warn_prefix}: covariance dimension mismatch for track {box.tracking_id} '
            f'(expected {expected_dim}, got {covariance.shape[0]}).',
            RuntimeWarning
        )
        return None

    if not np.allclose(covariance, covariance.T, atol=1e-8):
        warnings.warn(f'{warn_prefix}: covariance is not symmetric for track {box.tracking_id}.', RuntimeWarning)

    eigvals = np.linalg.eigvalsh(covariance)
    if np.any(eigvals < -1e-8):
        warnings.warn(f'{warn_prefix}: covariance is not PSD for track {box.tracking_id}.', RuntimeWarning)

    return covariance


def compute_nees(gt_box: TrackingBox,
                 pred_box: TrackingBox,
                 state_indices: List[int] = None,
                 warn_prefix: str = 'Tracking NEES') -> Tuple[float, int]:
    pred_state = _default_state_vector(pred_box)
    gt_state = _default_state_vector(gt_box)

    if state_indices is None:
        indices = list(range(len(pred_state)))
    else:
        indices = list(state_indices)

    pred_sel = pred_state[indices]
    gt_sel = gt_state[indices]
    residual = pred_sel - gt_sel

    yaw_state_index = 8
    if yaw_state_index in indices:
        yaw_pos = indices.index(yaw_state_index)
        residual[yaw_pos] = _normalize_angle_diff(float(residual[yaw_pos]))

    covariance = parse_covariance(pred_box, expected_dim=len(pred_state), warn_prefix=warn_prefix)
    if covariance is None:
        return np.nan, len(indices)
    covariance_sel = covariance[np.ix_(indices, indices)]

    try:
        inv_covariance = np.linalg.inv(covariance_sel)
    except np.linalg.LinAlgError:
        warnings.warn(f'{warn_prefix}: covariance inversion failed, using pseudo-inverse for {pred_box.tracking_id}.',
                      RuntimeWarning)
        inv_covariance = np.linalg.pinv(covariance_sel)

    nees = float(residual.T @ inv_covariance @ residual)
    return nees, len(indices)


class TrackingEvaluation(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 nees_state_indices: List[int] = None,
                 verbose: bool = True,
                 output_dir: str = None,
                 render_classes: List[str] = None):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.metric_worst = metric_worst
        self.nees_state_indices = nees_state_indices
        self.verbose = verbose
        self.output_dir = output_dir
        self.render_classes = [] if render_classes is None else render_classes

        self.n_scenes = len(self.tracks_gt)

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)
        accumulators = []
        thresh_metrics = []
        tp_metric_stats_by_threshold = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        mh = create_motmetrics()

        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        thresholds, recalls = self.compute_thresholds(gt_box_count)
        md.confidence = thresholds
        md.recall_hypo = recalls
        if self.verbose:
            print('Computed thresholds\n')

        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes with the exact same score.
            if threshold in thresholds[:t]:
                continue

            # Accumulate track data.
            acc, _, tp_metric_stats = self.accumulate_threshold(threshold)
            accumulators.append(acc)

            # Compute metrics for current threshold.
            thresh_name = self.name_gen(threshold)
            thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)
            thresh_metrics.append(thresh_summary)
            tp_metric_stats_by_threshold.append(tp_metric_stats)

            # Print metrics to stdout.
            if self.verbose:
                print_threshold_metrics(thresh_summary.to_dict())

        # Concatenate all metrics. We only do this for more convenient access.
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Get the number of thresholds which were not achieved (i.e. nan).
        unachieved_thresholds = np.array([t for t in thresholds if np.isnan(t)])
        num_unachieved_thresholds = len(unachieved_thresholds)

        # Get the number of thresholds which were achieved (i.e. not nan).
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        num_duplicate_thresholds = len(valid_thresholds) - len(np.unique(valid_thresholds))

        # Sanity check.
        assert num_unachieved_thresholds + num_duplicate_thresholds + len(thresh_metrics) == self.num_thresholds

        # Figure out how many times each threshold should be repeated.
        rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]

        # Store all traditional metrics.
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            # Skip metrics which we don't output.
            if metric_name == '':
                continue

            # Retrieve and store values for current metric.
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        worst = np.nan  # We can't know how these error types are distributed.
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once, duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)

            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        # Store TP error and NEES metrics.
        extended_metric_names = [
            'tp_translation_error_mean',
            'tp_scale_error_mean',
            'tp_velocity_error_mean',
            'tp_orientation_error_mean',
            'nees_mean',
            'nees_calibration_score'
        ]

        if len(tp_metric_stats_by_threshold) == 0:
            for metric_name in extended_metric_names:
                md.set_metric(metric_name, [np.nan] * TrackingMetricData.nelem)
            return md

        per_threshold_metrics = {metric_name: [] for metric_name in extended_metric_names}
        for stats in tp_metric_stats_by_threshold:
            for metric_name in TP_ERROR_METRIC_MAP.keys():
                metric_count = stats['tp_error_counts'][metric_name]
                if metric_count == 0:
                    value = np.nan
                else:
                    value = stats['tp_error_sums'][metric_name] / metric_count
                per_threshold_metrics[metric_name].append(value)

            if stats['nees_count'] == 0:
                nees_mean = np.nan
            else:
                nees_mean = stats['nees_sum'] / stats['nees_count']
            per_threshold_metrics['nees_mean'].append(nees_mean)

            if np.isnan(nees_mean) or np.isnan(stats['nees_dof']):
                calibration = np.nan
            else:
                calibration = (nees_mean - stats['nees_dof']) ** 2
            per_threshold_metrics['nees_calibration_score'].append(calibration)

        for metric_name in extended_metric_names:
            values = np.array(per_threshold_metrics[metric_name], dtype=float)
            assert len(rep_counts) == len(values)
            values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])

            all_values = [np.nan] * num_unachieved_thresholds
            all_values.extend(values)
            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def accumulate_threshold(self, threshold: float = None) -> Tuple[pandas.DataFrame, List[float], Dict[str, float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP, TP/NEES stats).
        """
        accs = []
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.
        tp_error_sums = {metric_name: 0.0 for metric_name in TP_ERROR_METRIC_MAP.keys()}
        tp_error_counts = {metric_name: 0 for metric_name in TP_ERROR_METRIC_MAP.keys()}
        nees_sum = 0.0
        nees_count = 0
        nees_dof = np.nan
        missing_covariance_for_nees = False

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            if self.class_name in self.render_classes and threshold is None:
                save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(save_path)
            else:
                renderer = None

            for timestamp in scene_tracks_gt.keys():
                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]

                # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                if threshold is not None:
                    frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                # Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                events = acc.events.loc[frame_id]
                matches = events[events.Type == 'MATCH']
                gt_id_to_box = {box.tracking_id: box for box in frame_gt}
                pred_id_to_box = {box.tracking_id: box for box in frame_pred}
                for _, row in matches.iterrows():
                    gt_match = gt_id_to_box[row.OId]
                    pred_match = pred_id_to_box[row.HId]

                    tp_errors = compute_tp_errors(gt_match, pred_match)
                    for metric_name, value in tp_errors.items():
                        if np.isfinite(value):
                            tp_error_sums[metric_name] += value
                            tp_error_counts[metric_name] += 1

                    nees_value, dof = compute_nees(
                        gt_match,
                        pred_match,
                        state_indices=self.nees_state_indices,
                        warn_prefix=f'Tracking NEES ({self.class_name})'
                    )
                    if np.isfinite(nees_value):
                        nees_sum += nees_value
                        nees_count += 1
                        nees_dof = float(dof)
                    elif pred_match.covariance is None:
                        missing_covariance_for_nees = True

                # Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    match_ids = matches.HId.values
                    match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                    scores.extend(match_scores)

                # Render the boxes in this frame.
                if self.class_name in self.render_classes and threshold is None:
                    renderer.render(events, timestamp, frame_gt, frame_pred)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1

            accs.append(acc)

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        if threshold is not None and sum(tp_error_counts.values()) > 0 and nees_count == 0 and missing_covariance_for_nees:
            warnings.warn(
                f'Tracking NEES unavailable for class {self.class_name} at threshold {threshold:.4f}: '
                f'covariance missing for matched true positives.',
                RuntimeWarning
            )

        tp_nees_stats = {
            'tp_error_sums': tp_error_sums,
            'tp_error_counts': tp_error_counts,
            'nees_sum': nees_sum,
            'nees_count': nees_count,
            'nees_dof': nees_dof
        }

        return acc_merged, scores, tp_nees_stats

    def compute_thresholds(self, gt_box_count: int) -> Tuple[List[float], List[float]]:
        """
        Compute the score thresholds for predefined recall values.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_box_count: The number of GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # Run accumulate to get the scores of TPs.
        _, scores, _ = self.accumulate_threshold(threshold=None)

        # Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds, [np.nan] * self.num_thresholds

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / gt_box_count
        assert len(scores) / gt_box_count <= 1

        # Determine thresholds.
        max_recall_achieved = np.max(rec)
        rec_interp = np.linspace(self.min_recall, 1, self.num_thresholds).round(12)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        # Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp
