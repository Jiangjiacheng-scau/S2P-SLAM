#!/usr/bin/env python3
"""Static regression checks for manuscript-critical implementation choices."""

from pathlib import Path
import re
import unittest


SOURCE = (Path(__file__).parents[1] / "src" / "semantic_octomap_node.cpp").read_text(
    encoding="utf-8"
)


class MethodInvariantTests(unittest.TestCase):
    def test_positive_period_state_is_explicit(self):
        self.assertIn("current_log_period_", SOURCE)
        self.assertIn("std::exp(log_period)", SOURCE)
        self.assertIn("PeriodEvolutionFactor", SOURCE)
        self.assertIn("PeriodMeasurementFactor", SOURCE)

    def test_discrete_associations_are_fixed_constructor_data(self):
        self.assertIn("fixed_index_", SOURCE)
        self.assertIn("fixed_sign_", SOURCE)
        periodic_body = SOURCE.split("class StructuralPeriodicityFactor", 1)[1].split(
            "class GroundPlaneFactor", 1
        )[0]
        self.assertNotIn("lround", periodic_body)

    def test_fallback_has_no_legacy_graph_reset(self):
        self.assertNotIn("hardReset", SOURCE)
        self.assertIn("mode_ = Mode::kFallback", SOURCE)
        self.assertIn("world_from_lio_ = current_pose_.compose(lio_local.inverse())", SOURCE)

    def test_rail_uses_yz_preupdate_covariance_and_has_no_longitudinal_row(self):
        self.assertIn("railSubspaceStatistic", SOURCE)
        self.assertIn("covariance_a(1, 1)", SOURCE)
        rail_body = SOURCE.split("class VirtualRailFactor", 1)[1].split(
            "class FactorGraphBackend", 1
        )[0]
        self.assertIn("residual(5)", rail_body)
        self.assertNotIn("p_a.x()", rail_body)

    def test_candidate_stage_cannot_update_octomap(self):
        process_body = SOURCE.split("void processPacket", 1)[1].split(
            "void publishNow", 1
        )[0]
        self.assertNotIn("updateNode", process_body)
        self.assertIn("promote(key, record)", process_body)
        self.assertIn("void integrateRay", SOURCE)

    def test_stsm_constants_match_the_manuscript(self):
        expected = {
            "voxel_resolution": "0.05",
            "viewpoint_baseline": "0.20",
            "persistence_hits": "3",
            "persistence_views": "2",
            "persistence_time": "2.0",
            "candidate_decay": "1.0",
            "semantic_threshold": "0.75",
            "occupancy_threshold": "0.50",
            "occupancy_hit": "0.60",
            "occupancy_miss": "0.40",
            "occupied_neighbor_min": "5",
        }
        for name, value in expected.items():
            self.assertRegex(SOURCE, rf"{name}\s*=\s*{re.escape(value)}")

    def test_final_export_uses_occupied_26_neighborhood(self):
        self.assertIn("occupiedNeighbors", SOURCE)
        self.assertIn("for (int dx = -1; dx <= 1; ++dx)", SOURCE)
        self.assertIn("occupied_neighbor_min", SOURCE)

    def test_vp_outage_has_bilateral_wall_centerline_fallback(self):
        self.assertIn("left_support.size() < 15", SOURCE)
        self.assertIn("right_support.size() < 15", SOURCE)
        self.assertIn("if (!vp.valid && wall_heading.valid)", SOURCE)
        self.assertIn("Vector3(0.0, slope, intercept)", SOURCE)

    def test_yaw_consensus_uses_nominal_axis_and_one_correction(self):
        calibration_body = SOURCE.split("void updateYawCorrection", 1)[1].split(
            "Rot3 correctedRcb", 1
        )[0]
        self.assertIn("nominal_t_bc_.rotation().rotate", calibration_body)
        self.assertNotIn("correctedRcb()", calibration_body)
        self.assertIn("aligned_lidar", calibration_body)
        self.assertIn("predicted_heading_body", calibration_body)
        self.assertIn("wrapPi(disparity - yaw_correction_)", calibration_body)

    def test_flow_uses_depth_binned_variance_not_inverse_range(self):
        self.assertIn("flowDepthVariance", SOURCE)
        self.assertIn("flowDepthVariance(depths[index])", SOURCE)
        self.assertNotIn("depths[index] * depths[index]", SOURCE)

    def test_period_candidate_geometry_uses_lateral_span(self):
        self.assertIn("const double lateral_span = max_y - min_y", SOURCE)
        self.assertIn("lateral_span > config_.structural_span", SOURCE)

    def test_period_peak_is_unique_and_subbin_neighbors_are_valid(self):
        self.assertIn("double second_peak", SOURCE)
        self.assertIn("peak <= second_peak + kEpsilon", SOURCE)
        self.assertIn("peak_lag + 1 < bins", SOURCE)

    def test_period_motion_polarity_is_aisle_local(self):
        self.assertIn(
            "aisle_.toAisle(current_pose_.translation()).x()", SOURCE
        )

    def test_period_phase_is_initialized_before_fixed_association(self):
        periodic_body = SOURCE.split(
            "if (features.period.event && features.period.phase_gate)", 1
        )[1].split("if (lidar_structural_available", 1)[0]
        self.assertIn("std::fmod(x_long, predicted_period)", periodic_body)
        self.assertIn("if (aisle_.phase < 0.0)", periodic_body)

    def test_lio_monitor_and_factor_share_keyframe_interval(self):
        self.assertIn(
            "last_keyframe_lio_local_.between(lio_local)", SOURCE
        )
        self.assertNotIn("last_lio_local_", SOURCE)

    def test_published_pose_covariance_reorders_gtsam_to_ros(self):
        self.assertIn("kRosToGtsam{3, 4, 5, 0, 1, 2}", SOURCE)

    def test_reported_factor_removals_have_explicit_switches(self):
        for switch in (
            "enable_periodicity",
            "enable_rail",
            "enable_vp",
            "enable_hybrid",
            "enable_semantic_gate",
            "enable_stsm",
        ):
            self.assertIn(switch, SOURCE)

    def test_lidar_structural_gates_require_registration_status(self):
        gate = SOURCE.split("const bool lidar_structural_available", 1)[1].split(
            ";", 1
        )[0]
        self.assertIn("lio_status_converged", gate)

    def test_stale_encoder_cannot_create_hybrid_factor(self):
        self.assertIn("config_.enable_hybrid && wheel.valid", SOURCE)


if __name__ == "__main__":
    unittest.main()
