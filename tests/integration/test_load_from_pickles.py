"""
CRITICAL: Integration tests for loading from existing pickle files.

Tests common workflow of starting Stage 2 or Stage 3 from pre-converted pickles.
This was previously completely untested but is a very common use case.
"""

from scenariomax.core import pipeline


class TestStage2FromPickles:
    """Test Stage 2 (processing) starting from pre-converted pickles."""

    def test_stage2_from_waymo_pickles(self, waymo_unified_pickles, temp_output_dir):
        """Test processing Waymo pickles with traffic_lights processor."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["traffic_lights"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0, "Should process scenarios"
        assert stats["errors"] == 0, "Should have no errors"

        # Check processed pickles created
        processed_dir = temp_output_dir / "processed" / "waymo"
        assert processed_dir.exists()

        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0, "Should create processed pickle files"

    def test_stage2_from_nuplan_pickles(self, nuplan_unified_pickles, temp_output_dir):
        """Test processing nuPlan pickles with traffic_lights processor."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(nuplan_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["traffic_lights"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        processed_dir = temp_output_dir / "processed" / "nuplan"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_stage2_with_validation_from_pickles(self, waymo_unified_pickles, temp_output_dir):
        """Test validation processor on existing pickles."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation"],
            processor_configs={"validation": {"validation_level": 2}},
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        # Validation may filter some scenarios
        assert stats["filtered"] >= 0

    def test_stage2_with_polyline_interpolation_from_pickles(
        self,
        waymo_unified_pickles,
        temp_output_dir,
        polyline_interpolation_config,
    ):
        """Test polyline_interpolation processor on existing pickles."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["polyline_interpolation"],
            processor_configs=polyline_interpolation_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_stage2_with_all_processors_from_pickles(
        self,
        waymo_unified_pickles,
        temp_output_dir,
        all_processors_config,
    ):
        """Test combining all processors on existing pickles."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=["validation", "traffic_lights", "polyline_interpolation"],
            processor_configs=all_processors_config,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0

    def test_stage2_no_processors_identity_copy(self, waymo_unified_pickles, temp_output_dir):
        """Test Stage 2 with no processors (should copy files)."""
        stats = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            processors=[],
            processor_configs=None,
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        # Files should be copied even without processing
        processed_dir = temp_output_dir / "processed" / "waymo"
        pkl_files = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files) > 0


class TestStage3FromUnprocessedPickles:
    """Test Stage 3 (formatting) starting from unified pickles (no Stage 2)."""

    def test_stage3_waymax_from_pickles(self, waymo_unified_pickles, temp_output_dir):
        """Test Waymax conversion from unified pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("*.tfrecord"))
        assert len(tfrecords) > 0

    def test_stage3_gpudrive_from_pickles(self, waymo_unified_pickles, temp_output_dir):
        """Test GPUDrive conversion from unified pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="gpudrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        gpudrive_dir = temp_output_dir / "gpudrive" / "waymo"
        json_files = list(gpudrive_dir.glob("*.json"))
        assert len(json_files) > 0

    def test_stage3_pufferdrive_from_pickles(self, waymo_unified_pickles, temp_output_dir):
        """Test PufferDrive conversion from unified pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0


class TestStage3FromProcessedPickles:
    """Test Stage 3 starting from processed pickles (after Stage 2)."""

    def test_stage3_waymax_from_processed_pickles(self, processed_pickles, temp_output_dir):
        """Test Waymax conversion from processed (Stage 2) pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(processed_pickles.parent),
            output_path=str(temp_output_dir),
            format="waymax",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0
        assert stats["errors"] == 0

        waymax_dir = temp_output_dir / "waymax"
        tfrecords = list(waymax_dir.glob("*.tfrecord"))
        assert len(tfrecords) > 0

    def test_stage3_gpudrive_from_processed_pickles(self, processed_pickles, temp_output_dir):
        """Test GPUDrive conversion from processed pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(processed_pickles.parent),
            output_path=str(temp_output_dir),
            format="gpudrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        gpudrive_dir = temp_output_dir / "gpudrive" / "waymo"
        json_files = list(gpudrive_dir.glob("*.json"))
        assert len(json_files) > 0

    def test_stage3_pufferdrive_from_processed_pickles(self, processed_pickles, temp_output_dir):
        """Test PufferDrive conversion from processed pickles."""
        stats = pipeline.format_unified_to_target(
            input_path=str(processed_pickles.parent),
            output_path=str(temp_output_dir),
            format="pufferdrive",
            num_workers=2,
        )

        assert stats["total_scenarios"] > 0

        pufferdrive_dir = temp_output_dir / "pufferdrive"
        bin_files = list(pufferdrive_dir.glob("map_*.bin"))
        assert len(bin_files) > 0


class TestFullWorkflowSeparateStages:
    """Test full workflow with pickles saved between each stage."""

    def test_convert_then_process_then_format(self, waymo_test_data, temp_output_dir):
        """
        Test complete workflow with separate stage executions:
        Stage 1 (save pickles) → Stage 2 (load, process, save) → Stage 3 (load, format)
        """
        # Stage 1: Convert to unified pickles
        stage1_output = temp_output_dir / "stage1"
        stage1_output.mkdir()

        datasets = {"waymo": {"path": str(waymo_test_data), "version": "v1.2"}}

        stats1 = pipeline.convert_raw_to_unified(
            datasets=datasets,
            output_path=str(stage1_output),
            num_workers=2,
            batch_size=10,
        )

        assert stats1["total_scenarios"] > 0
        unified_dir = stage1_output / "unified" / "waymo"
        assert unified_dir.exists()
        pkl_files_stage1 = list(unified_dir.glob("*.pkl"))
        assert len(pkl_files_stage1) > 0

        # Stage 2: Process unified pickles
        stage2_output = temp_output_dir / "stage2"
        stage2_output.mkdir()

        stats2 = pipeline.process_unified_scenarios(
            input_path=str(stage1_output / "unified"),
            output_path=str(stage2_output),
            processors=["traffic_lights", "polyline_interpolation"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats2["total_scenarios"] > 0
        processed_dir = stage2_output / "processed" / "waymo"
        assert processed_dir.exists()
        pkl_files_stage2 = list(processed_dir.glob("*.pkl"))
        assert len(pkl_files_stage2) > 0

        # Stage 3: Format processed pickles to all formats
        stage3_output = temp_output_dir / "stage3"
        stage3_output.mkdir()

        # Test all three formats
        for fmt in ["waymax", "gpudrive", "pufferdrive"]:
            fmt_output = temp_output_dir / f"stage3_{fmt}"
            fmt_output.mkdir()

            stats3 = pipeline.format_unified_to_target(
                input_path=str(stage2_output / "processed"),
                output_path=str(fmt_output),
                format=fmt,
                num_workers=2,
            )

            assert stats3["total_scenarios"] > 0, f"Failed for format: {fmt}"

    def test_stage2_and_stage3_from_same_pickles(self, waymo_unified_pickles, temp_output_dir):
        """
        Test running Stage 2 and Stage 3 independently from same unified pickles.
        """
        # Run Stage 2
        stage2_output = temp_output_dir / "stage2"
        stage2_output.mkdir()

        stats2 = pipeline.process_unified_scenarios(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(stage2_output),
            processors=["traffic_lights"],
            processor_configs=None,
            num_workers=2,
        )

        assert stats2["total_scenarios"] > 0

        # Run Stage 3 from same unified pickles (independent of Stage 2)
        stage3_output = temp_output_dir / "stage3"
        stage3_output.mkdir()

        stats3 = pipeline.format_unified_to_target(
            input_path=str(waymo_unified_pickles.parent),
            output_path=str(stage3_output),
            format="waymax",
            num_workers=2,
        )

        assert stats3["total_scenarios"] > 0

        # Both should succeed independently
        assert stats2["total_scenarios"] == stats3["total_scenarios"]


class TestPickleReusability:
    """Test that pickles can be reused multiple times."""

    def test_multiple_format_conversions_from_same_pickles(
        self,
        waymo_unified_pickles,
        temp_output_dir,
    ):
        """Test converting same pickles to multiple formats."""
        formats = ["waymax", "gpudrive", "pufferdrive"]

        for fmt in formats:
            fmt_output = temp_output_dir / fmt
            fmt_output.mkdir()

            stats = pipeline.format_unified_to_target(
                input_path=str(waymo_unified_pickles.parent),
                output_path=str(fmt_output),
                format=fmt,
                num_workers=2,
            )

            assert stats["total_scenarios"] > 0, f"Failed for format: {fmt}"

    def test_multiple_processing_runs_from_same_pickles(
        self,
        waymo_unified_pickles,
        temp_output_dir,
    ):
        """Test processing same pickles multiple times with different processors."""
        processor_configs = [
            ["validation"],
            ["traffic_lights"],
            ["polyline_interpolation"],
            ["validation", "traffic_lights", "polyline_interpolation"],
        ]

        for i, processors in enumerate(processor_configs):
            proc_output = temp_output_dir / f"proc_{i}"
            proc_output.mkdir()

            stats = pipeline.process_unified_scenarios(
                input_path=str(waymo_unified_pickles.parent),
                output_path=str(proc_output),
                processors=processors,
                processor_configs=None,
                num_workers=2,
            )

            assert stats["total_scenarios"] > 0, f"Failed for processors: {processors}"
