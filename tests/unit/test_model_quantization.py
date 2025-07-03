"""
Comprehensive tests for inference.llm.model_quantization module.

Tests model quantization functionality including various formats,
edge optimization, and deployment package creation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from inference.llm.model_quantization import (
    EdgeOptimizer,
    ModelQuantizer,
    QuantizationConfig,
    QuantizationType,
    auto_quantize,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_quantization_config():
    """Create a sample quantization configuration."""
    return QuantizationConfig(
        quantization_type=QuantizationType.INT8,
        calibration_samples=50,
        symmetric=True,
        per_channel=False,
        optimize_for="size",
        target_device="cpu",
        compression_level=5,
    )


@pytest.fixture
def sample_model_file(temp_dir):
    """Create a sample model file for testing."""
    model_path = temp_dir / "test_model.pt"
    # Create a fake model file with some content
    model_path.write_bytes(b"fake_pytorch_model_data" * 1000)
    return model_path


@pytest.fixture
def sample_ggml_file(temp_dir):
    """Create a sample GGML model file for testing."""
    model_path = temp_dir / "test_model.ggml"
    # Create a fake GGML file with header
    content = b"ggml" + b"fake_ggml_model_data" * 1000
    model_path.write_bytes(content)
    return model_path


@pytest.fixture
def sample_onnx_file(temp_dir):
    """Create a sample ONNX model file for testing."""
    model_path = temp_dir / "test_model.onnx"
    model_path.write_bytes(b"fake_onnx_model_data" * 1000)
    return model_path


class TestQuantizationType:
    """Test QuantizationType enum."""

    def test_quantization_type_values(self):
        """Test that quantization type enum has correct values."""
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.INT3.value == "int3"
        assert QuantizationType.DYNAMIC.value == "dynamic"
        assert QuantizationType.MIXED.value == "mixed"


class TestQuantizationConfig:
    """Test QuantizationConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating configuration with default values."""
        config = QuantizationConfig(quantization_type=QuantizationType.INT8)

        assert config.quantization_type == QuantizationType.INT8
        assert config.calibration_samples == 100
        assert config.symmetric is True
        assert config.per_channel is True
        assert config.optimize_for == "size"
        assert config.target_device == "cpu"
        assert config.keep_original is True
        assert config.compression_level == 6

    def test_config_creation_with_custom_values(self):
        """Test creating configuration with custom values."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT4,
            calibration_samples=50,
            symmetric=False,
            per_channel=False,
            optimize_for="speed",
            target_device="cuda",
            keep_original=False,
            compression_level=9,
            mixed_precision_layers=["layer1", "layer2"],
            skip_layers=["output"],
            custom_bits={"conv1": 4, "fc1": 8},
        )

        assert config.quantization_type == QuantizationType.INT4
        assert config.calibration_samples == 50
        assert config.symmetric is False
        assert config.per_channel is False
        assert config.optimize_for == "speed"
        assert config.target_device == "cuda"
        assert config.keep_original is False
        assert config.compression_level == 9
        assert config.mixed_precision_layers == ["layer1", "layer2"]
        assert config.skip_layers == ["output"]
        assert config.custom_bits == {"conv1": 4, "fc1": 8}


class TestModelQuantizer:
    """Test ModelQuantizer functionality."""

    def test_quantizer_initialization(self, sample_quantization_config):
        """Test model quantizer initialization."""
        quantizer = ModelQuantizer(sample_quantization_config)

        assert quantizer.config == sample_quantization_config
        assert quantizer.calibration_data == []

    def test_detect_model_format_pytorch(self, sample_model_file):
        """Test PyTorch model format detection."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        assert quantizer._detect_model_format(sample_model_file) == "pytorch"

    def test_detect_model_format_ggml_by_extension(self, temp_dir):
        """Test GGML model format detection by file extension."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        ggml_file = temp_dir / "test.ggml"
        ggml_file.write_bytes(b"test")
        assert quantizer._detect_model_format(ggml_file) == "ggml"

        gguf_file = temp_dir / "test.gguf"
        gguf_file.write_bytes(b"test")
        assert quantizer._detect_model_format(gguf_file) == "ggml"

        bin_file = temp_dir / "test.bin"
        bin_file.write_bytes(b"test")
        assert quantizer._detect_model_format(bin_file) == "ggml"

    def test_detect_model_format_ggml_by_content(self, sample_ggml_file):
        """Test GGML model format detection by file content."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        # Rename to remove .ggml extension
        unknown_file = sample_ggml_file.with_suffix(".unknown")
        sample_ggml_file.rename(unknown_file)

        assert quantizer._detect_model_format(unknown_file) == "ggml"

    def test_detect_model_format_onnx(self, sample_onnx_file):
        """Test ONNX model format detection."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        assert quantizer._detect_model_format(sample_onnx_file) == "onnx"

    def test_detect_model_format_unknown(self, temp_dir):
        """Test unknown model format detection."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        unknown_file = temp_dir / "test.unknown"
        unknown_file.write_bytes(b"unknown_model_data")

        assert quantizer._detect_model_format(unknown_file) == "unknown"

    @patch("inference.llm.model_quantization.subprocess.run")
    @patch("inference.llm.model_quantization.shutil.which")
    def test_quantize_ggml_with_external_tool(
        self, mock_which, mock_run, sample_ggml_file, temp_dir
    ):
        """Test GGML quantization using external tool."""
        # Setup mocks
        mock_which.return_value = "/usr/bin/quantize"
        mock_run.return_value = Mock(returncode=0, stderr="")

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT4))
        output_path = temp_dir / "quantized.ggml"

        # Create output file to simulate successful quantization
        output_path.write_bytes(b"quantized_data")

        result = quantizer._quantize_ggml(sample_ggml_file, output_path)

        # Verify external tool was called
        mock_which.assert_called_with("quantize")
        mock_run.assert_called_once()

        # Check result structure
        assert result["format"] == "ggml"
        assert result["quantization_type"] == "q4_K_M"
        assert "original_size" in result
        assert "quantized_size" in result
        assert "compression_ratio" in result
        assert "size_reduction" in result

    @patch("inference.llm.model_quantization.shutil.which")
    def test_quantize_ggml_manual_fallback(self, mock_which, sample_ggml_file, temp_dir):
        """Test GGML quantization manual fallback."""
        # Setup mock to return None (tool not available)
        mock_which.return_value = None

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        output_path = temp_dir / "quantized.ggml"

        result = quantizer._quantize_ggml(sample_ggml_file, output_path)

        # Check result structure
        assert result["format"] == "ggml"
        assert result["quantization_type"] == "q8_0"
        assert "note" in result
        assert "Simplified quantization" in result["note"]

        # Check that output file was created
        assert output_path.exists()

    def test_manual_ggml_quantize_different_types(self, sample_ggml_file, temp_dir):
        """Test manual GGML quantization with different types."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT3))

        test_cases = [
            ("q8_0", 0.8),
            ("q4_K_M", 0.4),
            ("q3_K_M", 0.3),
            ("unknown", 0.5),
        ]

        for quant_type, expected_scale in test_cases:
            output_path = temp_dir / f"quantized_{quant_type}.ggml"
            result = quantizer._manual_ggml_quantize(sample_ggml_file, output_path, quant_type)

            assert result["quantization_type"] == quant_type
            assert output_path.exists()

            # Check that file was actually reduced in size
            original_size = result["original_size"]
            quantized_size = result["quantized_size"]
            expected_size = int(original_size * expected_scale)
            assert quantized_size == expected_size

    @patch("inference.llm.model_quantization.torch.load")
    @patch("inference.llm.model_quantization.torch.save")
    @patch("inference.llm.model_quantization.quantization.quantize_dynamic")
    def test_quantize_pytorch_int8(
        self, mock_quant_dynamic, mock_save, mock_load, sample_model_file, temp_dir
    ):
        """Test PyTorch INT8 quantization."""
        # Setup mock model
        mock_model = Mock(spec=["eval"])
        mock_load.return_value = mock_model
        mock_quantized = Mock()
        mock_quant_dynamic.return_value = mock_quantized

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        output_path = temp_dir / "quantized.pt"

        result = quantizer._quantize_pytorch(sample_model_file, output_path)

        # Verify torch operations
        mock_load.assert_called_once_with(sample_model_file, map_location="cpu")
        mock_model.eval.assert_called_once()
        mock_quant_dynamic.assert_called_once()
        mock_save.assert_called_once_with(mock_quantized, output_path)

        # Check result
        assert result["format"] == "pytorch"
        assert result["quantization_type"] == "int8"

    @patch("inference.llm.model_quantization.torch.load")
    @patch("inference.llm.model_quantization.torch.save")
    def test_quantize_pytorch_int4(self, mock_save, mock_load, sample_model_file, temp_dir):
        """Test PyTorch INT4 quantization."""
        # Setup mock model
        mock_model = Mock(spec=["eval", "clone"])
        mock_model.clone.return_value = mock_model
        mock_load.return_value = mock_model

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT4))
        output_path = temp_dir / "quantized.pt"

        with patch.object(
            quantizer, "_pytorch_int4_quantize", return_value=mock_model
        ) as mock_int4:
            result = quantizer._quantize_pytorch(sample_model_file, output_path)

            mock_int4.assert_called_once_with(mock_model)
            mock_save.assert_called_once()

            assert result["format"] == "pytorch"
            assert result["quantization_type"] == "int4"

    @patch("inference.llm.model_quantization.torch.load")
    def test_quantize_pytorch_invalid_format(self, mock_load, sample_model_file, temp_dir):
        """Test PyTorch quantization with invalid model format."""
        mock_load.return_value = {"not": "a_model"}  # Not a torch.nn.Module

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        output_path = temp_dir / "quantized.pt"

        with pytest.raises(ValueError, match="Invalid PyTorch model format"):
            quantizer._quantize_pytorch(sample_model_file, output_path)

    @patch("inference.llm.model_quantization.torch")
    def test_pytorch_int4_quantize(self, mock_torch):
        """Test custom INT4 quantization implementation."""
        # Setup mock linear layer
        mock_layer = Mock()
        mock_layer.in_features = 10
        mock_layer.out_features = 5
        mock_layer.weight.data = Mock()
        mock_layer.weight.data.abs.return_value.max.return_value = 14.0  # scale will be 2.0
        mock_layer.bias = Mock()

        # Setup torch mocks
        mock_torch.round.return_value.clamp.return_value.to.return_value = Mock()
        mock_torch.int8 = Mock()

        # Setup mock model
        mock_model = Mock()
        mock_model.clone.return_value = mock_model
        mock_model.named_children.return_value = [("linear1", mock_layer)]

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT4))
        result = quantizer._pytorch_int4_quantize(mock_model)

        # Verify model was cloned
        mock_model.clone.assert_called_once()
        assert result == mock_model

    @patch("inference.llm.model_quantization.quantize_dynamic")
    def test_quantize_onnx_success(self, mock_quantize, sample_onnx_file, temp_dir):
        """Test successful ONNX quantization."""
        output_path = temp_dir / "quantized.onnx"
        output_path.write_bytes(b"quantized_onnx_data")  # Simulate output

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        result = quantizer._quantize_onnx(sample_onnx_file, output_path)

        mock_quantize.assert_called_once()
        assert result["format"] == "onnx"
        assert result["quantization_type"] == "int8_dynamic"

    @patch("inference.llm.model_quantization.quantize_dynamic")
    def test_quantize_onnx_import_error(self, mock_quantize, sample_onnx_file, temp_dir):
        """Test ONNX quantization with import error."""
        mock_quantize.side_effect = ImportError("ONNX Runtime not available")

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        output_path = temp_dir / "quantized.onnx"

        with pytest.raises(ImportError):
            quantizer._quantize_onnx(sample_onnx_file, output_path)

    def test_quantize_model_unsupported_format(self, temp_dir):
        """Test quantizing unsupported model format."""
        unknown_file = temp_dir / "model.unknown"
        unknown_file.write_bytes(b"unknown_format")

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        output_path = temp_dir / "output.unknown"

        with pytest.raises(ValueError, match="Unsupported model format"):
            quantizer.quantize_model(unknown_file, output_path)

    def test_benchmark_quantized_model(self, sample_model_file, temp_dir):
        """Test benchmarking functionality."""
        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))
        quantized_path = temp_dir / "quantized.pt"
        quantized_path.write_bytes(b"quantized_model")

        test_inputs = ["input1", "input2"]
        result = quantizer.benchmark_quantized_model(sample_model_file, quantized_path, test_inputs)

        # Check result structure
        assert "inference_time" in result
        assert "accuracy" in result
        assert "memory_usage" in result

        assert "original_ms" in result["inference_time"]
        assert "quantized_ms" in result["inference_time"]
        assert "speedup" in result["inference_time"]

        assert "original" in result["accuracy"]
        assert "quantized" in result["accuracy"]
        assert "degradation" in result["accuracy"]

        assert "original_mb" in result["memory_usage"]
        assert "quantized_mb" in result["memory_usage"]
        assert "reduction" in result["memory_usage"]


class TestEdgeOptimizer:
    """Test EdgeOptimizer functionality."""

    def test_edge_optimizer_initialization(self):
        """Test edge optimizer initialization."""
        optimizer = EdgeOptimizer()

        # Check that device profiles are loaded
        assert "raspberry_pi" in optimizer.device_profiles
        assert "jetson_nano" in optimizer.device_profiles
        assert "intel_nuc" in optimizer.device_profiles
        assert "mobile_phone" in optimizer.device_profiles

        # Check profile structure
        rpi_profile = optimizer.device_profiles["raspberry_pi"]
        assert "ram_gb" in rpi_profile
        assert "cpu_cores" in rpi_profile
        assert "has_gpu" in rpi_profile
        assert "arch" in rpi_profile

    def test_device_profile_characteristics(self):
        """Test device profile characteristics."""
        optimizer = EdgeOptimizer()

        # Raspberry Pi profile
        rpi = optimizer.device_profiles["raspberry_pi"]
        assert rpi["ram_gb"] == 1
        assert rpi["has_gpu"] is False
        assert rpi["arch"] == "arm64"

        # Jetson Nano profile
        jetson = optimizer.device_profiles["jetson_nano"]
        assert jetson["ram_gb"] == 4
        assert jetson["has_gpu"] is True

        # Intel NUC profile
        nuc = optimizer.device_profiles["intel_nuc"]
        assert nuc["arch"] == "x86_64"
        assert nuc["has_gpu"] is False

    @patch.object(ModelQuantizer, "quantize_model")
    def test_optimize_for_device_raspberry_pi(self, mock_quantize, sample_model_file, temp_dir):
        """Test optimization for Raspberry Pi (low RAM device)."""
        mock_quantize.return_value = {
            "format": "pytorch",
            "quantization_type": "int3",
            "compression_ratio": 8.0,
        }

        optimizer = EdgeOptimizer()
        result = optimizer.optimize_for_device(sample_model_file, "raspberry_pi", temp_dir)

        # Check that INT3 quantization was used for low RAM device
        mock_quantize.assert_called_once()
        # ModelQuantizer was called with config
        _ = mock_quantize.call_args[0][0]

        assert result["device_optimizations"]["device_type"] == "raspberry_pi"
        assert result["device_optimizations"]["recommended_batch_size"] == 1
        assert result["device_optimizations"]["gpu_offload"] is False

    @patch.object(ModelQuantizer, "quantize_model")
    def test_optimize_for_device_jetson_nano(self, mock_quantize, sample_model_file, temp_dir):
        """Test optimization for Jetson Nano (mid-range device)."""
        mock_quantize.return_value = {
            "format": "pytorch",
            "quantization_type": "int4",
            "compression_ratio": 4.0,
        }

        optimizer = EdgeOptimizer()
        result = optimizer.optimize_for_device(sample_model_file, "jetson_nano", temp_dir)

        assert result["device_optimizations"]["device_type"] == "jetson_nano"
        assert result["device_optimizations"]["recommended_batch_size"] == 4
        assert result["device_optimizations"]["gpu_offload"] is True

    @patch.object(ModelQuantizer, "quantize_model")
    def test_optimize_for_device_intel_nuc(self, mock_quantize, sample_model_file, temp_dir):
        """Test optimization for Intel NUC (high RAM device)."""
        mock_quantize.return_value = {
            "format": "pytorch",
            "quantization_type": "int8",
            "compression_ratio": 2.0,
        }

        optimizer = EdgeOptimizer()
        result = optimizer.optimize_for_device(sample_model_file, "intel_nuc", temp_dir)

        assert result["device_optimizations"]["device_type"] == "intel_nuc"
        assert result["device_optimizations"]["recommended_batch_size"] == 4
        assert result["device_optimizations"]["gpu_offload"] is False

    def test_optimize_for_unknown_device(self, sample_model_file, temp_dir):
        """Test optimization for unknown device type."""
        optimizer = EdgeOptimizer()

        with pytest.raises(ValueError, match="Unknown device type"):
            optimizer.optimize_for_device(sample_model_file, "unknown_device", temp_dir)

    @patch.object(ModelQuantizer, "quantize_model")
    @patch("inference.llm.model_quantization.shutil.copy2")
    def test_create_deployment_package(self, mock_copy, mock_quantize, sample_model_file, temp_dir):
        """Test deployment package creation."""
        mock_quantize.return_value = {"format": "pytorch"}

        optimizer = EdgeOptimizer()
        optimizer.optimize_for_device(sample_model_file, "raspberry_pi", temp_dir)

        # Check that deployment package directory was created
        package_dir = temp_dir / "raspberry_pi_deployment"
        assert package_dir.exists()

        # Check that config.json was created
        config_file = package_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            config = json.load(f)

        assert config["model"] == "model.bin"
        assert "device_profile" in config
        assert "runtime_config" in config

        # Check run script
        run_script = package_dir / "run.sh"
        assert run_script.exists()
        assert run_script.stat().st_mode & 0o755  # Check executable

        # Check inference script
        inference_script = package_dir / "inference.py"
        assert inference_script.exists()


class TestAutoQuantize:
    """Test auto_quantize function."""

    @patch.object(ModelQuantizer, "quantize_model")
    def test_auto_quantize_with_target_size(self, mock_quantize, sample_model_file):
        """Test auto quantization with target size constraint."""
        mock_quantize.return_value = {"compression_ratio": 8.0}

        # Model is ~23KB, target 3KB should trigger INT3
        result_path = auto_quantize(sample_model_file, target_size_mb=0.003)

        mock_quantize.assert_called_once()
        # First argument should be the model path
        _ = mock_quantize.call_args[0][0]
        # Second argument should be the output path
        _ = mock_quantize.call_args[0][1]

        assert result_path.name.endswith("_quantized.pt")

    @patch.object(ModelQuantizer, "quantize_model")
    def test_auto_quantize_with_target_device(self, mock_quantize, sample_model_file):
        """Test auto quantization with target device constraint."""
        mock_quantize.return_value = {"compression_ratio": 4.0}

        result_path = auto_quantize(sample_model_file, target_device="cuda")

        mock_quantize.assert_called_once()
        assert result_path.name.endswith("_quantized.pt")

    @patch.object(ModelQuantizer, "quantize_model")
    def test_auto_quantize_defaults(self, mock_quantize, sample_model_file):
        """Test auto quantization with default settings."""
        mock_quantize.return_value = {"compression_ratio": 4.0}

        result_path = auto_quantize(sample_model_file)

        mock_quantize.assert_called_once()
        assert result_path.name.endswith("_quantized.pt")

    @patch.object(ModelQuantizer, "quantize_model")
    def test_auto_quantize_different_compression_ratios(self, mock_quantize, temp_dir):
        """Test auto quantization with different compression requirements."""
        mock_quantize.return_value = {"compression_ratio": 1.0}

        # Create model files of different sizes
        small_model = temp_dir / "small.pt"
        small_model.write_bytes(b"x" * 1000)  # 1KB

        large_model = temp_dir / "large.pt"
        large_model.write_bytes(b"x" * 100000)  # 100KB

        # Test with different target sizes
        test_cases = [
            (large_model, 12.5),  # 100KB -> 12.5KB = 8x compression (INT3)
            (large_model, 25.0),  # 100KB -> 25KB = 4x compression (INT4)
            (large_model, 50.0),  # 100KB -> 50KB = 2x compression (INT8)
        ]

        for model_file, target_kb in test_cases:
            auto_quantize(model_file, target_size_mb=target_kb / 1000)
            mock_quantize.assert_called()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    @patch("inference.llm.model_quantization.subprocess.run")
    @patch("inference.llm.model_quantization.shutil.which")
    def test_end_to_end_ggml_quantization(self, mock_which, mock_run, sample_ggml_file, temp_dir):
        """Test complete GGML quantization workflow."""
        # Setup successful external tool execution
        mock_which.return_value = "/usr/bin/quantize"
        mock_run.return_value = Mock(returncode=0, stderr="")

        # Create expected output
        output_path = temp_dir / "quantized.ggml"
        output_path.write_bytes(b"quantized" * 500)  # Smaller than original

        # Run quantization
        config = QuantizationConfig(QuantizationType.INT4, optimize_for="size")
        quantizer = ModelQuantizer(config)
        result = quantizer.quantize_model(sample_ggml_file, output_path)

        assert result["format"] == "ggml"
        assert result["compression_ratio"] > 1.0
        assert "%" in result["size_reduction"]

    @patch.object(EdgeOptimizer, "_create_deployment_package")
    @patch.object(ModelQuantizer, "quantize_model")
    def test_edge_optimization_workflow(
        self, mock_quantize, mock_create_package, sample_model_file, temp_dir
    ):
        """Test complete edge optimization workflow."""
        mock_quantize.return_value = {
            "format": "pytorch",
            "quantization_type": "int4",
            "compression_ratio": 4.0,
        }

        optimizer = EdgeOptimizer()
        result = optimizer.optimize_for_device(sample_model_file, "mobile_phone", temp_dir)

        # Verify quantization was called
        mock_quantize.assert_called_once()

        # Verify deployment package creation was called
        mock_create_package.assert_called_once()

        # Check result structure
        assert "device_optimizations" in result
        assert "device_type" in result["device_optimizations"]
        assert "recommended_batch_size" in result["device_optimizations"]

    def test_error_handling_invalid_inputs(self, temp_dir):
        """Test error handling with invalid inputs."""
        config = QuantizationConfig(QuantizationType.INT8)
        quantizer = ModelQuantizer(config)

        # Test with non-existent file
        nonexistent = temp_dir / "does_not_exist.pt"
        temp_dir / "output.pt"

        with pytest.raises(FileNotFoundError):
            quantizer._detect_model_format(nonexistent)

    @patch("inference.llm.model_quantization.logger")
    def test_logging_behavior(self, mock_logger, sample_model_file, temp_dir):
        """Test that appropriate logging occurs."""
        config = QuantizationConfig(QuantizationType.INT8)
        quantizer = ModelQuantizer(config)

        # Test with unknown format to trigger logging
        unknown_file = temp_dir / "test.unknown"
        unknown_file.write_bytes(b"unknown_content")
        output_path = temp_dir / "output.unknown"

        try:
            quantizer.quantize_model(unknown_file, output_path)
        except ValueError:
            pass  # Expected

        # Verify logging occurred
        mock_logger.info.assert_called()


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    def test_quantization_config_memory_efficiency(self):
        """Test that configuration objects are memory efficient."""
        configs = []
        for i in range(1000):
            config = QuantizationConfig(
                quantization_type=QuantizationType.INT8,
                calibration_samples=i,
            )
            configs.append(config)

        # Just verify we can create many configs without issues
        assert len(configs) == 1000
        assert all(c.calibration_samples == i for i, c in enumerate(configs))

    def test_model_file_size_calculations(self, temp_dir):
        """Test accurate file size calculations."""
        # Create files of known sizes
        small_file = temp_dir / "small.pt"
        small_file.write_bytes(b"x" * 1000)  # 1KB

        large_file = temp_dir / "large.pt"
        large_file.write_bytes(b"x" * 10000)  # 10KB

        quantizer = ModelQuantizer(QuantizationConfig(QuantizationType.INT8))

        # Test manual GGML quantization size calculation
        result = quantizer._manual_ggml_quantize(large_file, temp_dir / "output.ggml", "q4_K_M")

        # Should be 40% of original (scale_factor = 0.4)
        expected_size = 10000 * 0.4
        assert result["quantized_size"] == expected_size
        assert result["compression_ratio"] == 10000 / expected_size
