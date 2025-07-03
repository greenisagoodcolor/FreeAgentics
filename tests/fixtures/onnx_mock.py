"""
Mock for ONNX Runtime to handle import errors gracefully.

This module provides mocks for ONNX Runtime functionality when the library
is not installed, allowing tests to run without the dependency.
"""

import sys
from unittest.mock import Mock


class MockQuantType:
    """Mock for onnxruntime.quantization.QuantType"""

    QInt8 = "qint8"
    QUInt8 = "quint8"
    QInt16 = "qint16"
    QUInt16 = "quint16"


def mock_quantize_dynamic(model_input, model_output, **kwargs):
    """Mock for onnxruntime.quantization.quantize_dynamic"""
    # Simulate successful quantization by creating output file
    if hasattr(model_output, "write_bytes"):
        model_output.write_bytes(b"mock_quantized_onnx_model")
    elif isinstance(model_output, str):
        with open(model_output, "wb") as f:
            f.write(b"mock_quantized_onnx_model")
    return None


class MockONNXRuntime:
    """Mock for the entire onnxruntime module"""

    class quantization:
        """Mock quantization submodule"""

        QuantType = MockQuantType
        quantize_dynamic = staticmethod(mock_quantize_dynamic)

        @staticmethod
        def quantize_static(model_input, model_output, calibration_data_reader, **kwargs):
            """Mock for static quantization"""
            mock_quantize_dynamic(model_input, model_output)

    class InferenceSession:
        """Mock InferenceSession for runtime testing"""

        def __init__(self, model_path, providers=None):
            self.model_path = model_path
            self.providers = providers or ["CPUExecutionProvider"]

        def run(self, output_names, input_feed):
            """Mock inference run"""
            # Return dummy outputs
            return [[[0.1, 0.2, 0.3, 0.4]]]

        def get_inputs(self):
            """Mock get input metadata"""
            mock_input = Mock()
            mock_input.name = "input"
            mock_input.shape = [1, 3, 224, 224]
            mock_input.type = "tensor(float)"
            return [mock_input]

        def get_outputs(self):
            """Mock get output metadata"""
            mock_output = Mock()
            mock_output.name = "output"
            mock_output.shape = [1, 1000]
            mock_output.type = "tensor(float)"
            return [mock_output]


def install_onnx_mock():
    """Install the ONNX mock into sys.modules"""
    # Create the mock module
    mock_onnxruntime = Mock(spec=["quantization", "InferenceSession"])
    mock_onnxruntime.quantization = MockONNXRuntime.quantization
    mock_onnxruntime.InferenceSession = MockONNXRuntime.InferenceSession

    # Install into sys.modules
    sys.modules["onnxruntime"] = mock_onnxruntime
    sys.modules["onnxruntime.quantization"] = MockONNXRuntime.quantization

    return mock_onnxruntime


def uninstall_onnx_mock():
    """Remove the ONNX mock from sys.modules"""
    modules_to_remove = ["onnxruntime", "onnxruntime.quantization"]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
