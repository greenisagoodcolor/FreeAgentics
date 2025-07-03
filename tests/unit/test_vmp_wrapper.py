"""
Test the VMP wrapper module without PyTorch dependencies
"""

import sys
from unittest.mock import Mock, patch


class TestVMPWrapper:
    """Test the variational_message_passing wrapper module"""

    def test_module_imports(self):
        """Test that the wrapper module exports the correct items"""
        # Mock the active_inference module to avoid PyTorch
        mock_inference_config = Mock()
        mock_vmp = Mock()

        with patch.dict(
            "sys.modules",
            {
                "inference.engine.active_inference": Mock(
                    InferenceConfig=mock_inference_config, VariationalMessagePassing=mock_vmp
                )
            },
        ):
            # Import should work
            from inference.algorithms import variational_message_passing

            # Check exports
            assert hasattr(variational_message_passing, "InferenceConfig")
            assert hasattr(variational_message_passing, "VariationalMessagePassing")
            assert variational_message_passing.InferenceConfig is mock_inference_config
            assert variational_message_passing.VariationalMessagePassing is mock_vmp

    def test_module_all_exports(self):
        """Test __all__ exports"""
        with patch.dict(
            "sys.modules",
            {
                "inference.engine.active_inference": Mock(
                    InferenceConfig=Mock(), VariationalMessagePassing=Mock()
                )
            },
        ):
            from inference.algorithms import variational_message_passing

            assert hasattr(variational_message_passing, "__all__")
            assert "InferenceConfig" in variational_message_passing.__all__
            assert "VariationalMessagePassing" in variational_message_passing.__all__
            assert len(variational_message_passing.__all__) == 2

    def test_backward_compatibility(self):
        """Test that the wrapper provides backward compatibility"""
        # This tests that the old import path still works
        mock_config = Mock(name="InferenceConfig")
        mock_vmp = Mock(name="VariationalMessagePassing")

        with patch.dict(
            "sys.modules",
            {
                "inference.engine.active_inference": Mock(
                    InferenceConfig=mock_config, VariationalMessagePassing=mock_vmp
                )
            },
        ):
            # Old import path
            from inference.algorithms.variational_message_passing import (
                InferenceConfig,
                VariationalMessagePassing,
            )

            assert InferenceConfig is mock_config
            assert VariationalMessagePassing is mock_vmp

    def test_module_docstring(self):
        """Test module has proper docstring"""
        with patch.dict(
            "sys.modules",
            {
                "inference.engine.active_inference": Mock(
                    InferenceConfig=Mock(), VariationalMessagePassing=Mock()
                )
            },
        ):
            from inference.algorithms import variational_message_passing

            assert variational_message_passing.__doc__ is not None
            assert "backward compatibility" in variational_message_passing.__doc__.lower()
