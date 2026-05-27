"""Tests for plato_training.npu_bridge — NPU bridge to Windows ONNX Runtime."""

import subprocess
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from plato_training.npu_bridge import NPUBridge, POWERSHELL


class TestNPUBridge:
    def test_not_available_when_no_powershell(self):
        with patch("os.path.isfile", return_value=False):
            bridge = NPUBridge()
            assert bridge.available is False

    def test_repr(self):
        with patch("os.path.isfile", return_value=False):
            bridge = NPUBridge()
            assert "NPUBridge" in repr(bridge)

    def test_run_inference_returns_none_when_unavailable(self):
        with patch("os.path.isfile", return_value=False):
            bridge = NPUBridge()
            result = bridge.run_inference("model.onnx", {"input": np.zeros((1, 3))})
            assert result is None

    def test_requires_sdk_property(self):
        with patch("os.path.isfile", return_value=False):
            bridge = NPUBridge()
            assert bridge.requires_sdk is False

    def test_available_when_npu_detected(self):
        with patch("plato_training.npu_bridge.os.path.isfile", return_value=True), \
             patch("plato_training.npu_bridge.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="NPU Compute Accelerator Device  OK  True"
            )
            bridge = NPUBridge()
            assert bridge.available is True
            assert bridge.requires_sdk is True

    def test_unavailable_when_npu_not_detected(self):
        with patch("plato_training.npu_bridge.os.path.isfile", return_value=True), \
             patch("plato_training.npu_bridge.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            bridge = NPUBridge()
            assert bridge.available is False

    def test_exception_during_check_means_unavailable(self):
        with patch("plato_training.npu_bridge.os.path.isfile", return_value=True), \
             patch("plato_training.npu_bridge.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("timeout")
            bridge = NPUBridge()
            assert bridge.available is False


class TestPowerShellConstant:
    def test_path_value(self):
        assert "powershell" in POWERSHELL.lower()
