"""
NPU Bridge — calls Windows-side ONNX Runtime via PowerShell.

When the Ryzen AI SDK is installed on Windows, this module:
1. Exports a PyTorch model to ONNX
2. Calls Windows PowerShell to run inference with VitisAIExecutionProvider
3. Returns results to WSL2

Falls back gracefully if NPU is unavailable.
"""

import subprocess
import json
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# PS1 script template that loads an ONNX model and runs inference
_INFERENCE_SCRIPT = r"""
$ErrorActionPreference = "Stop"
conda activate ryzen-ai 2>$null

python -c @"
import sys, json, numpy as np, onnxruntime as ort

model_path = sys.argv[1]
inputs_path = sys.argv[2]
output_path = sys.argv[3]

providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession(model_path, providers=providers)

inputs = np.load(inputs_path, allow_pickle=True).item()
results = sess.run(None, inputs)

out = {name: arr.tolist() for name, arr in zip([o.name for o in sess.get_outputs()], results)}
with open(output_path, 'w') as f:
    json.dump(out, f)
print("OK")
"@ $(wslpath -w '{model_path}') $(wslpath -w '{inputs_path}') $(wslpath -w '{output_path}')
"""


class NPUBridge:
    """Bridge to AMD XDNA 2 NPU (Ryzen AI HX 370) via Windows ONNX Runtime."""

    def __init__(self):
        self._available = False
        self._checked = False
        self._powershell = POWERSHELL
        self._check_npu()

    def _check_npu(self) -> None:
        """Check if the NPU device is present on the Windows side."""
        self._checked = True
        if not os.path.isfile(self._powershell):
            return
        try:
            result = subprocess.run(
                [self._powershell, "-Command",
                 "Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -eq 'NPU Compute Accelerator Device' -and $_.Status -eq 'OK' } | Select-Object -First 1"],
                capture_output=True, text=True, timeout=10,
            )
            self._available = "NPU Compute Accelerator Device" in result.stdout
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def requires_sdk(self) -> bool:
        """True if NPU hardware is present but Ryzen AI SDK hasn't been installed yet."""
        return self._available  # hardware OK, but SDK install needed for actual use

    def _wsl_to_win(self, wsl_path: str) -> str:
        """Convert a WSL path to a Windows path."""
        result = subprocess.run(
            ["wslpath", "-w", wsl_path],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()

    def run_inference(
        self,
        onnx_model_path: str,
        input_data: Dict[str, np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """
        Run an ONNX model on the NPU via Windows PowerShell + Ryzen AI SDK.

        Args:
            onnx_model_path: Path to the ONNX model file (WSL path).
            input_data: Dict of input name -> numpy array.

        Returns:
            Dict of output name -> value, or None if NPU unavailable.
        """
        if not self._available:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_path = os.path.join(tmpdir, "inputs.npz")
            output_path = os.path.join(tmpdir, "outputs.json")

            # Save inputs
            np.savez(inputs_path, **input_data)

            # Build and run PowerShell script
            script = _INFERENCE_SCRIPT.format(
                model_path=onnx_model_path,
                inputs_path=inputs_path,
                output_path=output_path,
            )

            result = subprocess.run(
                [self._powershell, "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"NPU inference failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )

            with open(output_path) as f:
                return json.load(f)

    def export_and_run(
        self,
        torch_model,
        dummy_input,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Export a PyTorch model to ONNX and run it on the NPU.

        Args:
            torch_model: A torch.nn.Module.
            dummy_input: Example input tensor(s) for tracing.
            input_names: Optional names for ONNX inputs.
            output_names: Optional names for ONNX outputs.

        Returns:
            Dict of output name -> value, or None if NPU unavailable.
        """
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            torch.onnx.export(
                torch_model,
                dummy_input,
                onnx_path,
                input_names=input_names or ["input"],
                output_names=output_names or ["output"],
                opset_version=14,
            )

            # Build input dict
            if isinstance(dummy_input, dict):
                input_data = {k: v.numpy() for k, v in dummy_input.items()}
            elif isinstance(dummy_input, (list, tuple)):
                names = input_names or [f"input_{i}" for i in range(len(dummy_input))]
                input_data = {n: t.numpy() for n, t in zip(names, dummy_input)}
            else:
                name = (input_names or ["input"])[0]
                input_data = {name: dummy_input.numpy()}

            return self.run_inference(onnx_path, input_data)


# Singleton for easy import
bridge = NPUBridge()
