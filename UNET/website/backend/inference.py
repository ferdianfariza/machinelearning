from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image


@dataclass
class InferenceResult:
    mask_base64: str
    threshold: float
    inference_ms: float
    width: int
    height: int


class FloodSegmenter:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        image_size: int = 256,
        threshold: float | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = self._build_model().to(self.device)

        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        self.image_size = int(config.get("IMAGE_SIZE", image_size))
        if threshold is None:
            self.threshold = float(config.get("THRESHOLD", 0.5))
        else:
            self.threshold = float(threshold)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _build_model() -> torch.nn.Module:
        return smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        resolved = checkpoint_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")

        checkpoint = torch.load(resolved, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint format is invalid. Expected a dict object.")
        return checkpoint

    @staticmethod
    def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError("Could not extract state_dict from checkpoint.")
        return state_dict

    @staticmethod
    def _encode_png_base64(mask: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("Failed to encode mask as PNG.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def predict(self, image: Image.Image) -> InferenceResult:
        rgb_image = image.convert("RGB")
        image_np = np.array(rgb_image)
        original_height, original_width = image_np.shape[:2]

        resized = cv2.resize(
            image_np,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        resized = resized.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).unsqueeze(0).to(self.device)

        with torch.no_grad():
            started = time.perf_counter()
            output = self.model(input_tensor)
            inference_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = output
        min_prob = float(probabilities.min().item())
        max_prob = float(probabilities.max().item())
        if min_prob < 0.0 or max_prob > 1.0:
            probabilities = torch.sigmoid(probabilities)

        binary = (probabilities >= self.threshold).to(torch.uint8).squeeze().detach().cpu().numpy()
        binary = (binary * 255).astype(np.uint8)

        binary_original_size = cv2.resize(
            binary,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return InferenceResult(
            mask_base64=self._encode_png_base64(binary_original_size),
            threshold=self.threshold,
            inference_ms=round(float(inference_ms), 2),
            width=original_width,
            height=original_height,
        )
