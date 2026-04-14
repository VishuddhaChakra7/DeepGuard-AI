"""
DeepGuard AI — Inference
=========================
Usage:
    python predict.py --image face.jpg
    python predict.py --image face.jpg --model best_model.pth --gradcam
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import IMG_SIZE, get_transforms
from model import DeepGuardNet
from utils import setup_logging

logger = setup_logging()


class DeepGuardPredictor:
    """
    Wraps a trained DeepGuardNet checkpoint for inference.

    Transforms are built once at construction time (not per-call).
    """

    def __init__(self, model_path: str = "best_model.pth", device: str | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please run train.py first."
            )

        logger.info(f"Loading model from {ckpt_path} on {self.device}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        backbone = ckpt.get("backbone", "efficientnet_b4")
        img_size = ckpt.get("image_size", 380)

        self.model = DeepGuardNet(backbone_name=backbone).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        val_auc = ckpt.get("val_auc", float("nan"))
        logger.info(f"Loaded '{backbone}'  Val AUC={val_auc:.4f}")

        # Build transforms once — reused for every prediction
        _, self.transform = get_transforms(img_size)
        self.img_size     = img_size
        self.class_names  = ckpt.get("class_names", ["Real", "Fake"])

    # ── Core predict ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image) -> dict:
        """
        Predict whether an image is a deepfake.

        Args:
            image: PIL.Image or a file path (str / Path).

        Returns:
            {
              "label":      "Real" | "Fake",
              "confidence": float (0-1),
              "fake_prob":  float (0-1),
              "real_prob":  float (0-1),
            }
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)

        fake_prob = float(torch.softmax(logits, 1)[0, 1])
        label     = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = fake_prob if label == "Fake" else 1.0 - fake_prob

        return {
            "label"     : label,
            "confidence": round(confidence, 4),
            "fake_prob" : round(fake_prob, 4),
            "real_prob" : round(1.0 - fake_prob, 4),
        }

    # ── GradCAM visualisation ─────────────────────────────────────────────
    def gradcam(self, image) -> np.ndarray:
        """
        Return a GradCAM heatmap overlaid on the input image as a uint8 RGB array.
        Requires the `grad-cam` package (pip install grad-cam).
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            raise ImportError("Install grad-cam: pip install grad-cam")

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        result = self.predict(image)
        pred_class = 1 if result["label"] == "Fake" else 0

        target_layer = self.model.backbone.conv_head
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

        # Reconstruct un-normalised RGB for overlay
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        rgb  = (tensor[0].cpu().numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)

        import cv2
        rgb_r     = cv2.resize(rgb.astype(np.float32), (self.img_size, self.img_size))
        heatmap_r = cv2.resize(heatmap, (self.img_size, self.img_size))
        return show_cam_on_image(rgb_r, heatmap_r, use_rgb=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepGuard AI — Inference")
    parser.add_argument("--image",   type=str, required=True,              help="Path to input image")
    parser.add_argument("--model",   type=str, default="best_model.pth",   help="Path to checkpoint")
    parser.add_argument("--device",  type=str, default=None,               help="'cuda' or 'cpu'")
    parser.add_argument("--gradcam", action="store_true",                   help="Save GradCAM output")
    args = parser.parse_args()

    predictor = DeepGuardPredictor(model_path=args.model, device=args.device)
    result    = predictor.predict(args.image)

    print("\n─── DeepGuard AI Prediction ───────────────────────")
    print(f"  Image      : {args.image}")
    print(f"  Verdict    : {result['label']}")
    print(f"  Confidence : {result['confidence'] * 100:.2f}%")
    print(f"  Real score : {result['real_prob']:.4f}")
    print(f"  Fake score : {result['fake_prob']:.4f}")
    print("────────────────────────────────────────────────────\n")

    if args.gradcam:
        from PIL import Image as PILImage
        overlay    = predictor.gradcam(args.image)
        out_path   = Path(args.image).stem + "_gradcam.png"
        PILImage.fromarray(overlay).save(out_path)
        print(f"GradCAM saved → {out_path}")
