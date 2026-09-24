"""Module 1: Image acquisition and preprocessing.

Loads images and applies noise removal / resizing before they are handed
off to the detection and classification modules.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove sensor/compression noise while preserving edges."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """Light preprocessing used before running detection: denoise only.

    Detection needs the image at its original scale so bounding boxes stay
    meaningful, so no resizing happens here.
    """
    return denoise(image)


def preprocess_crop_for_classifier(crop: np.ndarray) -> np.ndarray:
    """Preprocess a single detected product crop before classification."""
    denoised = denoise(crop)
    return resize(denoised, (256, 256))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preview preprocessing on a single image.")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--out", default="outputs/preprocessed_preview.png")
    args = parser.parse_args()

    img = load_image(args.image)
    processed = preprocess_for_detection(img)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, processed)
    print(f"Saved preprocessed preview to {args.out}")
