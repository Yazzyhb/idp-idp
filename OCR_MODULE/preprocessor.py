import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium


def load_document(file_path: str) -> list[np.ndarray]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=300/72)
            pil_image = bitmap.to_pil()
            pages.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return pages
    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        return [img]
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _is_digital(image: np.ndarray) -> bool:
    """True if the image is a clean digital render (not a degraded paper scan)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    # digital PDFs: mostly white background (mean > 200), low noise (std < 60)
    return mean > 200 and std < 60


def _sharpen(image: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask — safe for both digital and scanned docs."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess(image: np.ndarray) -> np.ndarray:
    if _is_digital(image):
        # clean digital render — heavy preprocessing hurts quality, just sharpen
        return _sharpen(image)
    # degraded paper scan — full pipeline
    image = deskew(image)
    image = denoise(image)
    image = binarize(image)
    return image


def preprocess_document(file_path: str) -> list[np.ndarray]:
    pages = load_document(file_path)
    return [preprocess(page) for page in pages]


def detect_and_correct_rotated_regions(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > image.shape[0] * image.shape[1] * 0.1:
            continue
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) > 5:
            x, y, w, h = cv2.boundingRect(contour)
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue
            M = cv2.getRotationMatrix2D((region.shape[1]//2, region.shape[0]//2), angle, 1.0)
            result[y1:y2, x1:x2] = cv2.warpAffine(
                region, M, (region.shape[1], region.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    return result


def detect_circular_stamps(image: np.ndarray) -> list[tuple]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=30, maxRadius=200
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype("int")
    return [(x, y, r) for x, y, r in circles]
