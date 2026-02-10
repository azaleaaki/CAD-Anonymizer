import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import sys
import os
from pathlib import Path


class GOSTConstants:
    """
    Constants based on GOST 2.104-2006 (Form 2: 185 × 55 mm)
    """
    STAMP_WIDTH_MM = 185.0
    STAMP_HEIGHT_MM = 55.0
    STAMP_ASPECT_RATIO = STAMP_WIDTH_MM / STAMP_HEIGHT_MM  # ≈3.36

    # Relative position of stamp on sheet (bottom-right corner)
    STAMP_RELATIVE_X = 0.75  # 75% from left
    STAMP_RELATIVE_Y = 0.85  # 85% from top

    # Field coordinates relative to the stamp area (x1, y1, x2, y2)
    FIELDS = {
        "project_name":     (0.15, 0.00, 0.75, 0.20),  # Наименование
        "document_code":   (0.75, 0.00, 1.00, 0.20),  # Обозначение
        "developed_fio":    (0.15, 0.20, 0.45, 0.40),  # Разработал (ФИО)
        "checked_fio":      (0.15, 0.40, 0.45, 0.60),  # Проверил (ФИО)
        "norm_control":     (0.15, 0.60, 0.45, 0.80),  # Нормоконтроль
        "approved_fio":     (0.15, 0.80, 0.45, 1.00),  # Утвердил (ФИО)
        "signatures":       (0.45, 0.20, 0.75, 1.00),  # Подписи
        "dates":            (0.75, 0.20, 1.00, 1.00),  # Даты
        "company":          (0.00, 0.20, 0.15, 1.00)   # Организация
    }

    DEFAULT_DPI = 300
    MM_TO_INCH = 25.4
    DPI_SCALING_FACTOR = DEFAULT_DPI / MM_TO_INCH


def compute_normalized_text_mask(image_roi: np.ndarray, min_confidence: int = 30) -> np.ndarray:
    if image_roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = image_roi.shape[:2]
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    
    # Adaptive binarization for robust edge preservation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extract text block coordinates via Tesseract
    data = pytesseract.image_to_data(
        binary,
        output_type=pytesseract.Output.DICT,
        config="--psm 6 --oem 3"
    )
    
    mask = np.zeros((h, w), dtype=np.uint8)
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        
        if conf >= min_confidence and len(text) > 0:
            x = data['left'][i]
            y = data['top'][i]
            w_box = data['width'][i]
            h_box = data['height'][i]
            
            margin = max(1, int(min(w_box, h_box) * 0.1))
            cv2.rectangle(mask, (x - margin, y - margin), (x + w_box + margin, y + h_box + margin), 255, -1)
    
    return mask


class GOSTCompliantAnonymizer:
    def __init__(self):
        self.constants = GOSTConstants()

    def extract_stamp_region(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        x1 = int(self.constants.STAMP_RELATIVE_X * w)
        y1 = int(self.constants.STAMP_RELATIVE_Y * h)
        x2 = w - 10
        y2 = h - 10
        
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return None, 0, 0
        
        roi = image[y1:y2, x1:x2]
        return roi, x1, y1

    def anonymize_field_in_stamp(self, stamp_img: np.ndarray, field_key: str) -> np.ndarray:
        rx1, ry1, rx2, ry2 = self.constants.FIELDS[field_key]
        h, w = stamp_img.shape[:2]
        
        x1 = int(rx1 * w)
        y1 = int(ry1 * h)
        x2 = int(rx2 * w)
        y2 = int(ry2 * h)
        
        if x1 >= x2 or y1 >= y2:
            return stamp_img
        
        roi = stamp_img[y1:y2, x1:x2]
        text_mask_local = compute_normalized_text_mask(roi)
        
        if text_mask_local.sum() == 0:
            return stamp_img
        
        full_mask = np.zeros(stamp_img.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = text_mask_local
        
        return cv2.inpaint(stamp_img, full_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def anonymize_page(self, image: np.ndarray) -> np.ndarray:
        stamp_roi, offset_x, offset_y = self.extract_stamp_region(image)
        
        if stamp_roi is None:
            return image
        
        result = image.copy()
        cleaned_stamp = stamp_roi.copy()
        
        for field_name in self.constants.FIELDS:
            cleaned_stamp = self.anonymize_field_in_stamp(cleaned_stamp, field_name)
        
        h_s, w_s = cleaned_stamp.shape[:2]
        result[offset_y:offset_y + h_s, offset_x:offset_x + w_s] = cleaned_stamp
        
        return result

    def process_pdf(self, input_pdf: str, output_pdf: str) -> bool:
        try:
            pages = convert_from_path(input_pdf, dpi=self.constants.DEFAULT_DPI)
            processed_images = []

            for pil_page in pages:
                cv_img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
                anonymized = self.anonymize_page(cv_img)
                pil_result = Image.fromarray(cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB))
                processed_images.append(pil_result)

            if not processed_images:
                print("Error: No pages were processed.")
                return False

            processed_images[0].save(
                output_pdf,
                save_all=True,
                append_images=processed_images[1:],
                resolution=self.constants.DEFAULT_DPI
            )

            return True

        except Exception as e:
            print(f"Processing failed: {e}")
            return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python anonymize.py input.pdf output.pdf")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    anonymizer = GOSTCompliantAnonymizer()
    success = anonymizer.process_pdf(input_file, output_file)

    if success:
        print(f"Success: {output_file} created.")
        sys.exit(0)
    else:
        print("Failed to create output file.")
        sys.exit(1)


if __name__ == "__main__":
    main()