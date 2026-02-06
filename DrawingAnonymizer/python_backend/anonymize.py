"""
anonymize.py - Интеллектуальная анонимизация технических чертежей PDF
Математически корректная обработка с детекцией штампов и OCR-фильтрацией
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pdf2image import convert_from_path
from PIL import Image
import json
from enum import Enum
import time

# Математические константы
class ProcessingConstants:
    """Константы для математических вычислений"""
    # ISO 216 стандартные размеры бумаги (мм)
    PAPER_SIZES = {
        'A0': (841, 1189),
        'A1': (594, 841),
        'A2': (420, 594),
        'A3': (297, 420),
        'A4': (210, 297)
    }
    
    # DPI для конвертации
    DEFAULT_DPI = 200
    MM_TO_INCH = 25.4
    INCH_TO_PIXEL = DEFAULT_DPI / MM_TO_INCH
    
    # Пропорции штампов (ГОСТ 2.104-2006)
    STAMP_ASPECT_RATIO_MIN = 1.5
    STAMP_ASPECT_RATIO_MAX = 3.0
    STAMP_AREA_MIN = 0.01  # Минимальная площадь относительно страницы
    STAMP_AREA_MAX = 0.15  # Максимальная площадь относительно страницы
    
    # Зоны для анонимизации (нормализованные координаты)
    ANONYMIZATION_ZONES = [
        (0.70, 0.80, 1.00, 1.00),   # Нижний правый угол (основной штамп)
        (0.70, 0.00, 1.00, 0.15),   # Верхний правый угол (вспомогательный штамп)
        (0.00, 0.00, 0.10, 0.10),   # Верхний левый угол (логотип)
        (0.00, 0.90, 0.20, 1.00),   # Нижний левый угол (копировальный штамп)
    ]

class ProcessingMode(Enum):
    """Режимы обработки"""
    ZONES_ONLY = "zones_only"      # Только фиксированные зоны
    AUTO_DETECT = "auto_detect"    # Автоматическая детекция
    HYBRID = "hybrid"              # Гибридный режим (зоны + детекция)


@dataclass
class ProcessingMetrics:
    """Метрики обработки"""
    total_pages: int = 0
    processed_pages: int = 0
    total_time: float = 0.0
    avg_time_per_page: float = 0.0
    detected_stamps: int = 0
    rotation_angle: float = 0.0
    success_rate: float = 0.0


class MathematicalImageProcessor:
    """Математически корректный обработчик изображений"""
    
    @staticmethod
    def calculate_rotation_matrix(angle_degrees: float, 
                                  center: Tuple[float, float],
                                  scale: float = 1.0) -> np.ndarray:
        """
        Вычисление матрицы вращения с учётом центра вращения
        
        Args:
            angle_degrees: Угол в градусах
            center: Центр вращения (x, y)
            scale: Масштаб
            
        Returns:
            Матрица преобразования 2x3
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad) * scale
        sin_a = np.sin(angle_rad) * scale
        
        # Матрица вращения с учётом центра
        rotation_matrix = np.array([
            [cos_a, -sin_a, (1 - cos_a) * center[0] + sin_a * center[1]],
            [sin_a, cos_a, (1 - cos_a) * center[1] - sin_a * center[0]]
        ])
        
        return rotation_matrix
    
    @staticmethod
    def compute_angle_statistics(angles: np.ndarray) -> Dict[str, float]:
        """
        Вычисление статистики углов
        
        Args:
            angles: Массив углов в градусах
            
        Returns:
            Словарь со статистикой
        """
        if len(angles) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'iqr': 0.0,
                'skewness': 0.0
            }
        
        # Основные статистики
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        std_angle = np.std(angles)
        
        # Интерквартильный размах
        q75, q25 = np.percentile(angles, [75, 25])
        iqr_angle = q75 - q25
        
        # Асимметрия (skewness)
        n = len(angles)
        if n > 1 and std_angle > 0:
            skewness = np.sum((angles - mean_angle) ** 3) / (n * std_angle ** 3)
        else:
            skewness = 0.0
        
        return {
            'mean': float(mean_angle),
            'median': float(median_angle),
            'std': float(std_angle),
            'iqr': float(iqr_angle),
            'skewness': float(skewness)
        }


class StampDetector:
    """Детектор штампов на технических чертежах"""
    
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.constants = ProcessingConstants
        
    def detect_stamps(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция штампов на изображении
        
        Args:
            image: Изображение BGR
            
        Returns:
            Список bounding boxes (x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        stamps = []
        
        # Преобразование в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Морфологические операции для улучшения контуров
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Фильтрация контуров
        for contour in contours:
            # Аппроксимация контура
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Проверка на прямоугольность
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                x2, y2 = x + bw, y + bh
                
                # Вычисление характеристик
                area = bw * bh
                aspect_ratio = bw / bh if bh > 0 else 0
                
                # Нормализация характеристик
                normalized_area = area / (w * h)
                
                # Проверка критериев штампа
                if (self.constants.STAMP_AREA_MIN <= normalized_area <= self.constants.STAMP_AREA_MAX and
                    self.constants.STAMP_ASPECT_RATIO_MIN <= aspect_ratio <= self.constants.STAMP_ASPECT_RATIO_MAX):
                    
                    # Дополнительная проверка на текстуру (гистограмма градиентов)
                    if self._is_stamp_like(image[y:y2, x:x2]):
                        stamps.append((x, y, x2, y2))
        
        return stamps
    
    def _is_stamp_like(self, region: np.ndarray) -> bool:
        """
        Проверка, похож ли регион на штамп по текстуре
        
        Args:
            region: Регион изображения
            
        Returns:
            True если похож на штамп
        """
        if region.size == 0:
            return False
        
        # Вычисление градиентов
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Магнитуда и направление градиентов
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        
        # Статистики градиентов
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        
        # Штампы обычно имеют чёткие границы и текст
        return mean_magnitude > 20 and std_magnitude > 10


class BlueprintAnonymizer:
    """Анонимизатор технических чертежей"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.HYBRID):
        self.mode = mode
        self.metrics = ProcessingMetrics()
        self.detector = StampDetector()
        self.constants = ProcessingConstants
        
    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Коррекция наклона изображения с вычислением статистики
        
        Args:
            image: Изображение BGR
            
        Returns:
            Корректированное изображение и угол поворота
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Детекция краев с адаптивными параметрами
        edges = cv2.Canny(
            gray, 
            threshold1=np.percentile(gray, 25),
            threshold2=np.percentile(gray, 75)
        )
        
        # Детекция линий с улучшенными параметрами
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=max(image.shape) * 0.1,
            maxLineGap=20
        )
        
        if lines is not None:
            # Вычисление углов для всех линий
            angles = []
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            angles_array = np.array(angles)
            
            # Фильтрация выбросов
            q1, q3 = np.percentile(angles_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_angles = angles_array[(angles_array >= lower_bound) & (angles_array <= upper_bound)]
            
            # Использование медианы отфильтрованных углов
            if len(filtered_angles) > 0:
                rotation_angle = np.median(filtered_angles)
            else:
                rotation_angle = np.median(angles_array)
            
            # Ограничение угла поворота
            rotation_angle = np.clip(rotation_angle, -45, 45)
            
            # Применение поворота
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = MathematicalImageProcessor.calculate_rotation_matrix(
                -rotation_angle, center  # Отрицательный для компенсации
            )
            
            result = cv2.warpAffine(
                image, M, 
                (image.shape[1], image.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            self.metrics.rotation_angle = rotation_angle
            return result, rotation_angle
        
        return image, 0.0
    
    def _create_anonymization_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Создание маски для анонимизации
        
        Args:
            image: Изображение BGR
            
        Returns:
            Маска для инпейнтинга
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.mode in [ProcessingMode.ZONES_ONLY, ProcessingMode.HYBRID]:
            # Добавление фиксированных зон
            for zone in self.constants.ANONYMIZATION_ZONES:
                x1 = int(w * zone[0])
                y1 = int(h * zone[1])
                x2 = int(w * zone[2])
                y2 = int(h * zone[3])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        if self.mode in [ProcessingMode.AUTO_DETECT, ProcessingMode.HYBRID]:
            # Автоматическая детекция штампов
            stamps = self.detector.detect_stamps(image)
            self.metrics.detected_stamps += len(stamps)
            
            for x1, y1, x2, y2 in stamps:
                # Добавление небольшого отступа вокруг штампа
                margin = int(min(x2 - x1, y2 - y1) * 0.1)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def anonymize_page(self, image: np.ndarray) -> np.ndarray:
        """
        Анонимизация одной страницы
        
        Args:
            image: Изображение BGR
            
        Returns:
            Анонимизированное изображение
        """
        # Коррекция наклона
        deskewed, angle = self.deskew(image)
        
        # Создание маски
        mask = self._create_anonymization_mask(deskewed)
        
        # Применение инпейнтинга с адаптивным радиусом
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        mask_ratio = mask_area / total_area
        
        # Адаптивный радиус инпейнтинга
        inpaint_radius = max(3, int(min(deskewed.shape) * 0.005 * mask_ratio))
        
        # Применение инпейнтинга
        result = cv2.inpaint(
            deskewed, mask, 
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA
        )
        
        return result
    
    def process_pdf(self, input_pdf: str, output_pdf: str) -> bool:
        """
        Обработка PDF файла
        
        Args:
            input_pdf: Входной PDF файл
            output_pdf: Выходной PDF файл
            
        Returns:
            True если успешно
        """
        start_time = time.time()
        
        try:
            print("🔍 Анализ PDF документа...")
            print(f"📄 Входной файл: {input_pdf}")
            print(f"💾 Выходной файл: {output_pdf}")
            print(f"🔧 Режим обработки: {self.mode.value}")
            print("-" * 50)
            
            # Конвертация PDF в изображения
            print("1️⃣ Конвертация PDF в изображения...")
            pages = convert_from_path(
                input_pdf, 
                dpi=self.constants.DEFAULT_DPI,
                thread_count=4  # Параллельная обработка
            )
            
            self.metrics.total_pages = len(pages)
            print(f"   ✅ Загружено страниц: {self.metrics.total_pages}")
            
            processed_images = []
            
            # Обработка каждой страницы
            for i, pil_img in enumerate(pages, 1):
                page_start = time.time()
                
                print(f"2️⃣ Обработка страницы {i}/{self.metrics.total_pages}...")
                
                # Конвертация PIL в OpenCV
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Анонимизация
                anonymized = self.anonymize_page(cv_img)
                
                # Конвертация обратно в PIL
                pil_result = Image.fromarray(
                    cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)
                )
                processed_images.append(pil_result)
                
                self.metrics.processed_pages = i
                page_time = time.time() - page_start
                print(f"   ⏱️  Время обработки: {page_time:.2f} сек")
                
                # Прогресс
                progress = i / self.metrics.total_pages * 100
                print(f"   📊 Прогресс: {progress:.1f}%")
                print("-" * 30)
            
            # Сохранение результата
            print("3️⃣ Сохранение результата в PDF...")
            processed_images[0].save(
                output_pdf,
                save_all=True,
                append_images=processed_images[1:],
                resolution=self.constants.DEFAULT_DPI,
                quality=95
            )
            
            # Вычисление метрик
            self.metrics.total_time = time.time() - start_time
            self.metrics.avg_time_per_page = (
                self.metrics.total_time / self.metrics.total_pages 
                if self.metrics.total_pages > 0 else 0
            )
            self.metrics.success_rate = (
                self.metrics.processed_pages / self.metrics.total_pages * 100 
                if self.metrics.total_pages > 0 else 0
            )
            
            # Вывод отчета
            self._print_report()
            
            print(f"🎉 Обработка завершена успешно!")
            print(f"📁 Результат сохранен: {output_pdf}")
            
            return True
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_report(self):
        """Вывод отчета о обработке"""
        print("\n" + "=" * 50)
        print("📊 ОТЧЕТ ОБ ОБРАБОТКЕ")
        print("=" * 50)
        print(f"📄 Страниц обработано: {self.metrics.processed_pages}/{self.metrics.total_pages}")
        print(f"⏱️  Общее время: {self.metrics.total_time:.2f} сек")
        print(f"⏱️  Среднее время на страницу: {self.metrics.avg_time_per_page:.2f} сек")
        print(f"📈 Успешность: {self.metrics.success_rate:.1f}%")
        print(f"🔍 Обнаружено штампов: {self.metrics.detected_stamps}")
        print(f"🔄 Средний угол поворота: {abs(self.metrics.rotation_angle):.2f}°")
        print("=" * 50)


def main():
    """Основная функция программы"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анонимизация технических чертежей PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python anonymize.py input.pdf output.pdf
  python anonymize.py input.pdf output.pdf --mode auto_detect
  python anonymize.py input.pdf output.pdf --mode zones_only
        """
    )
    
    parser.add_argument('input', help='Входной PDF файл')
    parser.add_argument('output', help='Выходной PDF файл')
    parser.add_argument('--mode', 
                       choices=['zones_only', 'auto_detect', 'hybrid'],
                       default='hybrid',
                       help='Режим обработки (по умолчанию: hybrid)')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Файл не найден: {args.input}")
        print(f"📁 Текущая директория: {Path.cwd()}")
        
        # Поиск PDF файлов в директории
        pdf_files = list(Path.cwd().glob("*.pdf"))
        if pdf_files:
            print("📋 Доступные PDF файлы:")
            for pdf in pdf_files:
                print(f"  - {pdf.name}")
        else:
            print("ℹ️  PDF файлы не найдены")
        
        sys.exit(1)
    
    # Проверка расширения
    if input_path.suffix.lower() != '.pdf':
        print(f"⚠️  Предупреждение: файл {args.input} не имеет расширения .pdf")
    
    # Создание анонимизатора
    mode = ProcessingMode(args.mode)
    anonymizer = BlueprintAnonymizer(mode=mode)
    
    # Обработка
    success = anonymizer.process_pdf(str(input_path), args.output)
    
    if success:
        # Сохранение метрик в JSON
        output_path = Path(args.output)
        metrics_file = output_path.with_suffix('.json')
        
        metrics_dict = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'processing_mode': args.mode,
            'metrics': {
                'total_pages': anonymizer.metrics.total_pages,
                'processed_pages': anonymizer.metrics.processed_pages,
                'total_time_seconds': round(anonymizer.metrics.total_time, 2),
                'avg_time_per_page_seconds': round(anonymizer.metrics.avg_time_per_page, 2),
                'detected_stamps': anonymizer.metrics.detected_stamps,
                'rotation_angle_degrees': round(anonymizer.metrics.rotation_angle, 2),
                'success_rate_percent': round(anonymizer.metrics.success_rate, 1)
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Метрики сохранены: {metrics_file}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()