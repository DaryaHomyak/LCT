#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный интегрированный архивный процессор
Версия с angle classifier и улучшенной детекцией
"""

import re
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv


load_dotenv()
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrehistoricalTextProcessor:
    """Постобработчик дореволюционных текстов"""

    def __init__(self):
        # Мапинг старых букв на новые
        self.old_to_new = {
            'ѣ': 'е', 'ѳ': 'ф', 'і': 'и', 'ѵ': 'и',
            'ъ': '', 'Ѣ': 'Е', 'Ѳ': 'Ф', 'І': 'И', 'Ѵ': 'И'
        }

        # Паттерны для исправления OCR ошибок
        self.error_patterns = [
            (r'([а-яё])1', r'і'),  # цифра 1 → і
            (r'о([рн])', r'он'),   # пропущенная н
            (r'([аеиоуыэюя])11([а-яё])', r'ии'),  # 11 → ии
            (r'рг', 'гг'),           # рг → гг
            (r'оо', 'со'),           # оо → со
        ]

    def clean_text(self, text: str) -> str:
        """Очистка и модернизация текста"""
        if not text:
            return text

        result = text

        # Замена старых букв
        for old, new in self.old_to_new.items():
            result = result.replace(old, new)

        # Исправление типичных OCR ошибок
        for pattern, replacement in self.error_patterns:
            result = re.sub(pattern, replacement, result)

        # Очистка от лишних символов
        result = re.sub(r'''[^\w\s\-.,;:!?()«»""']''', ' ', result)
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()

        return result

class LinearSegmentator:
    """Упрощенная сегментация по строкам"""

    def __init__(self):
        self.min_line_height = 10
        self.max_line_gap = 50

    def process_segmentation(self, image):
        """Сегментация изображения на строки"""
        try:
            height, width = image.shape[:2]

            # Получаем горизонтальные проекции
            horizontal_projection = np.sum(image < 128, axis=1)

            # Находим строки текста
            lines = []
            in_line = False
            line_start = 0

            threshold = width * 0.1  # Минимум 10% пикселей для строки

            for i, proj in enumerate(horizontal_projection):
                if proj > threshold and not in_line:
                    # Начало строки
                    line_start = i
                    in_line = True
                elif proj <= threshold and in_line:
                    # Конец строки
                    line_end = i
                    if line_end - line_start >= self.min_line_height:
                        lines.append((line_start, line_end))
                    in_line = False

            # Если последняя строка не закрылась
            if in_line:
                lines.append((line_start, height))

            # Если строк не найдено, возвращаем весь образ
            if not lines:
                lines = [(0, height)]

            # Вырезаем сегменты
            segments = []
            for i, (start, end) in enumerate(lines):
                segment = image[start:end, :]
                segments.append({
                    'id': i + 1,
                    'image': segment,
                    'bbox': (0, start, width, end),
                    'shape': segment.shape
                })

            return segments

        except Exception as e:
            logger.error(f"Ошибка сегментации: {e}")
            # Возвращаем весь образ как один сегмент
            return [{
                'id': 1,
                'image': image,
                'bbox': (0, 0, image.shape[1], image.shape[0]),
                'shape': image.shape
            }]

class ImprovedOCREngine:
    """Улучшенный OCR движок с angle classifier"""

    def __init__(self):
        self.paddle_ocr = None
        self._initialize_paddle()

    def _initialize_paddle(self):
        """Инициализация PaddleOCR с angle classifier"""
        try:
            from paddleocr import PaddleOCR

            # ИСПРАВЛЕНО: включаем angle classifier и улучшаем настройки
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,  # ВКЛЮЧЕН angle classifier
                lang='ru',
                det_db_thresh=0.2,       # Понижен порог детекции
                det_db_box_thresh=0.3,   # Понижен порог боксов
                det_db_unclip_ratio=2.2, # Увеличено для лучшего охвата
                max_side_len=4096,       # Увеличено максимальное разрешение
                cls_thresh=0.9,          # Порог классификатора углов
                rec_image_shape='3, 48, 320',  # Форма для распознавания
                drop_score=0.3           # Понижен drop_score для большего recall
            )
            logger.info("PaddleOCR инициализирован с angle classifier")

        except ImportError:
            logger.warning("PaddleOCR не найден. OCR будет недоступен.")
            self.paddle_ocr = None
        except Exception as e:
            logger.error(f"Ошибка инициализации PaddleOCR: {e}")
            self.paddle_ocr = None

    def recognize_text(self, image_segment) -> Dict[str, Any]:
        """Распознавание текста с fallback обработкой"""
        if self.paddle_ocr is None:
            return {
                'raw_text': 'OCR недоступен',
                'confidence': 0.0,
                'boxes': []
            }

        try:
            # Основная попытка распознавания
            result = self.paddle_ocr.ocr(image_segment, cls=True)

            # Если результат пустой, пробуем улучшенную обработку
            if not result or not result[0]:
                logger.info("Первая попытка OCR неудачна, применяем улучшения...")
                enhanced_image = self._enhance_image_for_ocr(image_segment)
                result = self.paddle_ocr.ocr(enhanced_image, cls=True)

            # Если все еще пусто, пробуем с другими параметрами
            if not result or not result[0]:
                logger.info("Вторая попытка OCR, используем агрессивные настройки...")
                # Временно меняем настройки для сложных случаев
                self.paddle_ocr.det_db_thresh = 0.1
                self.paddle_ocr.det_db_box_thresh = 0.2
                result = self.paddle_ocr.ocr(image_segment, cls=True)
                # Возвращаем настройки
                self.paddle_ocr.det_db_thresh = 0.2
                self.paddle_ocr.det_db_box_thresh = 0.3

            return self._process_paddle_result(result)

        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            return {
                'raw_text': f'Ошибка OCR: {str(e)}',
                'confidence': 0.0,
                'boxes': []
            }

    def _enhance_image_for_ocr(self, image):
        """Улучшение изображения для OCR"""
        try:
            # Увеличение изображения
            height, width = image.shape[:2]
            scale_factor = 1.7
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # CLAHE для улучшения контраста
            if len(upscaled.shape) == 3:
                lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(upscaled)

            # Дополнительная фильтрация
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

            return enhanced

        except Exception as e:
            logger.warning(f"Ошибка улучшения изображения: {e}")
            return image

    def _process_paddle_result(self, result) -> Dict[str, Any]:
        """Обработка результата PaddleOCR"""
        if not result or not result[0]:
            return {
                'raw_text': '',
                'confidence': 0.0,
                'boxes': []
            }

        texts = []
        confidences = []
        boxes = []

        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]
                text_info = line[1]

                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                elif isinstance(text_info, str):
                    text = text_info
                    confidence = 0.5  # Дефолтная уверенность
                else:
                    continue

                if text and text.strip():
                    texts.append(text.strip())
                    confidences.append(float(confidence))
                    boxes.append(bbox)

        # Объединяем тексты
        combined_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'raw_text': combined_text,
            'confidence': avg_confidence,
            'boxes': boxes
        }

class ArchiveDocumentProcessor:
    """Главный процессор архивных документов"""

    def __init__(self, use_postprocessing=True):
        logger.info("Инициализация исправленной системы обработки архивных документов")

        self.use_postprocessing = use_postprocessing
        self.segmentator = LinearSegmentator()
        self.ocr_engine = ImprovedOCREngine()
        self.text_processor = PrehistoricalTextProcessor() if use_postprocessing else None

        # Счетчики для статистики
        self.processed_documents = 0
        self.total_processing_time = 0.0

    def process_document(self, image_path: str) -> List[Dict[str, Any]]:
        """Основная функция обработки документа"""
        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Не удалось загрузить изображение: {image_path}")
                return []

            # Конвертация в серый
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            logger.info(f"Обработка изображения {image_path}, размер: {gray.shape}")

            # Сегментация
            segments = self.segmentator.process_segmentation(gray)
            logger.info(f"Найдено сегментов: {len(segments)}")

            results = []

            for segment_info in segments:
                segment_id = segment_info['id']
                segment_image = segment_info['image']
                segment_shape = segment_info['shape']

                logger.info(f"Обработка сегмента {segment_id}, размер: {segment_shape}")

                # OCR распознавание
                ocr_result = self.ocr_engine.recognize_text(segment_image)
                raw_text = ocr_result['raw_text']
                confidence = ocr_result['confidence']

                # Постобработка текста
                cleaned_text = raw_text
                if self.text_processor and raw_text:
                    cleaned_text = self.text_processor.clean_text(raw_text)

                # Оценка уверенности (улучшенная)
                confidence_estimate = self._calculate_confidence(raw_text, cleaned_text, confidence)

                segment_result = {
                    'segment_id': segment_id,
                    'raw_text': raw_text,
                    'cleaned_text': cleaned_text,
                    'segment_shape': segment_shape,
                    'confidence_estimate': confidence_estimate,
                    'ocr_boxes': ocr_result.get('boxes', [])
                }

                results.append(segment_result)

                logger.info(f"Сегмент {segment_id}: уверенность {confidence_estimate:.3f}, "
                           f"символов: {len(raw_text)}")

            self.processed_documents += 1
            logger.info(f"Документ обработан. Найдено {len(results)} сегментов текста")

            return results

        except Exception as e:
            logger.error(f"Ошибка обработки документа {image_path}: {e}")
            # Возвращаем хотя бы один сегмент с ошибкой
            return [{
                'segment_id': 1,
                'raw_text': f'Ошибка обработки: {str(e)}',
                'cleaned_text': f'Ошибка обработки: {str(e)}',
                'segment_shape': (100, 100),
                'confidence_estimate': 0.0,
                'ocr_boxes': []
            }]

    def _calculate_confidence(self, raw_text: str, cleaned_text: str, ocr_confidence: float) -> float:
        """Расчет итоговой уверенности"""
        if not raw_text:
            return 0.0

        # Базовая уверенность от OCR
        base_confidence = ocr_confidence

        # Бонус за длину текста
        length_bonus = min(len(raw_text) / 100.0, 0.2)

        # Штраф за слишком много цифр (возможно, шум)
        digit_ratio = sum(c.isdigit() for c in raw_text) / len(raw_text)
        digit_penalty = digit_ratio * 0.3

        # Бонус за наличие словарных слов
        word_bonus = 0.0
        words = raw_text.split()
        if len(words) > 0:
            # Простая проверка на русские слова
            russian_words = sum(1 for word in words if len(word) > 2 and
                              any(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word.lower()))
            word_bonus = min(russian_words / len(words) * 0.3, 0.3)

        # Итоговая уверенность
        final_confidence = base_confidence + length_bonus + word_bonus - digit_penalty

        return max(0.0, min(1.0, final_confidence))

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика обработки"""
        return {
            'processed_documents': self.processed_documents,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': (self.total_processing_time / max(self.processed_documents, 1))
        }

# Тестирование (если запускается напрямую)
if __name__ == "__main__":
    processor = ArchiveDocumentProcessor()

    # Тестовая обработка
    test_image = "test_image.jpg"
    if Path(test_image).exists():
        results = processor.process_document(test_image)

        print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
        print(f"Обработано сегментов: {len(results)}")

        for result in results:
            print(f"\n--- Сегмент {result['segment_id']} ---")
            print(f"Исходный текст: {result['raw_text'][:100]}...")
            print(f"Очищенный текст: {result['cleaned_text'][:100]}...")
            print(f"Уверенность: {result['confidence_estimate']:.3f}")
    else:
        print(f"Тестовое изображение {test_image} не найдено")
        print("Создайте тестовое изображение или укажите путь к существующему файлу")
