#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенная система обработки архивных документов
Только PaddleOCR + сегментация по вертикальным/горизонтальным линиям

Компоненты:
1. Детекция линий разметки (вертикальные/горизонтальные)
2. Сегментация на ячейки по координатам линий
3. OCR только через PaddleOCR (без Tesseract/EasyOCR)
4. Постобработка через YandexGPT (опционально)

Дата: 2025
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from PIL import Image
import requests
from dotenv import load_dotenv


load_dotenv()  # загрузит .env в os.environ

# Тише логи Paddle (до импорта paddleocr)
os.environ.setdefault('FLAGS_log_level', '3')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentSegmentator:
    """
    Класс для сегментации архивных документов по линиям разметки
    """

    def __init__(self):
        self.original_image = None
        self.debug_images = {}

    def load_image(self, image_path: str) -> np.ndarray:
        """Загружает изображение и возвращает серое изображение"""
        logger.info(f"Загружаем изображение: {image_path}")
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.debug_images['original'] = self.original_image.copy()
        self.debug_images['gray'] = gray.copy()
        return gray

    def detect_lines(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Детекция вертикальных и горизонтальных линий разметки
        Возвращает координаты линий для сегментации
        """
        logger.info("Детектируем линии разметки")

        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)

        # Бинаризация
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 10
        )

        # Детекция вертикальных линий
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0]//30))
        vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # Детекция горизонтальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1]//30, 1))
        horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # Объединяем маски линий
        lines_mask = cv2.bitwise_or(vertical_mask, horizontal_mask)
        self.debug_images['lines_mask'] = lines_mask.copy()

        # Находим координаты вертикальных линий
        vertical_coords = []
        v_projection = np.sum(vertical_mask, axis=0)
        v_threshold = np.max(v_projection) * 0.3
        for x in range(len(v_projection)):
            if v_projection[x] > v_threshold:
                vertical_coords.append(x)

        # Группируем близкие линии и берем средние координаты
        vertical_lines = self._group_coordinates(vertical_coords, min_distance=20)

        # Находим координаты горизонтальных линий
        horizontal_coords = []
        h_projection = np.sum(horizontal_mask, axis=1)
        h_threshold = np.max(h_projection) * 0.3
        for y in range(len(h_projection)):
            if h_projection[y] > h_threshold:
                horizontal_coords.append(y)

        # Группируем близкие линии
        horizontal_lines = self._group_coordinates(horizontal_coords, min_distance=20)

        logger.info(f"Найдено вертикальных линий: {len(vertical_lines)}")
        logger.info(f"Найдено горизонтальных линий: {len(horizontal_lines)}")

        return vertical_lines, horizontal_lines

    def _group_coordinates(self, coords: List[int], min_distance: int = 20) -> List[int]:
        """Группирует близкие координаты и возвращает средние значения"""
        if not coords:
            return []

        coords = sorted(set(coords))  # убираем дубли и сортируем
        groups = []
        current_group = [coords[0]]

        for i in range(1, len(coords)):
            if coords[i] - coords[i-1] <= min_distance:
                current_group.append(coords[i])
            else:
                groups.append(current_group)
                current_group = [coords[i]]
        groups.append(current_group)

        # Возвращаем средние координаты групп
        return [int(np.mean(group)) for group in groups]

    def segment_by_lines(self, vertical_lines: List[int], horizontal_lines: List[int]) -> List[np.ndarray]:
        """
        Сегментация изображения на ячейки по координатам линий
        """
        logger.info("Выполняем сегментацию по линиям")

        if not self.original_image.any():
            return []

        # Добавляем границы изображения как линии
        height, width = self.original_image.shape[:2]
        v_coords = sorted([0] + vertical_lines + [width])
        h_coords = sorted([0] + horizontal_lines + [height])

        segments = []
        segment_id = 0

        # Создаем сегменты по пересечениям линий
        for i in range(len(h_coords) - 1):
            for j in range(len(v_coords) - 1):
                y1, y2 = h_coords[i], h_coords[i + 1]
                x1, x2 = v_coords[j], v_coords[j + 1]

                # Проверяем размер сегмента (исключаем слишком маленькие)
                if (y2 - y1) > 30 and (x2 - x1) > 30:
                    segment = self.original_image[y1:y2, x1:x2]
                    if segment.size > 0:
                        segments.append(segment)
                        segment_id += 1

                        # Сохраняем координаты сегмента для отладки
                        self.debug_images[f'segment_{segment_id}'] = {
                            'image': segment,
                            'bbox': (x1, y1, x2, y2)
                        }

        logger.info(f"Создано сегментов: {len(segments)}")
        return segments

    def process_segmentation(self, gray: np.ndarray) -> List[np.ndarray]:
        """Полный процесс сегментации: детекция линий + разделение на сегменты"""
        vertical_lines, horizontal_lines = self.detect_lines(gray)
        return self.segment_by_lines(vertical_lines, horizontal_lines)


class OCREngine:
    """
    Упрощенный OCR движок - только PaddleOCR
    """

    def __init__(self, lang: str = 'ru'):
        self.lang = lang
        self._setup_paddle()

    def _setup_paddle(self):
        """Настройка PaddleOCR с оптимизированными параметрами"""
        try:
            from paddleocr import PaddleOCR

            # Без поворота текста, оптимизированные пороги для детекции
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=False,  # Убираем определение ориентации
                lang=self.lang,
                det_db_thresh=0.2,        # Понижаем порог детекции
                det_db_box_thresh=0.3,    # Порог фильтрации боксов
                det_db_unclip_ratio=2.2,  # Расширение контуров
                max_side_len=4096         # Максимальный размер стороны
            )
            logger.info("PaddleOCR настроен (без поворота текста)")
        except ImportError:
            logger.error("PaddleOCR не установлен. Установите: pip install paddleocr")
            raise

    def recognize_text(self, image_segment: np.ndarray) -> str:
        """Распознавание текста в сегменте"""
        try:
            # Конвертируем в RGB для PaddleOCR
            if len(image_segment.shape) == 2:
                rgb_image = cv2.cvtColor(image_segment, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)

            # OCR распознавание
            result = self.paddle_ocr.ocr(rgb_image)

            # Извлекаем текст из результата
            text_lines = []
            if result and result[0]:
                for detection in result[0]:
                    if len(detection) >= 2 and detection[1]:
                        text = detection[1][0] if isinstance(detection[1][0], str) else ""
                        if text.strip():
                            text_lines.append(text)

            return "\n".join(text_lines)

        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            return ""


class YandexGPTProcessor:
    """
    Постобработка текста через YandexGPT с сохранением дореволюционной орфографии
    """

    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None):
        self.api_key = api_key or os.getenv('YANDEX_API_KEY')
        self.folder_id = folder_id or os.getenv('YANDEX_FOLDER_ID')

        if not self.api_key or not self.folder_id:
            logger.warning("YandexGPT API ключи не настроены. Используется заглушка.")
            self.use_mock = True
        else:
            self.use_mock = False

        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }

        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        return """Ты специалист по российской исторической лингвистике. 
Твоя задача исправить ошибки распознавания в тексте, сохранив 
при этом дореволюционную орфографию (орфографию до 1918 года).

Правила дореволюционной орфографии, которые нужно СОХРАНИТЬ:

Буква "ѣ" (ять): бѣлый, смѣхъ, дѣло, мѣсто, цѣлый
Буква "i" (и десятеричное): мiръ, Россiя, историческiй, армiя
Буква "ѳ" (фита): Ѳома, ариѳметика, орѳографiя
Буква "ѵ" (ижица): реже, в церковных словах
Твёрдый знак (ъ) на конце: домъ, столъ, хлѣбъ, мiръ
Окончания -аго/-яго: добраго, синяго, третьяго

ИНСТРУКЦИЯ:
- Исправляй только ошибки OCR
- НЕ модернизируй орфографию
- Сохраняй исторические формы и окончания

РЕЗУЛЬТАТ (только исправленный текст без комментариев):"""

    def _mock_processing(self, text: str) -> str:
        """Простая заглушка для демонстрации"""
        replacements = {
            'дело': 'дѣло',
            'место': 'мѣсто',
            'мир': 'мiръ',
            'Фома': 'Ѳома',
            'хлеб': 'хлѣбъ',
        }
        out = text
        for a, b in replacements.items():
            out = out.replace(a, b).replace(a.capitalize(), b)
        return out

    def _yandex_gpt_processing(self, text: str) -> str:
        """Обработка через YandexGPT API"""
        prompt = f"{self.system_prompt}\n\nТЕКСТ ДЛЯ ОБРАБОТКИ:\n{text}\n\nРЕЗУЛЬТАТ:"
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 4000},
            "messages": [{"role": "user", "text": prompt}]
        }
        try:
            resp = requests.post(self.url, headers=self.headers, json=data, timeout=30)
            js = resp.json()
            if 'result' in js and 'alternatives' in js['result']:
                return js['result']['alternatives'][0]['message']['text'].strip()
            logger.error(f"Неожиданный ответ YandexGPT: {js}")
            return text
        except Exception as e:
            logger.error(f"Ошибка запроса к YandexGPT: {e}")
            return text

    def process_text(self, text_to_correct: str) -> str:
        """Обработка текста с исправлением ошибок OCR"""
        if not text_to_correct.strip():
            return ""

        if self.use_mock:
            return self._mock_processing(text_to_correct)
        return self._yandex_gpt_processing(text_to_correct)


class ArchiveDocumentProcessor:
    """
    Упрощенный процессор архивных документов
    """

    def __init__(self,
                 use_postprocessing: bool = True,
                 yandex_api_key: Optional[str] = None,
                 yandex_folder_id: Optional[str] = None):

        logger.info("Инициализация упрощенной системы обработки архивных документов")

        self.segmentator = DocumentSegmentator()
        self.ocr_engine = OCREngine()

        self.use_postprocessing = use_postprocessing
        self.text_processor: Optional[YandexGPTProcessor] = None
        if use_postprocessing:
            self.text_processor = YandexGPTProcessor(
                api_key=yandex_api_key,
                folder_id=yandex_folder_id
            )

        logger.info(f"Система готова. OCR: PaddleOCR, Постобработка: {use_postprocessing}")

    def _estimate_text_quality(self, text: str) -> float:
        """Оценка качества распознанного текста"""
        if not text:
            return 0.0

        cyr = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        old = set('ѣѳiѵ')
        letters = [c for c in text.lower() if c.isalpha()]

        if not letters:
            return 0.0

        good = sum(1 for c in letters if c in cyr or c in old)
        return min(1.0, good / len(letters))

    def _save_segment_image(self, segment: np.ndarray, output_dir: str, segment_id: int):
        """Сохранение изображения сегмента"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"segment_{segment_id:02d}.png")
        cv2.imwrite(path, segment)

    def _save_debug_images(self, output_dir: str):
        """Сохранение отладочных изображений"""
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        for name, image in self.segmentator.debug_images.items():
            if isinstance(image, dict):  # Сегменты с bbox
                cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), image['image'])
            else:
                cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), image)

        logger.info(f"Отладочные изображения сохранены в: {debug_dir}")

    def process_document(self, image_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Полный процесс обработки документа:
        1. Загрузка изображения
        2. Детекция линий разметки
        3. Сегментация по линиям на ячейки
        4. OCR каждой ячейки
        5. Постобработка через YandexGPT (опционально)
        """
        logger.info(f"Начинаем обработку документа: {image_path}")

        # Загрузка изображения
        gray = self.segmentator.load_image(image_path)

        # Сегментация по линиям
        segments = self.segmentator.process_segmentation(gray)

        if not segments:
            logger.warning("Сегменты не найдены. Обрабатываем как единое изображение.")
            segments = [self.segmentator.original_image]

        results = []

        # Обработка каждого сегмента
        for i, segment in enumerate(segments):
            logger.info(f"Обрабатываем сегмент {i + 1}/{len(segments)}")

            # OCR распознавание
            raw_text = self.ocr_engine.recognize_text(segment)
            logger.info(f"OCR результат: {raw_text[:100]}...")

            # Постобработка
            if self.use_postprocessing and raw_text.strip():
                cleaned_text = self.text_processor.process_text(raw_text) if self.text_processor else raw_text
            else:
                cleaned_text = raw_text

            # Формируем результат
            item = {
                'segment_id': i + 1,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'segment_shape': segment.shape,
                'confidence_estimate': self._estimate_text_quality(cleaned_text)
            }
            results.append(item)

            # Сохраняем сегмент
            if output_dir:
                self._save_segment_image(segment, output_dir, i + 1)

        # Сохраняем отладочные изображения
        if output_dir:
            self._save_debug_images(output_dir)

        logger.info(f"Обработка завершена. Обработано сегментов: {len(results)}")
        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Сохранение результатов распознавания в текстовый файл"""
        logger.info(f"Сохраняем результаты в: {output_path}")

        dir_name = os.path.dirname(output_path)
        if not dir_name:
            dir_name = os.getcwd()
        os.makedirs(dir_name, exist_ok=True)

        full_path = output_path if os.path.isabs(output_path) else os.path.join(dir_name, os.path.basename(output_path))

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ АРХИВНОГО ДОКУМЕНТА\n")
            f.write("=" * 60 + "\n\n")

            for result in results:
                f.write(f"СЕГМЕНТ {result['segment_id']}\n")
                f.write(f"Размер: {result['segment_shape']}\n")
                f.write(f"Качество: {result['confidence_estimate']:.2f}\n")
                f.write("-" * 40 + "\n")
                f.write("Исходный OCR:\n")
                f.write(f"{result['raw_text']}\n\n")
                f.write("После постобработки:\n")
                f.write(f"{result['cleaned_text']}\n")
                f.write("=" * 60 + "\n\n")

        logger.info(f"Результаты сохранены: {full_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Упрощенная система обработки архивных документов (только PaddleOCR + сегментация по линиям)'
    )
    parser.add_argument('input_image', help='Путь к изображению документа')
    parser.add_argument('--output-dir', default='output', help='Директория для выходных файлов')
    parser.add_argument('--no-postprocessing', action='store_true',
                        help='Отключить постобработку через YandexGPT')
    parser.add_argument('--yandex-api-key', help='API ключ YandexGPT')
    parser.add_argument('--yandex-folder-id', help='Folder ID YandexGPT')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.input_image):
        logger.error(f"Файл не найден: {args.input_image}")
        return 1

    try:
        processor = ArchiveDocumentProcessor(
            use_postprocessing=not args.no_postprocessing,
            yandex_api_key=args.yandex_api_key,
            yandex_folder_id=args.yandex_folder_id
        )

        results = processor.process_document(args.input_image, args.output_dir)

        # Сохранение результатов
        results_file = os.path.join(args.output_dir, 'results.txt')
        processor.save_results(results, results_file)

        print("\nОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"Результаты сохранены в: {results_file}")
        print(f"Сегменты сохранены в: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
