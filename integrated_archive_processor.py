#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированная система обработки архивных документов
Объединяет сегментацию, OCR и постобработку в единый пайплайн

Компоненты:
1. Сегментация изображений (из Prep_for_recognition.ipynb)
2. OCR распознавание (Tesseract, PaddleOCR, EasyOCR)
3. Постобработка через YandexGPT (из Lidery_tsifrovoi_transformatsii)

Дата: 2025
"""

import os
import sys
import cv2
import json
import math
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
    Класс для предобработки и сегментации архивных документов
    """

    def __init__(self):
        self.binary_image = None
        self.original_image = None
        self.debug_images = {}
        self.last_rotation = 0

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

    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения: контраст и бинаризация (для Tesseract)
        Для PaddleOCR будет использоваться исходный RGB без этой бинаризации.
        """
        logger.info("Выполняем предобработку изображения")

        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Адаптивная бинаризация (инверт: текст белый)
        bin_inv = cv2.adaptiveThreshold(
            gray_eq, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41, 12
        )

        # Легкая дилатация по вертикали
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        bin_inv = cv2.dilate(bin_inv, kernel, iterations=1)

        # Нормализуем в {0,255}
        bin_inv = (bin_inv > 0).astype(np.uint8) * 255

        self.binary_image = bin_inv
        self.debug_images['binary_inverted'] = bin_inv.copy()
        return bin_inv

    def _strip_margins(self, binary_img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Обрезает пустые поля сверху и снизу"""
        hproj = cv2.reduce(binary_img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        hproj_smooth = cv2.GaussianBlur(hproj, (0, 0), 5)

        if len(hproj_smooth) == 0 or np.max(hproj_smooth) == 0:
            return binary_img, 0, binary_img.shape[0]

        threshold = 0.05 * np.max(hproj_smooth)

        # Поиск верхней границы
        top = 0
        for i, val in enumerate(hproj_smooth):
            if val > threshold:
                top = i
                break

        # Поиск нижней границы
        bottom = len(hproj_smooth)
        for i in range(len(hproj_smooth) - 1, -1, -1):
            if hproj_smooth[i] > threshold:
                bottom = i + 1
                break

        if bottom <= top:
            return binary_img, 0, binary_img.shape[0]

        return binary_img[top:bottom, :], top, bottom

    def segment_columns(self, binary_img: np.ndarray) -> List[np.ndarray]:
        """
        Сегментация изображения на колонки текста (для Tesseract-пайплайна)
        """
        logger.info("Выполняем сегментацию на колонки")

        # Убираем поля
        cropped, top, bottom = self._strip_margins(binary_img)
        if cropped.size == 0:
            return [binary_img]

        # Вертикальная проекция
        vproj = cv2.reduce(cropped, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        vproj_smooth = cv2.GaussianBlur(vproj.astype(np.float32), (0, 0), 3)

        if np.max(vproj_smooth) > 0:
            vproj_norm = vproj_smooth / np.max(vproj_smooth)
        else:
            return [cropped]

        # Поиск промежутков между колонками
        threshold = 0.1
        gaps = vproj_norm < threshold

        segments = []
        start = 0
        in_gap = False
        min_segment_width = 50

        for i, is_gap in enumerate(gaps):
            if is_gap and not in_gap:
                if i - start > min_segment_width:
                    segment = cropped[:, start:i]
                    if segment.shape[1] > 0:
                        segments.append(segment)
                in_gap = True
            elif not is_gap and in_gap:
                start = i
                in_gap = False

        # Последний сегмент
        if not in_gap and len(vproj_norm) - start > min_segment_width:
            segment = cropped[:, start:]
            if segment.shape[1] > 0:
                segments.append(segment)

        if not segments:
            segments = [cropped]

        logger.info(f"Найдено сегментов: {len(segments)}")
        return segments


class OCREngine:
    """
    Класс для распознавания текста различными OCR движками
    """

    def __init__(self, engine: str = 'tesseract', lang: str = 'rus'):
        self.engine = engine.lower()
        self.lang = lang
        self._setup_engine()

    def _setup_engine(self):
        """Инициализация выбранного OCR движка"""
        logger.info(f"Настраиваем OCR движок: {self.engine}")

        if self.engine == 'tesseract':
            self._setup_tesseract()
        elif self.engine == 'paddle':
            self._setup_paddle()
        elif self.engine == 'easy':
            self._setup_easy()
        else:
            raise ValueError(f"Неподдерживаемый OCR движок: {self.engine}")

    # -------------------- Tesseract --------------------

    def _setup_tesseract(self):
        """Настройка Tesseract OCR для дореволюционной орфографии"""
        try:
            import pytesseract
            self.tesseract = pytesseract

            old_cyrillic = 'ѣѳiѵѢѲIѴ'
            modern_cyrillic = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
            digits_punct = '0123456789.,;:!?()[]\\"\' '

            whitelist = modern_cyrillic + old_cyrillic + digits_punct
            self.tesseract_config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}'

            logger.info("Tesseract настроен для дореволюционной орфографии")
        except ImportError:
            logger.error("Tesseract не установлен. Установите: pip install pytesseract")
            raise

    def _preprocess_for_tesseract(self, image: np.ndarray) -> np.ndarray:
        """Дополнительная предобработка для Tesseract"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.medianBlur(enhanced, 3)

        # Масштабирование
        scale_factor = 2.0
        height, width = denoised.shape
        scaled = cv2.resize(
            denoised,
            (int(width * scale_factor), int(height * scale_factor)),
            interpolation=cv2.INTER_CUBIC
        )

        # Инверсия: текст должен быть черным на белом
        if np.mean(scaled) < 128:
            scaled = cv2.bitwise_not(scaled)

        return scaled

    def _tesseract_recognize(self, image_segment: np.ndarray) -> str:
        processed = self._preprocess_for_tesseract(image_segment)
        pil_image = Image.fromarray(processed)
        text = self.tesseract.image_to_string(
            pil_image,
            lang=self.lang,
            config=self.tesseract_config
        )
        return text.strip()

    # -------------------- PaddleOCR 3.x --------------------

    def _setup_paddle(self):
        """Настройка PaddleOCR 3.x (без устаревших аргументов)"""
        try:
            from paddleocr import PaddleOCR
            # Никаких show_log / use_gpu — API 3.x
            # Выбор CPU/GPU делает сам PaddlePaddle по установленному бэкенду
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,  # общая ориентация текстовых строк/страницы
                lang='ru'
            )
            logger.info("PaddleOCR настроен (3.x)")
        except ImportError:
            logger.error("PaddleOCR не установлен. Установите: pip install paddleocr")
            raise

    def _order_paddle_result_by_columns(self, result, n_cols: Optional[int] = None) -> str:
        """
        Робастная сборка результата PaddleOCR 3.x:
        - Поддерживает bbox как 4 пары точек, как 8 чисел, и как dict с 'points'/'bbox'
        - Безопасно читает (text, score), dict{'text','score'} и строку
        - Группирует по колонкам и сортирует строки сверху-вниз
        """

        def _extract_box(det0):
            if det0 is None:
                return None
            # 4 пары точек
            if isinstance(det0, (list, tuple)) and len(det0) == 4 and all(
                    isinstance(p, (list, tuple)) and len(p) == 2 for p in det0):
                return [(float(p[0]), float(p[1])) for p in det0]
            # 8 чисел
            if isinstance(det0, (list, tuple)) and len(det0) == 8 and all(isinstance(v, (int, float)) for v in det0):
                xs = list(map(float, det0))
                return [(xs[0], xs[1]), (xs[2], xs[3]), (xs[4], xs[5]), (xs[6], xs[7])]
            # dict с points/bbox
            if isinstance(det0, dict):
                if 'points' in det0 and isinstance(det0['points'], (list, tuple)) and len(det0['points']) >= 4:
                    pts = det0['points']
                    return [(float(pts[0][0]), float(pts[0][1])),
                            (float(pts[1][0]), float(pts[1][1])),
                            (float(pts[2][0]), float(pts[2][1])),
                            (float(pts[3][0]), float(pts[3][1]))]
                if 'bbox' in det0 and isinstance(det0['bbox'], (list, tuple)) and len(det0['bbox']) == 4:
                    x, y, w, h = map(float, det0['bbox'])
                    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            return None

        def _extract_text_score(det1):
            # (text, score)
            if det1 is None:
                return "", 0.0
            if isinstance(det1, (list, tuple)) and len(det1) >= 1:
                text = det1[0] if isinstance(det1[0], str) else ""
                score = float(det1[1]) if len(det1) > 1 and isinstance(det1[1], (int, float)) else 0.0
                return text, score
            # dict
            if isinstance(det1, dict):
                text = det1.get('text', "")
                score = float(det1.get('score', 0.0))
                return text if isinstance(text, str) else "", score
            # строка
            if isinstance(det1, str):
                return det1, 0.0
            return "", 0.0

        # Уплощаем возможную вложенность [[...]]
        rows = result
        if not rows:
            return ""
        if isinstance(rows, (list, tuple)) and len(rows) == 1 and isinstance(rows[0], (list, tuple)):
            rows = rows[0]

        items = []
        for det in rows:
            det0 = det[0] if isinstance(det, (list, tuple)) and len(det) >= 1 else (
                det.get('box') if isinstance(det, dict) else None)
            # ВАЖНО: не берём det[0] у dict, чтобы не ловить KeyError(0)
            det1 = det[1] if isinstance(det, (list, tuple)) and len(det) >= 2 else (
                det if isinstance(det, dict) else None)

            box = _extract_box(det0)
            text, score = _extract_text_score(det1)
            if not box or not text:
                continue

            x1, y1 = box[0];
            x2, _ = box[1];
            x3, y3 = box[2]
            xc = (x1 + x2 + x3) / 3.0
            yc = (y1 + y3) / 2.0
            items.append((xc, yc, text, score))

        if not items:
            return ""

        xs = sorted([it[0] for it in items])
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        if n_cols is None:
            if diffs:
                med = float(np.median(diffs))
                thr = max(40.0, 3.0 * med)
                n_cols = min(max(len([d for d in diffs if d > thr]) + 1, 2), 12)
            else:
                n_cols = 7

        # Если все центры почти совпадают по X, ставим хотя бы 2 колонки
        if n_cols < 2:
            n_cols = 2

        edges = np.linspace(min(xs), max(xs) + 1e-6, n_cols + 1)
        cols: List[List[Tuple[float, str]]] = [[] for _ in range(n_cols)]
        for xc, yc, text, _ in items:
            # Защита от левого попадания в бин
            idx = int(np.searchsorted(edges, xc) - 1)
            if idx < 0:
                idx = 0
            if idx >= n_cols:
                idx = n_cols - 1
            cols[idx].append((yc, text))

        ordered: List[str] = []
        for col in cols:
            col.sort(key=lambda t: t[0])
            ordered.extend([t for _, t in col])

        return "\n".join(ordered).strip()

    def _paddle_recognize(self, image_segment: np.ndarray) -> str:
        """
        PaddleOCR 3.x: подаём RGB и используем устойчивую сборку колонок.
        """
        if len(image_segment.shape) == 2:
            rgb_image = cv2.cvtColor(image_segment, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)

        result = self.paddle_ocr.ocr(rgb_image)  # API 3.x без cls
        try:
            # При желании можно жестко задать n_cols=7 для метрической книги:
            # text = self._order_paddle_result_by_columns(result, n_cols=7)
            text = self._order_paddle_result_by_columns(result, n_cols=None)
        except Exception as e:
            logger.error(f"Сборка колонок не удалась: {e}")
            # Фолбэк: плоская склейка строк
            lines = []
            rows = result
            if isinstance(rows, (list, tuple)) and len(rows) == 1 and isinstance(rows[0], (list, tuple)):
                rows = rows[0]
            for det in rows or []:
                try:
                    # Ожидаем det[1] = (text, score) или dict
                    if isinstance(det, (list, tuple)) and len(det) >= 2 and isinstance(det[1], (list, tuple)) and det[
                        1]:
                        if isinstance(det[1][0], str):
                            lines.append(det[1][0])
                    elif isinstance(det, dict) and isinstance(det.get('text'), str):
                        lines.append(det['text'])
                except Exception:
                    continue
            text = "\n".join(lines).strip()
        return text

    # -------------------- EasyOCR --------------------

    def _setup_easy(self):
        """Настройка EasyOCR"""
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(['ru'], gpu=False, verbose=False)
            logger.info("EasyOCR настроен")
        except ImportError:
            logger.error("EasyOCR не установлен. Установите: pip install easyocr")
            raise

    def _easy_recognize(self, image_segment: np.ndarray) -> str:
        if len(image_segment.shape) == 2:
            image_array = cv2.cvtColor(image_segment, cv2.COLOR_GRAY2BGR)
        else:
            image_array = image_segment

        results = self.easy_reader.readtext(image_array)
        parts = []
        for item in results:
            if len(item) >= 3:
                _, text, conf = item[0], item[1], item[2]
                if conf > 0.5 and text:
                    parts.append(text)
        return "\n".join(parts).strip()

    # -------------------- Public --------------------

    def recognize_text(self, image_segment: np.ndarray) -> str:
        """Распознавание текста в сегменте изображения"""
        try:
            if self.engine == 'tesseract':
                return self._tesseract_recognize(image_segment)
            elif self.engine == 'paddle':
                return self._paddle_recognize(image_segment)
            elif self.engine == 'easy':
                return self._easy_recognize(image_segment)
        except Exception as e:
            logger.error(f"Ошибка распознавания ({self.engine}): {e}")
            return ""


class YandexGPTProcessor:
    """
    Класс для постобработки распознанного текста через YandexGPT
    """

    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None):
        self.api_key = api_key or os.getenv('YANDEX_API_KEY') or "TEST_API_KEY_PLACEHOLDER"
        self.folder_id = folder_id or os.getenv('YANDEX_FOLDER_ID') or "TEST_FOLDER_ID_PLACEHOLDER"

        if (self.api_key == "TEST_API_KEY_PLACEHOLDER") or (self.folder_id == "TEST_FOLDER_ID_PLACEHOLDER"):
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
        # Лёгкая псевдокоррекция для демонстрации
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
        if not text_to_correct.strip():
            return ""
        if self.use_mock:
            return self._mock_processing(text_to_correct)
        return self._yandex_gpt_processing(text_to_correct)


class ArchiveDocumentProcessor:
    """
    Главный класс для обработки архивных документов
    Интегрирует все компоненты системы
    """

    def __init__(self,
                 ocr_engine: str = 'tesseract',
                 use_postprocessing: bool = True,
                 yandex_api_key: Optional[str] = None,
                 yandex_folder_id: Optional[str] = None):

        logger.info("Инициализация системы обработки архивных документов")

        self.segmentator = DocumentSegmentator()
        self.ocr_engine = OCREngine(engine=ocr_engine)

        self.use_postprocessing = use_postprocessing
        self.text_processor: Optional[YandexGPTProcessor] = None
        if use_postprocessing:
            self.text_processor = YandexGPTProcessor(
                api_key=yandex_api_key,
                folder_id=yandex_folder_id
            )

        logger.info(f"Система готова. OCR: {ocr_engine}, Постобработка: {use_postprocessing}")

    def _estimate_text_quality(self, text: str) -> float:
        """Простая оценка доли кириллицы (включая дореволюционные символы)"""
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
        os.makedirs(output_dir, exist_ok=True)
        # Для визуализации делаем текст чёрным на белом
        img = segment
        if len(img.shape) == 2:
            display = cv2.bitwise_not(img)
        else:
            # в цвете инверсию не делаем
            display = img.copy()
        path = os.path.join(output_dir, f"segment_{segment_id:02d}.png")
        cv2.imwrite(path, display)

    def process_document(self, image_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Полный цикл обработки документа:
        - Загрузка и предобработка
        - Для Paddle: распознаём полную RGB-страницу без бинаризации/сегментации
        - Для Tesseract/Easy: сегментируем колонки по бинарной маске
        - Постобработка через YandexGPT (опционально)
        """
        logger.info(f"Начинаем обработку документа: {image_path}")

        # 1) Загрузка и предобработка
        gray = self.segmentator.load_image(image_path)
        binary = self.segmentator.preprocess(gray)

        results: List[Dict] = []

        # 2) Ветвление пайплайна
        if self.ocr_engine.engine == 'paddle':
            # Подаём полную страницу в RGB без сегментации
            segments = [self.segmentator.original_image]
        else:
            # Сегментация колонок по бинарной маске
            segments = self.segmentator.segment_columns(binary)

        # 3) OCR + постобработка
        for i, segment in enumerate(segments):
            logger.info(f"Обрабатываем сегмент {i + 1}/{len(segments)}")

            raw_text = self.ocr_engine.recognize_text(segment)
            logger.info(f"OCR результат: {raw_text[:100]}...")

            if self.use_postprocessing and raw_text.strip():
                cleaned_text = self.text_processor.process_text(raw_text) if self.text_processor else raw_text
            else:
                cleaned_text = raw_text

            item = {
                'segment_id': i + 1,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'segment_shape': segment.shape,
                'confidence_estimate': self._estimate_text_quality(cleaned_text)
            }
            results.append(item)

            if output_dir:
                self._save_segment_image(segment, output_dir, i + 1)

        logger.info(f"Обработка завершена. Обработано сегментов: {len(results)}")
        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Сохраняет результаты распознавания в текстовый файл (безопасно обрабатывает пустой каталог)"""
        logger.info(f"Сохраняем результаты в: {output_path}")

        dir_name = os.path.dirname(output_path)
        if not dir_name:
            dir_name = os.getcwd()  # текущая директория по умолчанию
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
        description='Система обработки архивных документов с дореволюционной орфографией'
    )
    parser.add_argument('input_image', help='Путь к изображению документа')
    parser.add_argument('--output-dir', default='output', help='Директория для выходных файлов')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'paddle', 'easy'],
                        default='tesseract', help='OCR движок')
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

    # Временный локальный fallback (можно удалить перед продом)
    os.environ.setdefault("YANDEX_API_KEY",
                          args.yandex_api_key or os.getenv("YANDEX_API_KEY", "TEST_API_KEY_PLACEHOLDER"))
    os.environ.setdefault("YANDEX_FOLDER_ID",
                          args.yandex_folder_id or os.getenv("YANDEX_FOLDER_ID", "TEST_FOLDER_ID_PLACEHOLDER"))

    try:
        processor = ArchiveDocumentProcessor(
            ocr_engine=args.ocr_engine,
            use_postprocessing=not args.no_postprocessing,
            yandex_api_key=args.yandex_api_key,
            yandex_folder_id=args.yandex_folder_id
        )

        results = processor.process_document(args.input_image, args.output_dir)

        # Если пользователь передал только имя файла — сохранится в текущую директорию
        results_file = os.path.join(args.output_dir, 'results.txt') if args.output_dir else 'results.txt'
        processor.save_results(results, results_file)

        print("\nОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"Результаты сохранены в: {results_file}")
        if args.output_dir:
            print(f"Сегменты сохранены в: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
