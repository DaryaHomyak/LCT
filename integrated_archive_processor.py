import re
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
from datetime import datetime

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
            (r'([а-яё])1', r'\1і'),  # цифра 1 → і
            (r'о([рн])', r'он\1'),  # пропущенная н
            (r'([аеиоуыэюя])11([а-яё])', r'\1ии\2'),  # 11 → ии
            (r'рг', 'гг'),  # рг → гг
            (r'оо', 'со'),  # оо → со

            # Исправление латинских букв → кириллические
            (r'[Aa]', 'а'), (r'[Bb]', 'в'), (r'[Cc]', 'с'),
            (r'[Ee]', 'е'), (r'[Hh]', 'н'), (r'[Kk]', 'к'),
            (r'[Mm]', 'м'), (r'[Oo]', 'о'), (r'[Pp]', 'р'),
            (r'[Tt]', 'т'), (r'[Xx]', 'х'), (r'[Yy]', 'у'),

            # Цифры → буквы
            (r'1', 'і'), (r'0', 'о'), (r'6', 'б')
        ]

        # Загружаем исторический словарь
        self.prerev_dictionary = self.load_historical_dictionary()

    def load_historical_dictionary(self):
        """Загрузка словаря дореволюционных слов"""
        # Базовый словарь дореволюционных слов
        historical_words = {
            # Местоимения и частицы
            'его', 'ея', 'оно', 'они', 'онѣ', 'имъ', 'ими', 'ихъ',
            'что', 'чтобы', 'какъ', 'гдѣ', 'когда', 'почему',

            # Глаголы
            'было', 'были', 'будетъ', 'имѣетъ', 'дѣлаетъ', 'говоритъ',
            'идетъ', 'беретъ', 'даетъ', 'знаeetъ', 'можетъ', 'хочетъ',

            # Существительные
            'домъ', 'городъ', 'человѣкъ', 'людей', 'дѣло', 'время',
            'мѣсто', 'жизнь', 'миръ', 'странѣ', 'государь', 'царь',

            # Прилагательные
            'великій', 'новый', 'старый', 'хорошій', 'плохой', 'большой',
            'маленькій', 'первый', 'послѣдній', 'русскій', 'важный',

            # Предлоги и союзы
            'для', 'безъ', 'подъ', 'надъ', 'передъ', 'послѣ', 'между',
            'черезъ', 'противъ', 'вмѣстѣ', 'также', 'однако', 'потому'
        }

        return set(historical_words)

    def clean_text(self, text: str) -> str:
        """Очистка и модернизация дореволюционного текста"""
        if not text:
            return text

        result = text

        # 1. Исправление OCR ошибок
        for pattern, replacement in self.error_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # 2. Замена старых букв на новые
        for old, new in self.old_to_new.items():
            result = result.replace(old, new)

        # 3. Финальная очистка
        # Удаляем лишние символы, оставляем буквы, цифры и основные знаки
        result = re.sub(r'[^\wа-яё\s\-.,;:!?()«»""\']', ' ', result, flags=re.IGNORECASE)

        # Убираем множественные пробелы
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()

        return result


class LinearSegmentator:

    def __init__(self, debug_mode=True):
        self.min_line_height = 10
        self.max_line_gap = 50
        self.debug_mode = debug_mode

        if self.debug_mode:
            self.debug_dir = Path("debug_segments")
            self.debug_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_debug_dir = self.debug_dir / timestamp
            self.current_debug_dir.mkdir(exist_ok=True)

            logger.info(f"Отладка: {self.current_debug_dir}")

    def _trim_outliers(self, proj):
        # Обрезаем крайние сильные пики (черные края) выше 95-го процентиля
        p95 = np.percentile(proj, 95)
        return np.clip(proj, 0, p95)

    def _adaptive_threshold(self, proj, width):
        # Используем медиану (50-й перцентиль) как порог
        median = np.median(proj)
        # Добавляем небольшой запас (10% от IQR) для надёжности
        q1, q3 = np.percentile(proj, [25, 75])
        iqr = q3 - q1
        thresh = median + 0.1 * iqr

        # Гарантируем минимум 2% от ширины
        min_thr = width * 0.02
        return max(thresh, min_thr)

    def _detect_separators(self, proj, low_thr):
        # Ищем «тонкие» пики (разделительные линии)
        separators = []
        in_sep = False
        start = 0
        for i, v in enumerate(proj):
            if v > low_thr and not in_sep:
                in_sep = True
                start = i
            elif v <= low_thr and in_sep:
                end = i
                if end - start >= self.min_line_height:
                    separators.append((start, end))
                in_sep = False
        return separators

    def process_segmentation(self, image, hp_percentile: float = 75.0, **kwargs):
        try:
            if self.debug_mode:
                debug_original = self.current_debug_dir / "00_original.jpg"
                cv2.imwrite(str(debug_original), image)
                logger.info(f"Сохранено: {debug_original}")

            height, width = image.shape[:2]
            proj = np.sum(image < 128, axis=1)

            # клиппинг по перцентилю для горизонтали
            proj_clip = np.clip(proj, 0, np.percentile(proj, hp_percentile))

            # Порог = медиана клиппинга
            thr = float(np.median(proj_clip))

            # Основные строки
            lines = []
            in_line = False
            for i, v in enumerate(proj_clip):
                if v > thr and not in_line:
                    start = i
                    in_line = True
                elif v <= thr and in_line:
                    end = i
                    in_line = False
                    if end - start >= self.min_line_height:
                        lines.append((start, end))
            if in_line:
                lines.append((start, height))

            # Тонкие разделители
            low_thr = thr * 0.3
            seps = self._detect_separators(proj_clip, low_thr)

            segments = []
            for idx, (start, end) in enumerate(lines, 1):
                # Добавляем отступы, но не выходим за границы изображения
                padded_start = max(0, start - 2)  # 2 пикселя сверху
                padded_end = min(height, end + 2)  # 2 пикселя снизу

                segment = image[padded_start:padded_end, :]  # ← ТЕПЕРЬ С ОТСТУПАМИ
                segments.append({
                    'id': idx,
                    'image': segment,
                    'bbox': (0, padded_start, width, padded_end),
                    'shape': segment.shape
                })

            if self.debug_mode:
                self._save_debug(segments, image, lines, proj_clip, thr)

            logger.info(f"Найдено {len(lines)} строк, создано {len(segments)} сегментов (thr={thr:.1f}, hp_pctl={hp_percentile:.1f})")
            return segments

        except Exception as e:
            logger.error(f"Ошибка сегментации: {e}")
            return [{'id': 1, 'image': image, 'bbox': (0, 0, width, height), 'shape': image.shape}]

    def _save_debug(self, segments, image, lines, projection, threshold):
        """Сохраняем отладочную информацию"""
        # График горизонтальной проекции
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(projection)
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Порог = {threshold:.1f}')
            plt.title('Горизонтальная проекция (черные пиксели по строкам)')
            plt.xlabel('Номер строки')
            plt.ylabel('Количество черных пикселей')
            plt.legend()
            plt.grid(True)

            projection_path = self.current_debug_dir / "01_projection.png"
            plt.savefig(projection_path, dpi=100, bbox_inches='tight')
            plt.close()

        except ImportError:
            logger.warning("matplotlib не установлен, график проекции не создан")
        except Exception as e:
            logger.warning(f"Ошибка создания графика: {e}")

        # Изображение с разметкой строк
        debug_markup = image.copy()
        if len(debug_markup.shape) == 2:
            debug_markup = cv2.cvtColor(debug_markup, cv2.COLOR_GRAY2BGR)

        # найденные строки
        for i, (start, end) in enumerate(lines):
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.rectangle(debug_markup, (0, start), (image.shape[1], end), color, 2)
            cv2.putText(debug_markup, f"Line {i + 1}", (5, start + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        markup_path = self.current_debug_dir / "02_lines_markup.jpg"
        cv2.imwrite(str(markup_path), debug_markup)

        # Сохраняем каждый сегмент
        for segment in segments:
            filename = f"segment_{segment['id']:02d}_h{segment['shape'][0]}_w{segment['shape'][1]}.jpg"
            segment_path = self.current_debug_dir / filename
            cv2.imwrite(str(segment_path), segment['image'])

            logger.info(f"Сегмент {segment['id']}: {segment['shape']}")

        # 4. Текстовый отчет
        report_path = self.current_debug_dir / "report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Отчет сегментации\n")
            f.write(f"Время: {datetime.now()}\n")
            f.write(f"Размер изображения: {image.shape}\n")
            f.write(f"Порог: {threshold:.1f} пикселей\n")
            f.write(f"Найдено строк: {len(lines)}\n\n")

            for i, (start, end) in enumerate(lines):
                f.write(f"Строка {i + 1}: Y={start}-{end}, высота={end - start}\n")


class GridSegmentator(LinearSegmentator):

    def __init__(self, debug_mode=True):
        super().__init__(debug_mode=debug_mode)

    def process_segmentation(self, image, hp_percentile: float = 75.0, vp_percentile: float = 75.0):
        height, width = image.shape[:2]

        # 1. Вертикальная проекция (колонки) - ИСПРАВЛЕНО: используем vp_percentile
        vp = np.sum(image < 128, axis=0)
        vp_clip = np.clip(vp, 0, np.percentile(vp, vp_percentile))
        thr_v = float(np.median(vp_clip))

        cols = []
        in_col = False
        col_start = 0
        for x, v in enumerate(vp_clip):
            if v > thr_v and not in_col:
                col_start = x
                in_col = True
            elif v <= thr_v and in_col:
                col_end = x
                in_col = False
                if col_end - col_start >= 50:  # минимальная ширина колонки
                    cols.append((col_start, col_end))
        if in_col:
            cols.append((col_start, width))
        if not cols:
            cols = [(0, width)]

        if self.debug_mode:
            self._save_columns_debug(image, cols, vp_clip, thr_v)

        # 2. Для каждой колонки - горизонтальная сегментация
        all_segments = []
        seg_id = 1
        for col_idx, (start_x, end_x) in enumerate(cols, 1):
            col_img = image[:, start_x:end_x]
            segments = super().process_segmentation(col_img, hp_percentile=hp_percentile)
            for seg in segments:
                # Корректируем bbox в глобальные координаты
                x0, y0, x1, y1 = seg['bbox'][0], seg['bbox'][1], seg['bbox'][2], seg['bbox'][3]
                seg['bbox'] = (start_x + x0, y0, start_x + x1, y1)
                seg['id'] = seg_id
                seg['column'] = col_idx
                all_segments.append(seg)
                seg_id += 1

        return all_segments

    def _save_columns_debug(self, image, cols, projection, thr_v):
        """Сохранение отладки колонок"""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(projection)
            plt.axhline(y=thr_v, color='r', linestyle='--', label=f'Порог = {thr_v:.1f}')
            plt.title('Вертикальная проекция (черные пиксели по столбцам)')
            plt.xlabel('Номер столбца')
            plt.ylabel('Количество черных пикселей')
            plt.legend()
            plt.grid(True)
            path = self.current_debug_dir / "03_vertical_projection.png"
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception:
            logger.warning("Не удалось сохранить график вертикальной проекции")

        # Разметка колонок на изображении
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        for i, (sx, ex) in enumerate(cols, 1):
            color = (255, 255, 0)
            cv2.rectangle(viz, (sx, 0), (ex, image.shape[0]), color, 2)
            cv2.putText(viz, f"Col{i}", (sx + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        path2 = self.current_debug_dir / "04_columns_markup.jpg"
        cv2.imwrite(str(path2), viz)


class ImprovedOCREngine:
    """ OCR движок с поддержкой TrOCR и PaddleOCR"""

    def __init__(self):
        self.paddle_ocr = None
        self.trocr_engine = None

        # Сначала пытаемся инициализировать TrOCR
        self.initialize_trocr()

        # Если TrOCR не удалось, используем PaddleOCR как fallback
        if self.trocr_engine is None:
            self.initialize_paddle()

    def initialize_trocr(self):
        """Инициализация TrOCR с русской моделью"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image

            logger.info("Загружаем TrOCR модель для русского языка...")

            processor = TrOCRProcessor.from_pretrained(
                'kazars24/trocr-base-handwritten-ru'
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                'kazars24/trocr-base-handwritten-ru'
            )

            self.trocr_engine = {
                'processor': processor,
                'model': model
            }

            logger.info("TrOCR успешно инициализирован")

        except ImportError:
            logger.warning("TrOCR библиотеки не установлены. Используйте: pip install transformers torch pillow")
            self.trocr_engine = None
        except Exception as e:
            logger.error(f"Ошибка при инициализации TrOCR: {e}")
            self.trocr_engine = None

    def initialize_paddle(self):
        """Инициализация PaddleOCR как fallback"""
        try:
            from paddleocr import PaddleOCR

            logger.info("Инициализируем PaddleOCR...")

            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ru',
                det_db_thresh=0.2,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.2,
                max_side_len=4096,
                cls_thresh=0.9,
                rec_image_shape=[3, 48, 320],
                drop_score=0.3
            )

            logger.info("PaddleOCR инициализирован")

        except ImportError:
            logger.error("PaddleOCR не установлен. Установите: pip install paddlepaddle paddleocr")
            self.paddle_ocr = None
        except Exception as e:
            logger.error(f"Ошибка при инициализации PaddleOCR: {e}")
            self.paddle_ocr = None

    def recognize_text(self, image_segment):
        """Сначала TrOCR, потом PaddleOCR"""

        # Проверяем размер сегмента
        if image_segment.shape[0] < 20 or image_segment.shape[1] < 50:
            logger.warning(f"Сегмент слишком мал: {image_segment.shape}")
            return {
                'raw_text': "",
                'confidence': 0.0,
                'boxes': []
            }

        # Сначала пробуем TrOCR
        if self.trocr_engine is not None:
            try:
                result = self.recognize_with_trocr(image_segment)

                # Если TrOCR ничего не распознал, пробуем PaddleOCR
                if not result['raw_text'].strip():
                    logger.info("TrOCR не распознал, пробуем PaddleOCR")
                    if self.paddle_ocr is not None:
                        return self.recognize_with_paddle(image_segment)

                return result

            except Exception as e:
                logger.warning(f"TrOCR failed: {e}")

        # Fallback на PaddleOCR
        if self.paddle_ocr is not None:
            return self.recognize_with_paddle(image_segment)

        return {
            'raw_text': "Все OCR движки недоступны",
            'confidence': 0.0,
            'boxes': []
        }

    def recognize_with_trocr(self, image_segment):
        """Распознавание через TrOCR"""
        from PIL import Image
        import torch

        # Конвертируем OpenCV изображение в PIL
        if len(image_segment.shape) == 3:
            # Цветное изображение BGR -> RGB
            pil_image = Image.fromarray(cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale -> RGB
            pil_image = Image.fromarray(image_segment).convert('RGB')

        processor = self.trocr_engine['processor']
        model = self.trocr_engine['model']

        # Процессинг изображения
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

        # Генерация текста
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        # Декодирование результата
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            'raw_text': text,
            'confidence': 0.85,  # TrOCR не возвращает confidence, используем оценочное значение
            'boxes': []  # TrOCR не возвращает bbox координаты
        }

    def recognize_with_paddle(self, image_segment):
        """Распознавание через PaddleOCR (оригинальный метод)"""
        try:
            # Улучшаем изображение перед распознаванием
            result = self.paddle_ocr.ocr(image_segment, cls=True)

            if not result or not result[0]:
                logger.info("PaddleOCR не смог распознать текст, пробуем с улучшенным изображением")
                enhanced_image = self.enhance_image_for_ocr(image_segment)
                result = self.paddle_ocr.ocr(enhanced_image, cls=True)

            if not result or not result[0]:
                logger.info("Второй проход PaddleOCR тоже не дал результата")
                return {
                    'raw_text': "",
                    'confidence': 0.0,
                    'boxes': []
                }

            return self.process_paddle_result(result)

        except Exception as e:
            logger.error(f"Ошибка в PaddleOCR: {e}")
            return {
                'raw_text': f"Ошибка OCR: {str(e)}",
                'confidence': 0.0,
                'boxes': []
            }

    def enhance_image_for_ocr(self, image):
        """Улучшение изображения для лучшего OCR"""
        try:
            # Увеличиваем разрешение
            height, width = image.shape[:2]
            scale_factor = 1.7
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Улучшаем контрастность
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

            # Убираем шум
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

            return enhanced

        except Exception as e:
            logger.warning(f"Ошибка при улучшении изображения: {e}")
            return image

    def process_paddle_result(self, result) -> Dict[str, Any]:
        """Обработка результатов PaddleOCR"""
        if not result or not result[0]:
            return {
                'raw_text': "",
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
                    confidence = 0.5
                else:
                    continue

                if text and text.strip():
                    texts.append(text.strip())
                    confidences.append(float(confidence))
                    boxes.append(bbox)

        combined_text = " ".join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'raw_text': combined_text,
            'confidence': avg_confidence,
            'boxes': boxes
        }


class ArchiveDocumentProcessor:
    """Процессор архивных документов"""

    def __init__(self, use_postprocessing=True, debug_mode=True):
        logger.info("Инициализация исправленной системы обработки архивных документов")

        self.use_postprocessing = use_postprocessing
        self.debug_mode = debug_mode

        self.segmentator = GridSegmentator(debug_mode=debug_mode)
        self.ocr_engine = ImprovedOCREngine()
        self.text_processor = PrehistoricalTextProcessor() if use_postprocessing else None

        # Статистика
        self.processed_documents = 0
        self.total_processing_time = 0.0

    def process_document(self, image_path: str, hp_percentile: float = 75.0, vp_percentile: float = 75.0) -> List[Dict[str, Any]]:
        """Основная функция обработки документа с двумя перцентилями"""
        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

            logger.info(f"Обработка изображения {image_path}, размер: {gray.shape}")

            # Сегментация с двумя перцентилями
            segments = self.segmentator.process_segmentation(
                gray,
                hp_percentile=hp_percentile,
                vp_percentile=vp_percentile
            )
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

                # Оценка уверенности
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


# Тестирование
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
