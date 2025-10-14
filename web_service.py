import os
import uuid
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import pdf2image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import aiofiles

# Импорт OCR системы
try:
    from integrated_archive_processor import ArchiveDocumentProcessor
except ImportError:
    class ArchiveDocumentProcessor:
        def __init__(self, use_postprocessing=True):
            pass

        def process_document(self, file_path, **kwargs):
            return []

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PERCENTILE = 75.0
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
DATABASE_URL = "sqlite:///./enhanced_archive_service.db"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
LOW_CONFIDENCE_THRESHOLD = 0.75  # Порог низкой уверенности

for dir_path in [UPLOAD_DIR, RESULTS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    original_image_path = Column(String)
    status = Column(String, default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    confidence_score = Column(Float, default=0.0)
    low_confidence_count = Column(Integer, default=0)
    total_segments = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    file_size = Column(Integer, default=0)
    image_width = Column(Integer, default=0)
    image_height = Column(Integer, default=0)


class TextSegmentModel(Base):
    __tablename__ = "text_segments"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer)
    segment_id = Column(Integer)
    raw_text = Column(Text)
    corrected_text = Column(Text)
    bbox = Column(JSON)
    polygon = Column(JSON, nullable=True)
    confidence = Column(Float)
    is_verified = Column(Boolean, default=False)
    is_corrected = Column(Boolean, default=False)
    correction_count = Column(Integer, default=0)
    last_corrected_at = Column(DateTime, nullable=True)


class AttributeModel(Base):
    __tablename__ = "attributes"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer)
    segment_id = Column(Integer, nullable=True)
    attr_type = Column(String)
    attr_value = Column(String)
    normalized_value = Column(String, nullable=True)
    confidence = Column(Float)
    bbox = Column(JSON)
    is_verified = Column(Boolean, default=False)
    extraction_method = Column(String, default="regex")


class SessionStatsModel(Base):
    __tablename__ = "session_stats"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    total_segments = Column(Integer, default=0)
    low_confidence_segments = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


class DocumentResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    confidence_score: float
    created_at: datetime
    processed_at: Optional[datetime]
    total_segments: int
    low_confidence_count: int
    processing_time: float


class VerificationRequest(BaseModel):
    segment_id: int
    corrected_text: str
    is_verified: bool = True


class RecognizeRequest(BaseModel):
    task_id: str
    hp_percentile: float = 75.0
    vp_percentile: float = 75.0


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class EnhancedDocumentProcessor:
    """Процессор документов"""

    def __init__(self):
        self.ocr_processor = ArchiveDocumentProcessor(use_postprocessing=True)

    async def process_document_enhanced(
            self, task_id: str, file_path: str, db: Session,
            hp_percentile: float = 75.0, vp_percentile: float = 75.0
    ) -> Dict:
        start_time = datetime.utcnow()
        logger.info(f"Начинаем обработку документа {task_id}")

        doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Документ не найден")

        doc.status = "processing"
        db.commit()

        try:
            # Получаем размеры изображения
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    doc.image_height, doc.image_width = image.shape[:2]
                    doc.file_size = os.path.getsize(file_path)
            except Exception as e:
                logger.warning(f"Не удалось получить размеры изображения: {e}")

            # OCR обработка
            try:
                results = self.ocr_processor.process_document(
                    file_path,
                    hp_percentile=hp_percentile,
                    vp_percentile=vp_percentile
                )
            except Exception as e:
                logger.error(f"Ошибка OCR: {e}")
                results = []
            if not results:
                results = [{
                    'segment_id': 1,
                    'raw_text': 'Не удалось распознать текст',
                    'cleaned_text': 'Не удалось распознать текст',
                    'segment_shape': (100, 100),
                    'confidence_estimate': 0.0
                }]

            segments = []
            total_confidence = 0.0
            low_confidence_count = 0

            for i, result in enumerate(results):
                segment_id = result.get('segment_id', i + 1)
                raw_text = result.get('raw_text', '')
                cleaned_text = result.get('cleaned_text', raw_text)
                shape = result.get('segment_shape', (100, 100))
                confidence = result.get('confidence_estimate', 0.0)

                bbox = self._shape_to_bbox(shape)

                segment_data = {
                    'segment_id': segment_id,
                    'raw_text': raw_text or '',
                    'corrected_text': cleaned_text or '',
                    'bbox': bbox,
                    'confidence': confidence
                }
                segments.append(segment_data)

                total_confidence += confidence
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_count += 1

            # Расчет метрик
            avg_confidence = total_confidence / max(len(results), 1)
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Сохраняем в БД
            self._save_to_database(db, doc.id, segments)

            # Обновляем документ
            doc.status = "completed"
            doc.processed_at = datetime.utcnow()
            doc.confidence_score = avg_confidence
            doc.low_confidence_count = low_confidence_count
            doc.total_segments = len(results)
            doc.processing_time = processing_time
            db.commit()

            # Обновляем статистику сессии
            self._update_session_stats(db, task_id, len(results), avg_confidence, low_confidence_count, processing_time)

            return {
                "task_id": task_id,
                "segments": segments,
                "confidence_score": avg_confidence,
                "low_confidence_count": low_confidence_count,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Ошибка обработки {task_id}: {e}")
            doc.status = "error"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

    def _shape_to_bbox(self, shape) -> List[float]:
        if hasattr(shape, '__len__') and len(shape) >= 2:
            h, w = shape[0], shape[1]
            return [0.0, 0.0, float(w), float(h)]
        return [0.0, 0.0, 100.0, 100.0]

    def _save_to_database(self, db: Session, doc_id: int, segments: List[Dict]):
        for segment in segments:
            db_segment = TextSegmentModel(
                document_id=doc_id,
                segment_id=segment['segment_id'],
                raw_text=segment['raw_text'],
                corrected_text=segment['corrected_text'],
                bbox=segment['bbox'],
                confidence=segment['confidence']
            )
            db.add(db_segment)
        db.commit()

    def _update_session_stats(self, db: Session, task_id: str, segments_count: int, avg_confidence: float,
                              low_confidence_count: int, processing_time: float):
        session_id = "default_session"

        stats = db.query(SessionStatsModel).filter(SessionStatsModel.session_id == session_id).first()
        if not stats:
            # Создаем новую запись с правильной инициализацией
            stats = SessionStatsModel(
                session_id=session_id,
                total_documents=0,
                processed_documents=0,
                avg_confidence=0.0,
                total_segments=0,
                low_confidence_segments=0,
                processing_time=0.0
            )
            db.add(stats)
            db.commit()
            db.refresh(stats)

        # защита от None значений
        stats.total_documents = (stats.total_documents or 0) + 1
        stats.processed_documents = (stats.processed_documents or 0) + 1
        stats.total_segments = (stats.total_segments or 0) + segments_count
        stats.low_confidence_segments = (stats.low_confidence_segments or 0) + low_confidence_count
        stats.processing_time = (stats.processing_time or 0.0) + processing_time

        old_sum = (stats.avg_confidence or 0.0) * max((stats.processed_documents - 1), 1)
        stats.avg_confidence = (old_sum + avg_confidence) / max(stats.processed_documents, 1)
        stats.updated_at = datetime.utcnow()

        db.commit()


async def save_uploaded_file(upload_file: UploadFile, task_id: str) -> str:
    file_extension = Path(upload_file.filename).suffix.lower()
    file_path = UPLOAD_DIR / f"{task_id}{file_extension}"

    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)

    return str(file_path)


def convert_pdf_to_images(pdf_path: str) -> List[str]:
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=300)
        image_paths = []

        base_name = Path(pdf_path).stem
        for i, image in enumerate(images):
            img_path = UPLOAD_DIR / f"{base_name}_page_{i + 1}.jpg"
            image.save(img_path, 'JPEG', quality=95)
            image_paths.append(str(img_path))

        return image_paths
    except Exception as e:
        logger.error(f"Ошибка конвертации PDF: {e}")
        return [pdf_path]


app = FastAPI(
    title="Архивный OCR Сервис",
    description="Веб-сервис для обработки архивных документов",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Глобальный процессор
document_processor = EnhancedDocumentProcessor()


@app.get("/", response_class=HTMLResponse)
async def enhanced_index():
    """Главная страница"""
    html_content = r"""<!DOCTYPE html>
<html lang="ru">
<head>
    <title>🏛️ Архивный OCR Сервис</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px; background: #f5f7fa; color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px;
            text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }

        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        .panel {
            background: white; border-radius: 15px; padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08); border: 1px solid #e1e8ed; margin-bottom: 30px;
        }
        .panel h2 {
            margin: 0 0 20px; color: #2c3e50; font-size: 1.4em;
            border-bottom: 2px solid #3498db; padding-bottom: 10px;
        }

        .upload-area {
            border: 3px dashed #3498db; border-radius: 10px;
            padding: 40px 20px; text-align: center; margin: 20px 0;
            background: linear-gradient(45deg, #f8f9ff, #e8f4fd);
            transition: all 0.3s ease; cursor: pointer;
        }
        .upload-area:hover {
            border-color: #2980b9; background: linear-gradient(45deg, #e8f4fd, #d6eaff);
            transform: translateY(-2px);
        }

        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
        .stat-card {
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white; padding: 20px; border-radius: 10px; text-align: center;
        }
        .stat-number { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 0.9em; opacity: 0.9; }

        .progress-item {
            background: #f8f9fa; border-left: 4px solid #3498db;
            padding: 15px; margin: 10px 0; border-radius: 5px;
        }
        .progress-item.completed { border-left-color: #27ae60; background: #d5f4e6; }
        .progress-item.error { border-left-color: #e74c3c; background: #fadbd8; }

        .document-card {
            background: white; border: 1px solid #ddd; border-radius: 10px;
            margin: 15px 0; overflow: hidden; box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .document-header {
            background: linear-gradient(90deg, #74b9ff, #0984e3);
            color: white; padding: 15px;
        }
        .document-body { padding: 20px; }

        .segment-box {
            background: #f1f3f4; border-radius: 8px; padding: 15px;
            margin: 10px 0; position: relative; border-left: 4px solid #3498db;
        }
        .segment-box.low-confidence { border-left-color: #f39c12; background: #fef9e7; }
        .segment-box.verified { border-left-color: #27ae60; background: #d5f4e6; }

        .btn {
            padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;
            font-weight: bold; text-decoration: none; display: inline-block;
            transition: all 0.3s ease; margin: 5px;
        }
        .btn-primary { background: #3498db; color: white; }
        .btn-primary:hover { background: #2980b9; transform: translateY(-2px); }
        .btn-success { background: #27ae60; color: white; }
        .btn-success:hover { background: #229954; }

        @media (max-width: 768px) {
            .main-content { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr; }
        }

        .loading {
            display: inline-block; width: 20px; height: 20px;
            border: 3px solid #f3f3f3; border-top: 3px solid #3498db;
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .text-block { white-space: pre-wrap; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🏛️ Архивный OCR Сервис</h1>
        <p>Система обработки архивных документов</p>
    </div>

    <div class="main-content">
        <div class="panel">
            <h2>📁 Загрузка документов</h2>
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div style="font-size: 3em; margin-bottom: 10px;">📄</div>
                <p><strong>Нажмите для выбора файлов</strong></p>
                <p>Поддерживаемые форматы: JPG, JPEG, TIFF, PDF</p>
                <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.tiff,.tif,.pdf" style="display:none">
            </div>
            <div id="uploadedList"></div>
        </div>

        <div class="panel">
            <h2>📊 Статистика сессии</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalDocs">0</div>
                    <div class="stat-label">Обработано документов</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="avgConfidence">0%</div>
                    <div class="stat-label">Средняя уверенность</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="lowConfidenceItems">0</div>
                    <div class="stat-label">Требует проверки</div>
                </div>
            </div>
            <button class="btn btn-primary" onclick="refreshStats()">🔄 Обновить статистику</button>
        </div>
    </div>

    <!-- Шаг 1: Подбор сегментации -->
    <div class="panel">
        <h2>🧭 Шаг 1: Подбор сегментации</h2>
        <p>Настройте перцентили для горизонтальной (строки) и вертикальной (колонки) проекций, нажмите «Предпросмотр», проверьте графики и разлиновку, затем переходите к распознаванию.</p>

        <div style="display:flex; gap:30px; flex-wrap:wrap;">
            <div>
                <label>Горизонтальный перцентиль (hp): <span id="hpVal">75</span>%</label><br>
                <input id="hpRange" type="range" min="0" max="100" value="75" oninput="hpVal.innerText=this.value" style="width:300px;">
            </div>
            <div>
                <label>Вертикальный перцентиль (vp): <span id="vpVal">75</span>%</label><br>
                <input id="vpRange" type="range" min="0" max="100" value="75" oninput="vpVal.innerText=this.value" style="width:300px;">
            </div>
            <div>
                <label>Документ:</label><br>
                <select id="previewTaskSelect" style="min-width:200px;"></select>
            </div>
            <div style="align-self:flex-end;">
                <button class="btn btn-primary" onclick="runPreview()">Предпросмотр</button>
            </div>
        </div>

        <div style="margin-top:15px;">
            <h3>Горизонтальная проекция</h3>
            <canvas id="hpCanvas" width="900" height="180" style="background:#fff;border:1px solid #e1e1e1;"></canvas>
            <h3>Вертикальная проекция</h3>
            <canvas id="vpCanvas" width="900" height="180" style="background:#fff;border:1px solid #e1e1e1;"></canvas>
        </div>

        <div style="margin-top:10px;">
            <h3>Разлиновка (горизонталь и вертикаль)</h3>
            <img id="overlayImg" style="max-width:100%;border:1px solid #e1e1e1;">
        </div>

        <div style="margin-top:10px;font-size:0.95em;line-height:1.5;">
            <strong>Как выбирать порог:</strong><br>
            • Начните с 50% — середина графика, часто удачное значение для страниц со средним контрастом.<br>
            • Если строк выделяется мало — увеличивайте hp (70–85%), чтобы учитывать только самые чёткие полосы.<br>
            • Если теряются тонкие строки — уменьшайте hp (30–45%), чтобы учитывать слабые пики (межстрочные интервалы).<br>
            • Если колонки делятся неверно — регулируйте vp независимо; низкий vp выявляет тонкие вертикальные разделители, высокий vp оставляет только явные колонны.<br>
            • Добивайтесь, чтобы на графиках «полочки» были выше порога, а промежутки ниже — так линии разделятся корректно.
        </div>
    </div>

    <!-- Шаг 2: Запуск распознавания -->
    <div class="panel">
        <h2>🧩 Шаг 2: Распознавание</h2>
        <p>Нажмите «Распознать», чтобы запустить OCR с выбранными перцентилями сегментации для выбранного документа.</p>
        <button class="btn btn-success" onclick="startRecognize()">Распознать</button>
    </div>

    <div class="panel" id="progressPanel" style="display:none">
        <h2>⏳ Прогресс обработки</h2>
        <div id="progressList"></div>
    </div>

    <div class="panel" id="resultsPanel" style="display:none">
        <h2>✅ Результаты распознавания</h2>
        <div id="resultsList"></div>
    </div>
</div>

<script>
    let processedDocuments = [];     // completed
    let uploadedTasks = [];          // uploaded (ещё не распознаны)

    // Загрузка файлов
    document.getElementById('fileInput').addEventListener('change', function(e) {
        uploadFiles(e.target.files);
    });

    async function uploadFiles(files) {
        if (!files.length) return;
        const uploadedList = document.getElementById('uploadedList');

        for (let file of files) {
            if (file.size > 100 * 1024 * 1024) {
                alert(`Файл ${file.name} слишком большой (>100MB)`);
                continue;
            }

            const formData = new FormData();
            formData.append('file', file);

            const item = document.createElement('div');
            item.className = 'progress-item';
            item.innerHTML = `
                <div>
                    <strong>📄 ${file.name}</strong><br>
                    <span class="status">Загружается...</span>
                    <div class="loading" style="float:right;"></div>
                </div>`;
            uploadedList.appendChild(item);

            try {
                const resp = await fetch('/upload', { method: 'POST', body: formData });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const result = await resp.json();

                uploadedTasks.push(result.task_id);
                refreshPreviewList();

                item.querySelector('.status').textContent = `Загружено (ID: ${result.task_id.slice(0,8)}), настройте перцентили и нажмите «Предпросмотр»`;
                const loader = item.querySelector('.loading'); if (loader) loader.remove();
            } catch (err) {
                item.className = 'progress-item error';
                item.querySelector('.status').textContent = `Ошибка: ${err.message}`;
                const loader = item.querySelector('.loading'); if (loader) loader.remove();
            }
        }
    }

    function refreshPreviewList() {
        const sel = document.getElementById('previewTaskSelect');
        const all = [...new Set([...uploadedTasks, ...processedDocuments])];
        const current = sel.value;
        sel.innerHTML = '';
        for (const tid of all) {
            const opt = document.createElement('option');
            opt.value = tid;
            opt.textContent = tid.slice(0,8);
            sel.appendChild(opt);
        }
        if (current) sel.value = current;
    }

    async function runPreview() {
        const sel = document.getElementById('previewTaskSelect');
        const taskId = sel.value;
        if (!taskId) { alert('Выберите документ для предпросмотра'); return; }
        const hp = parseFloat(document.getElementById('hpRange').value);
        const vp = parseFloat(document.getElementById('vpRange').value);

        const res = await fetch(`/preview?task_id=${encodeURIComponent(taskId)}&hp_percentile=${hp}&vp_percentile=${vp}`);
        if (!res.ok) { alert('Ошибка предпросмотра'); return; }
        const data = await res.json();

        drawProjection('hpCanvas', data.hp_clip, data.thr_h, '#1f77b4', 'red');
        drawProjection('vpCanvas', data.vp_clip, data.thr_v, '#2ca02c', 'blue');
        document.getElementById('overlayImg').src = 'data:image/png;base64,' + data.overlay_b64;
    }

    async function startRecognize() {
        const sel = document.getElementById('previewTaskSelect');
        const taskId = sel.value;
        if (!taskId) { alert('Выберите документ для распознавания'); return; }
        const hp = parseFloat(document.getElementById('hpRange').value);
        const vp = parseFloat(document.getElementById('vpRange').value);

        const resp = await fetch('/recognize', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ task_id: taskId, hp_percentile: hp, vp_percentile: vp })
        });
        if (!resp.ok) { alert('Ошибка запуска распознавания'); return; }

        const progressPanel = document.getElementById('progressPanel');
        progressPanel.style.display = 'block';
        const progressItem = document.createElement('div');
        progressItem.className = 'progress-item';
        progressItem.innerHTML = `
            <div>
                <strong>📄 ${taskId.slice(0,8)}</strong><br>
                <span class="status">Запущено распознавание...</span>
                <div class="loading" style="float:right;"></div>
            </div>`;
        document.getElementById('progressList').appendChild(progressItem);

        monitorProgress(taskId, progressItem, taskId);
    }

    function drawProjection(canvasId, arr, thr, lineColor, thrColor) {
        const c = document.getElementById(canvasId);
        const ctx = c.getContext('2d');
        ctx.clearRect(0,0,c.width,c.height);
        if (!arr || arr.length === 0) return;

        const maxVal = Math.max(...arr) || 1;
        const scaleX = c.width / arr.length;
        const scaleY = (c.height - 10) / maxVal;

        // порог
        ctx.strokeStyle = thrColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        const yThr = c.height - Math.min(thr * scaleY, c.height - 1);
        ctx.moveTo(0, yThr);
        ctx.lineTo(c.width, yThr);
        ctx.stroke();

        // кривая
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const x = i * scaleX;
            const y = c.height - Math.min(arr[i] * scaleY, c.height - 1);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    async function monitorProgress(taskId, progressItem, filename) {
        const checkStatus = async () => {
            try {
                const response = await fetch(`/status/${taskId}`);
                const status = await response.json();

                if (status.status === 'completed') {
                    progressItem.className = 'progress-item completed';
                    progressItem.querySelector('.status').innerHTML = `
                        ✅ Готово<br>
                        <small>Уверенность: ${(status.confidence_score * 100).toFixed(1)}% |
                        Сегментов: ${status.total_segments} |
                        Время: ${status.processing_time.toFixed(1)}с</small>
                    `;
                    const loader = progressItem.querySelector('.loading'); if (loader) loader.remove();

                    processedDocuments.push(taskId);
                    showResults(taskId);
                    refreshStats();
                    refreshPreviewList();

                } else if (status.status === 'error') {
                    progressItem.className = 'progress-item error';
                    progressItem.querySelector('.status').textContent = '❌ Ошибка обработки';
                    const loader = progressItem.querySelector('.loading'); if (loader) loader.remove();
                } else {
                    setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                progressItem.className = 'progress-item error';
                progressItem.querySelector('.status').textContent = `❌ Ошибка связи: ${error.message}`;
                const loader = progressItem.querySelector('.loading'); if (loader) loader.remove();
            }
        };
        checkStatus();
    }

    async function showResults(taskId) {
        const resultsPanel = document.getElementById('resultsPanel');
        const resultsList = document.getElementById('resultsList');
        resultsPanel.style.display = 'block';

        try {
            const response = await fetch(`/results/${taskId}`);
            const data = await response.json();

            const documentCard = document.createElement('div');
            documentCard.className = 'document-card';
            documentCard.innerHTML = `
                <div class="document-header">
                    <h3>📄 Документ ${taskId.slice(0, 8)}...</h3>
                    <div>
                        Уверенность: ${(data.confidence_score * 100).toFixed(1)}% |
                        Сегментов: ${data.segments.length} |
                        Требует проверки: ${data.low_confidence_count}
                    </div>
                </div>
                <div class="document-body">
                    <div style="margin-bottom: 20px;">
                        <button class="btn btn-primary" onclick="toggleSegments('${taskId}')">👁️ Показать/скрыть сегменты</button>
                    </div>
                    <div id="segments-${taskId}" style="display:none">
                        ${data.segments.map(segment => `
                            <div class="segment-box ${getConfidenceClass(segment.confidence)}">
                                <div style="margin-bottom: 10px;">
                                    <strong>Сегмент ${segment.segment_id}</strong>
                                    (${(segment.confidence * 100).toFixed(1)}%)
                                </div>
                                <div style="margin: 10px 0;">
                                    <strong>Исходный текст:</strong><br>
                                    <div class="text-block" style="background:#f8f9fa;padding:10px;border-radius:5px;margin:5px 0;">
                                        ${segment.raw_text ? segment.raw_text.replace(/</g,'&lt;').replace(/>/g,'&gt;') : '<em>Пусто</em>'}
                                    </div>
                                </div>
                                <div style="margin: 10px 0;">
                                    <strong>Обработанный текст:</strong><br>
                                    <div class="text-block" style="background:#e8f5e8;padding:10px;border-radius:5px;margin:5px 0;">
                                        ${segment.corrected_text ? segment.corrected_text.replace(/</g,'&lt;').replace(/>/g,'&gt;') : '<em>Пусто</em>'}
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            resultsList.appendChild(documentCard);
        } catch (error) {
            console.error('Ошибка получения результатов:', error);
        }
    }

    function toggleSegments(taskId) {
        const segments = document.getElementById(`segments-${taskId}`);
        segments.style.display = segments.style.display === 'none' ? 'block' : 'none';
    }

    function getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'high-confidence';
        if (confidence >= 0.5) return 'medium-confidence';
        return 'low-confidence';
    }

    async function refreshStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            document.getElementById('totalDocs').textContent = stats.total_documents;
            document.getElementById('avgConfidence').textContent = Math.round(stats.avg_confidence * 100) + '%';
            document.getElementById('lowConfidenceItems').textContent = stats.low_confidence_segments || 0;
        } catch (error) {
            console.error('Ошибка получения статистики:', error);
        }
    }

    refreshStats();
    setInterval(refreshStats, 30000);
    document.addEventListener('DOMContentLoaded', refreshPreviewList);
</script>
</body>
</html>"""
    return html_content


@app.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    """Загрузка документа"""

    # Проверки размера и формата
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"Файл слишком большой (максимум {MAX_FILE_SIZE // 1024 // 1024}MB)")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ['.jpg', '.jpeg', '.tiff', '.tif', '.pdf']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

    # Создаем задачу
    task_id = str(uuid.uuid4())
    file_path = await save_uploaded_file(file, task_id)

    # Обработка PDF
    if file_extension == '.pdf':
        image_paths = convert_pdf_to_images(file_path)
        if image_paths:
            file_path = image_paths[0]  # Берем первую страницу

    # Сохраняем в БД
    doc = DocumentModel(
        task_id=task_id,
        filename=file.filename,
        file_path=file_path,
        status="uploaded",  # остается uploaded до команды пользователя
        file_size=file.size
    )
    db.add(doc)
    db.commit()

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "file_size": file.size
    }


@app.get("/preview")
async def preview(
        task_id: str,
        hp_percentile: float = 75.0,
        vp_percentile: float = 75.0,
        db: Session = Depends(get_db)
):
    """Предпросмотр с двумя перцентилями"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    img = cv2.imread(doc.file_path)
    if img is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape

    hp = np.sum(gray < 128, axis=1)
    vp = np.sum(gray < 128, axis=0)

    # Раздельная обрезка «хвостов»
    hp_clip = np.clip(hp, 0, np.percentile(hp, hp_percentile))
    vp_clip = np.clip(vp, 0, np.percentile(vp, vp_percentile))

    thr_h = float(np.median(hp_clip))
    thr_v = float(np.median(vp_clip))

    def find_lines(proj, thr, min_len=10):
        lines, in_band = [], False
        for i, v in enumerate(proj):
            if v > thr and not in_band:
                start = i;
                in_band = True
            elif v <= thr and in_band:
                end = i;
                in_band = False
                if end - start >= min_len:
                    lines.append([start, end])
        if in_band:
            lines.append([start, len(proj)])
        return lines

    h_lines = find_lines(hp_clip, thr_h)
    v_lines = find_lines(vp_clip, thr_v)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for y0, y1 in h_lines:
        cv2.line(overlay, (0, y0), (w, y0), (0, 255, 0), 1)
        cv2.line(overlay, (0, y1), (w, y1), (0, 255, 0), 1)
    for x0, x1 in v_lines:
        cv2.line(overlay, (x0, 0), (x0, h), (255, 0, 0), 1)
        cv2.line(overlay, (x1, 0), (x1, h), (255, 0, 0), 1)

    _, buf = cv2.imencode(".png", overlay)
    b64_overlay = base64.b64encode(buf.tobytes()).decode()

    return {
        "hp_percentile": hp_percentile,
        "vp_percentile": vp_percentile,
        "hp_clip": hp_clip.tolist(),
        "vp_clip": vp_clip.tolist(),
        "thr_h": thr_h,
        "thr_v": thr_v,
        "overlay_b64": b64_overlay,
        "width": w, "height": h,
        "h_lines": h_lines, "v_lines": v_lines
    }


@app.post("/recognize")
async def recognize_endpoint(req: RecognizeRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Запуск распознавания с двумя перцентилями"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == req.task_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if not (0.0 <= req.hp_percentile <= 100.0) or not (0.0 <= req.vp_percentile <= 100.0):
        raise HTTPException(status_code=400, detail="percentile должен быть 0..100")

    background_tasks.add_task(
        process_document_enhanced_async_with_percentiles,
        req.task_id, doc.file_path, req.hp_percentile, req.vp_percentile
    )
    return {"ok": True, "task_id": req.task_id, "hp_percentile": req.hp_percentile, "vp_percentile": req.vp_percentile}


async def process_document_enhanced_async_with_percentiles(task_id: str, file_path: str, hp_percentile: float,
                                                           vp_percentile: float):
    """Фоновая задача с передачей параметров"""
    try:
        with SessionLocal() as new_db:
            await document_processor.process_document_enhanced(
                task_id, file_path, new_db,
                hp_percentile=hp_percentile,
                vp_percentile=vp_percentile
            )
    except Exception as e:
        logger.error(f"Ошибка фоновой обработки {task_id}: {e}")
        with SessionLocal() as error_db:
            doc = error_db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()
            if doc:
                doc.status = "error"
                error_db.commit()


@app.get("/status/{task_id}")
async def get_enhanced_status(task_id: str, db: Session = Depends(get_db)):
    """Получение статуса обработки"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    return DocumentResponse(
        task_id=doc.task_id,
        filename=doc.filename,
        status=doc.status,
        confidence_score=doc.confidence_score or 0.0,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
        total_segments=doc.total_segments or 0,
        low_confidence_count=doc.low_confidence_count or 0,
        processing_time=doc.processing_time or 0.0
    )


@app.get("/results/{task_id}")
async def get_enhanced_results(task_id: str, db: Session = Depends(get_db)):
    """Получение результатов распознавания"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if doc.status != "completed":
        raise HTTPException(status_code=404, detail="Результаты не готовы")

    # Получаем сегменты
    segments = db.query(TextSegmentModel).filter(TextSegmentModel.document_id == doc.id).all()
    segments_data = []

    for seg in segments:
        segments_data.append({
            'id': seg.id,
            'segment_id': seg.segment_id,
            'raw_text': seg.raw_text or "",
            'corrected_text': seg.corrected_text or "",
            'bbox': seg.bbox or [0, 0, 100, 100],
            'confidence': seg.confidence or 0.0,
            'is_verified': seg.is_verified or False,
            'is_corrected': seg.is_corrected or False,
            'correction_count': seg.correction_count or 0
        })

    return {
        "task_id": task_id,
        "confidence_score": doc.confidence_score or 0.0,
        "low_confidence_count": doc.low_confidence_count or 0,
        "total_segments": doc.total_segments or 0,
        "processing_time": doc.processing_time or 0.0,
        "segments": segments_data,
        "attributes": []
    }


@app.get("/stats")
async def get_enhanced_stats(db: Session = Depends(get_db)):
    total_docs = db.query(DocumentModel).count()
    processed_docs = db.query(DocumentModel).filter(DocumentModel.status == "completed").count()

    completed_docs = db.query(DocumentModel).filter(DocumentModel.status == "completed").all()

    if completed_docs:
        avg_confidence = sum(doc.confidence_score or 0.0 for doc in completed_docs) / len(completed_docs)
        low_confidence_segments = sum(doc.low_confidence_count or 0 for doc in completed_docs)
    else:
        avg_confidence = 0.0
        low_confidence_segments = 0

    return {
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "avg_confidence": avg_confidence,
        "low_confidence_segments": low_confidence_segments
    }


@app.post("/set_percentile")
async def set_percentile(value: float = Body(embed=True)):
    """Устанавливает перцентиль по умолчанию для всех следующих обработок"""
    global DEFAULT_PERCENTILE
    if value < 0 or value > 100:
        raise HTTPException(status_code=400, detail="percentile должен быть в диапазоне 0..100")
    DEFAULT_PERCENTILE = float(value)
    return {"ok": True, "percentile": DEFAULT_PERCENTILE}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.1.0",
        "features": [
            "document_upload",
            "ocr_processing",
            "statistics_monitoring",
            "dual_percentile_preview"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "web_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
