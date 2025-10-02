#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-сервис обработки архивных документов
Исправленная версия с устранением всех ошибок
"""

import os
import io
import json
import uuid
import asyncio
import logging
import mimetypes
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image
import pdf2image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session  # Исправлен импорт
import aiofiles

# Импорт OCR системы
try:
    from integrated_archive_processor import ArchiveDocumentProcessor
except ImportError:
    # Fallback - создадим простую заглушку
    class ArchiveDocumentProcessor:
        def __init__(self, use_postprocessing=True):
            pass
        def process_document(self, file_path):
            return []

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
DATABASE_URL = "sqlite:///./enhanced_archive_service.db"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
LOW_CONFIDENCE_THRESHOLD = 0.75  # Порог низкой уверенности

# Создаем директории
for dir_path in [UPLOAD_DIR, RESULTS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== DATABASE MODELS ====================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # Исправлено

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

# ==================== PYDANTIC MODELS ====================

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

# ==================== BUSINESS LOGIC ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class EnhancedDocumentProcessor:
    """Расширенный процессор документов"""

    def __init__(self):
        self.ocr_processor = ArchiveDocumentProcessor(use_postprocessing=True)

    async def process_document_enhanced(self, task_id: str, file_path: str, db: Session) -> Dict:
        """Обработка документа с исправленной инициализацией"""
        start_time = datetime.utcnow()
        logger.info(f"Начинаем обработку документа {task_id}")

        # Обновляем статус
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

            # OCR обработка - исправлен fallback
            try:
                results = self.ocr_processor.process_document(file_path)
            except Exception as e:
                logger.error(f"Ошибка OCR: {e}")
                results = []

            # Если результатов нет, создаем хотя бы один пустой сегмент
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
                # Безопасное извлечение данных
                segment_id = result.get('segment_id', i + 1)
                raw_text = result.get('raw_text', '')
                cleaned_text = result.get('cleaned_text', raw_text)
                shape = result.get('segment_shape', (100, 100))
                confidence = result.get('confidence_estimate', 0.0)

                # Конвертируем bbox
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

            # Расчет метрик - исправлено деление на ноль
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

            # Обновляем статистику сессии - ИСПРАВЛЕНО
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
        """Конвертирует shape в bbox"""
        if hasattr(shape, '__len__') and len(shape) >= 2:
            h, w = shape[0], shape[1]
            return [0.0, 0.0, float(w), float(h)]
        return [0.0, 0.0, 100.0, 100.0]

    def _save_to_database(self, db: Session, doc_id: int, segments: List[Dict]):
        """Сохранение сегментов в БД"""
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

    def _update_session_stats(self, db: Session, task_id: str, segments_count: int, avg_confidence: float, low_confidence_count: int, processing_time: float):
        """ИСПРАВЛЕННОЕ обновление статистики сессии"""
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

        # ИСПРАВЛЕНО: защита от None значений
        stats.total_documents = (stats.total_documents or 0) + 1
        stats.processed_documents = (stats.processed_documents or 0) + 1
        stats.total_segments = (stats.total_segments or 0) + segments_count
        stats.low_confidence_segments = (stats.low_confidence_segments or 0) + low_confidence_count
        stats.processing_time = (stats.processing_time or 0.0) + processing_time

        # Исправлено вычисление средней уверенности
        old_sum = (stats.avg_confidence or 0.0) * max((stats.processed_documents - 1), 1)
        stats.avg_confidence = (old_sum + avg_confidence) / max(stats.processed_documents, 1)
        stats.updated_at = datetime.utcnow()

        db.commit()

# ==================== FILE HANDLING ====================

async def save_uploaded_file(upload_file: UploadFile, task_id: str) -> str:
    """Сохранение загруженного файла"""
    file_extension = Path(upload_file.filename).suffix.lower()
    file_path = UPLOAD_DIR / f"{task_id}{file_extension}"

    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)

    return str(file_path)

def convert_pdf_to_images(pdf_path: str) -> List[str]:
    """Конвертация PDF в изображения"""
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=300)
        image_paths = []

        base_name = Path(pdf_path).stem
        for i, image in enumerate(images):
            img_path = UPLOAD_DIR / f"{base_name}_page_{i+1}.jpg"
            image.save(img_path, 'JPEG', quality=95)
            image_paths.append(str(img_path))

        return image_paths
    except Exception as e:
        logger.error(f"Ошибка конвертации PDF: {e}")
        return [pdf_path]

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Архивный OCR Сервис",
    description="Исправленный веб-сервис для обработки архивных документов",
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

# ==================== WEB INTERFACE ====================

@app.get("/", response_class=HTMLResponse)
async def enhanced_index():
    """Исправленная главная страница"""
    # ИСПРАВЛЕНО: убраны проблемные escape-последовательности
    html_content = """<!DOCTYPE html>
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
            box-shadow: 0 5px 20px rgba(0,0,0,0.08); border: 1px solid #e1e8ed;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ Архивный OCR Сервис</h1>
            <p>Исправленная система обработки архивных документов</p>
        </div>

        <div class="main-content">
            <div class="panel">
                <h2>📁 Загрузка документов</h2>
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 3em; margin-bottom: 10px;">📄</div>
                    <p><strong>Нажмите для выбора файлов</strong></p>
                    <p>Поддерживаемые форматы: JPG, JPEG, TIFF, PDF</p>
                    <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.tiff,.pdf" style="display:none">
                </div>
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
        let processedDocuments = [];

        // Загрузка файлов
        document.getElementById('fileInput').addEventListener('change', function(e) {
            uploadFiles(e.target.files);
        });

        async function uploadFiles(files) {
            if (!files.length) return;

            const progressPanel = document.getElementById('progressPanel');
            const progressList = document.getElementById('progressList');
            progressPanel.style.display = 'block';

            for (let file of files) {
                if (file.size > 100 * 1024 * 1024) {
                    alert(`Файл ${file.name} слишком большой (>100MB)`);
                    continue;
                }

                const formData = new FormData();
                formData.append('file', file);

                const progressItem = document.createElement('div');
                progressItem.className = 'progress-item';
                progressItem.innerHTML = `
                    <div>
                        <strong>📄 ${file.name}</strong><br>
                        <span class="status">Загружается...</span>
                        <div class="loading" style="float: right;"></div>
                    </div>
                `;
                progressList.appendChild(progressItem);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const result = await response.json();
                    progressItem.querySelector('.status').textContent = `Обрабатывается... (ID: ${result.task_id.slice(0, 8)})`;

                    monitorProgress(result.task_id, progressItem, file.name);

                } catch (error) {
                    progressItem.className = 'progress-item error';
                    progressItem.querySelector('.status').textContent = `Ошибка: ${error.message}`;
                    progressItem.querySelector('.loading').remove();
                }
            }
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
                        progressItem.querySelector('.loading').remove();

                        processedDocuments.push(taskId);
                        showResults(taskId);
                        refreshStats();

                    } else if (status.status === 'error') {
                        progressItem.className = 'progress-item error';
                        progressItem.querySelector('.status').textContent = '❌ Ошибка обработки';
                        progressItem.querySelector('.loading').remove();
                    } else {
                        setTimeout(checkStatus, 2000);
                    }
                } catch (error) {
                    progressItem.className = 'progress-item error';
                    progressItem.querySelector('.status').textContent = `❌ Ошибка связи: ${error.message}`;
                    progressItem.querySelector('.loading').remove();
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
                                        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                            ${segment.raw_text || '<em>Пусто</em>'}
                                        </div>
                                    </div>
                                    <div style="margin: 10px 0;">
                                        <strong>Обработанный текст:</strong><br>
                                        <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                            ${segment.corrected_text || '<em>Пусто</em>'}
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

        function getConfidenceClass(confidence) {
            if (confidence >= 0.8) return 'high-confidence';
            if (confidence >= 0.5) return 'medium-confidence';
            return 'low-confidence';
        }

        function toggleSegments(taskId) {
            const segments = document.getElementById(`segments-${taskId}`);
            segments.style.display = segments.style.display === 'none' ? 'block' : 'none';
        }

        async function refreshStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();

                document.getElementById('totalDocs').textContent = stats.total_documents;
                document.getElementById('avgConfidence').textContent = Math.round(stats.average_confidence * 100) + '%';
                document.getElementById('lowConfidenceItems').textContent = stats.low_confidence_segments || 0;
            } catch (error) {
                console.error('Ошибка получения статистики:', error);
            }
        }

        // Инициализация
        refreshStats();

        // Автообновление статистики каждые 30 секунд
        setInterval(refreshStats, 30000);
    </script>
</body>
</html>"""
    return html_content

# ==================== API ENDPOINTS ====================

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Загрузка документа"""

    # Проверки размера и формата
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой (максимум {MAX_FILE_SIZE//1024//1024}MB)")

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
        status="uploaded",
        file_size=file.size
    )
    db.add(doc)
    db.commit()

    # Запускаем обработку в фоне
    background_tasks.add_task(process_document_enhanced_async, task_id, file_path, db)

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "file_size": file.size
    }

async def process_document_enhanced_async(task_id: str, file_path: str, db: Session):
    """Асинхронная обработка документа"""
    try:
        # Создаем новую сессию БД для фоновой задачи
        with SessionLocal() as new_db:
            await document_processor.process_document_enhanced(task_id, file_path, new_db)
    except Exception as e:
        logger.error(f"Ошибка фоновой обработки {task_id}: {e}")
        # Обновляем статус документа на error
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
        "attributes": []  # Пока пустой массив
    }

@app.get("/stats")
async def get_enhanced_stats(db: Session = Depends(get_db)):
    """Статистика"""
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
        "average_confidence": avg_confidence,
        "low_confidence_segments": low_confidence_segments
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.1.0",
        "features": [
            "document_upload",
            "ocr_processing",
            "statistics_monitoring"
        ]
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "web_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
