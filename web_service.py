#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-сервис обработки архивных документов
Реализует требования ТЗ: загрузка, OCR, верификация, экспорт

Основные функции:
- Загрузка образов (JPG, TIFF, PDF)
- Предварительная обработка и нормализация
- OCR с поддержкой дореволюционной орфографии  
- Атрибутивное распознавание (ФИО, даты, адреса)
- Верификация и коррекция результатов
- Экспорт данных по критериям

Дата: 2025
"""

import os
import io
import json
import uuid
import asyncio
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image
import pdf2image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import aiofiles

# Импорт нашей системы OCR
from integrated_archive_processor import ArchiveDocumentProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")  
DATABASE_URL = "sqlite:///./archive_service.db"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Создаем директории
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# ==================== DATABASE MODELS ====================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    status = Column(String, default="uploaded")  # uploaded, processing, completed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    confidence_score = Column(Float, default=0.0)
    low_confidence_count = Column(Integer, default=0)
    ocr_engine = Column(String, default="paddle")

class TextSegmentModel(Base):
    __tablename__ = "text_segments"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer)
    segment_id = Column(Integer)
    raw_text = Column(Text)
    corrected_text = Column(Text)
    bbox = Column(JSON)  # [x1, y1, x2, y2]
    confidence = Column(Float)
    is_verified = Column(Boolean, default=False)
    is_corrected = Column(Boolean, default=False)

class AttributeModel(Base):
    __tablename__ = "attributes"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer)
    attr_type = Column(String)  # name, date, address, archive_code
    attr_value = Column(String)
    confidence = Column(Float)
    bbox = Column(JSON)
    is_verified = Column(Boolean, default=False)

class ProcessingStatsModel(Base):
    __tablename__ = "processing_stats"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC MODELS ====================

class DocumentResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    confidence_score: float
    created_at: datetime
    processed_at: Optional[datetime]

class TextSegmentResponse(BaseModel):
    id: int
    segment_id: int
    raw_text: str
    corrected_text: str
    bbox: List[float]
    confidence: float
    is_verified: bool
    is_corrected: bool

class AttributeResponse(BaseModel):
    id: int
    attr_type: str
    attr_value: str
    confidence: float
    bbox: List[float]
    is_verified: bool

class CorrectionRequest(BaseModel):
    segment_id: int
    corrected_text: str

class ExportRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    min_confidence: Optional[float] = 0.0
    include_attributes: List[str] = ["name", "date", "address", "archive_code"]
    export_format: str = "json"  # json, csv, xlsx

# ==================== DEPENDENCY ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== BUSINESS LOGIC ====================

@dataclass
class ProcessingResult:
    task_id: str
    segments: List[Dict]
    attributes: List[Dict]
    confidence_score: float
    low_confidence_count: int

class AttributeExtractor:
    """Извлечение атрибутов из текста (ФИО, даты, адреса, архивные шифры)"""

    def __init__(self):
        import re
        self.patterns = {
            'name': [
                r'\b([А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+)\b',  # ФИО
                r'\b([А-ЯЁ][а-яё]+ [А-ЯЁ]\. [А-ЯЁ]\.)\b',  # Фамилия И.О.
            ],
            'date': [
                r'\b(\d{1,2}\s+[а-яё]{3,}\s+\d{4})\b',  # 15 сентября 1885
                r'\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b',    # 15.09.1885
                r'\b(\d{4}\s*г\.)\b',                    # 1885 г.
            ],
            'address': [
                r'(?:село|деревня|город|слобода)\s+([А-ЯЁ][а-яё]+)',
                r'(?:губернии|уезда|волости)\s+([А-ЯЁ][а-яё]+)',
            ],
            'archive_code': [
                r'\b(Ф\.\s*\d+\s*Оп\.\s*\d+\s*Д\.\s*\d+)\b',  # Ф.1 Оп.2 Д.3
                r'\b([А-Я]{1,5}-\d+)\b',  # АБВ-123
            ]
        }

    def extract(self, text: str, bbox_info: Dict = None) -> List[Dict]:
        import re
        attributes = []

        for attr_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    attr_value = match.group(1) if match.groups() else match.group(0)
                    confidence = 0.8  # Базовая уверенность для regex

                    # Примерные координаты (в реальности нужно точное позиционирование)
                    attr_bbox = bbox_info.get('bbox', [0, 0, 100, 100]) if bbox_info else [0, 0, 100, 100]

                    attributes.append({
                        'attr_type': attr_type,
                        'attr_value': attr_value.strip(),
                        'confidence': confidence,
                        'bbox': attr_bbox
                    })

        return attributes

class DocumentProcessor:
    """Основной класс обработки документов"""

    def __init__(self):
        self.ocr_processor = ArchiveDocumentProcessor(
            use_postprocessing=True
        )
        self.attribute_extractor = AttributeExtractor()

    async def process_document(self, task_id: str, file_path: str, db: Session) -> ProcessingResult:
        """Полная обработка документа"""
        logger.info(f"Начинаем обработку документа {task_id}")

        # Обновляем статус
        doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()
        doc.status = "processing"
        db.commit()

        try:
            # OCR обработка
            results = self.ocr_processor.process_document(file_path)

            segments = []
            all_attributes = []
            total_confidence = 0
            low_confidence_count = 0

            for result in results:
                # Сохраняем сегмент текста
                segment = {
                    'segment_id': result['segment_id'],
                    'raw_text': result['raw_text'],
                    'corrected_text': result['cleaned_text'],
                    'bbox': self._shape_to_bbox(result['segment_shape']),
                    'confidence': result['confidence_estimate']
                }
                segments.append(segment)

                total_confidence += result['confidence_estimate']
                if result['confidence_estimate'] < 0.7:
                    low_confidence_count += 1

                # Извлекаем атрибуты
                text_for_extraction = result['cleaned_text'] or result['raw_text']
                if text_for_extraction:
                    attrs = self.attribute_extractor.extract(
                        text_for_extraction, 
                        {'bbox': segment['bbox']}
                    )
                    all_attributes.extend(attrs)

            avg_confidence = total_confidence / len(results) if results else 0.0

            # Сохраняем в БД
            self._save_to_database(db, doc.id, segments, all_attributes)

            # Обновляем документ
            doc.status = "completed"
            doc.processed_at = datetime.utcnow()
            doc.confidence_score = avg_confidence
            doc.low_confidence_count = low_confidence_count
            db.commit()

            return ProcessingResult(
                task_id=task_id,
                segments=segments,
                attributes=all_attributes,
                confidence_score=avg_confidence,
                low_confidence_count=low_confidence_count
            )

        except Exception as e:
            logger.error(f"Ошибка обработки {task_id}: {e}")
            doc.status = "error"
            db.commit()
            raise HTTPException(status_code=500)

    def _shape_to_bbox(self, shape) -> List[float]:
        """Конвертирует shape сегмента в bbox"""
        if len(shape) >= 2:
            h, w = shape[0], shape[1]
            return [0.0, 0.0, float(w), float(h)]
        return [0.0, 0.0, 100.0, 100.0]

    def _save_to_database(self, db: Session, doc_id: int, segments: List[Dict], attributes: List[Dict]):
        """Сохранение результатов в БД"""
        # Сохраняем сегменты
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

        # Сохраняем атрибуты
        for attr in attributes:
            db_attr = AttributeModel(
                document_id=doc_id,
                attr_type=attr['attr_type'],
                attr_value=attr['attr_value'],
                confidence=attr['confidence'],
                bbox=attr['bbox']
            )
            db.add(db_attr)

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
    images = pdf2image.convert_from_path(pdf_path, dpi=300)
    image_paths = []

    base_name = Path(pdf_path).stem
    for i, image in enumerate(images):
        img_path = UPLOAD_DIR / f"{base_name}_page_{i+1}.jpg"
        image.save(img_path, 'JPEG', quality=95)
        image_paths.append(str(img_path))

    return image_paths

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Архивный OCR Сервис",
    description="Веб-сервис для обработки архивных документов с распознаванием текста",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальный процессор
document_processor = DocumentProcessor()

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Архивный OCR Сервис</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #999; }
            .results { margin-top: 30px; }
            .document { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
            .segment { background: #f5f5f5; margin: 5px 0; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>🏛️ Архивный OCR Сервис</h1>
        <p>Загрузите архивные документы для автоматического распознавания текста</p>

        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>📁 Нажмите для выбора файлов или перетащите сюда</p>
            <p>Поддерживаемые форматы: JPG, JPEG, TIFF, PDF</p>
            <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.tiff,.pdf" style="display:none" onchange="uploadFiles(this.files)">
        </div>

        <div id="progress" style="display:none">
            <h3>📊 Прогресс обработки</h3>
            <div id="progressList"></div>
        </div>

        <div id="results" class="results" style="display:none">
            <h3>✅ Результаты</h3>
            <div id="resultsList"></div>
        </div>

        <script>
        async function uploadFiles(files) {
            const progress = document.getElementById('progress');
            const progressList = document.getElementById('progressList');
            progress.style.display = 'block';

            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);

                const progressItem = document.createElement('div');
                progressItem.innerHTML = `📄 ${file.name}: Загружается...`;
                progressList.appendChild(progressItem);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    progressItem.innerHTML = `📄 ${file.name}: Обрабатывается... (ID: ${result.task_id})`;

                    // Отслеживаем прогресс
                    checkProgress(result.task_id, progressItem, file.name);

                } catch (error) {
                    progressItem.innerHTML = `❌ ${file.name}: Ошибка загрузки`;
                }
            }
        }

        async function checkProgress(taskId, progressItem, filename) {
            const checkStatus = async () => {
                try {
                    const response = await fetch(`/status/${taskId}`);
                    const status = await response.json();

                    if (status.status === 'completed') {
                        progressItem.innerHTML = `✅ ${filename}: Готово (${(status.confidence_score * 100).toFixed(1)}% уверенности)`;
                        showResults(taskId);
                    } else if (status.status === 'error') {
                        progressItem.innerHTML = `❌ ${filename}: Ошибка обработки`;
                    } else {
                        progressItem.innerHTML = `⏳ ${filename}: Обрабатывается...`;
                        setTimeout(checkStatus, 2000);
                    }
                } catch (error) {
                    progressItem.innerHTML = `❌ ${filename}: Ошибка проверки статуса`;
                }
            };

            checkStatus();
        }

        async function showResults(taskId) {
            const results = document.getElementById('results');
            const resultsList = document.getElementById('resultsList');
            results.style.display = 'block';

            try {
                const response = await fetch(`/results/${taskId}`);
                const data = await response.json();

                const resultDiv = document.createElement('div');
                resultDiv.className = 'document';
                resultDiv.innerHTML = `
                    <h4>📄 Документ ${taskId}</h4>
                    <p><strong>Уверенность:</strong> ${(data.confidence_score * 100).toFixed(1)}%</p>
                    <p><strong>Сегментов с низкой уверенностью:</strong> ${data.low_confidence_count}</p>
                    <div id="segments-${taskId}"></div>
                    <button onclick="exportData('${taskId}')">📊 Экспорт данных</button>
                    <button onclick="showVerification('${taskId}')">✏️ Верификация</button>
                `;

                // Показываем сегменты
                const segmentsDiv = document.getElementById(`segments-${taskId}`);
                data.segments.forEach(segment => {
                    const segmentDiv = document.createElement('div');
                    segmentDiv.className = 'segment';
                    segmentDiv.innerHTML = `
                        <strong>Сегмент ${segment.segment_id} (${(segment.confidence * 100).toFixed(1)}%)</strong><br>
                        <strong>Исходный:</strong> ${segment.raw_text}<br>
                        <strong>Исправленный:</strong> ${segment.corrected_text}
                    `;
                    segmentsDiv.appendChild(segmentDiv);
                });

                resultsList.appendChild(resultDiv);
            } catch (error) {
                console.error('Ошибка получения результатов:', error);
            }
        }

        async function exportData(taskId) {
            window.open(`/export/${taskId}?format=json`, '_blank');
        }

        function showVerification(taskId) {
            alert(`Функция верификации для документа ${taskId} (реализуется отдельным интерфейсом)`);
        }
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Загрузка документа"""

    # Проверки
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ['.jpg', '.jpeg', '.tiff', '.pdf']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

    # Создаем задачу
    task_id = str(uuid.uuid4())
    file_path = await save_uploaded_file(file, task_id)

    # Сохраняем в БД
    doc = DocumentModel(
        task_id=task_id,
        filename=file.filename,
        file_path=file_path,
        status="uploaded"
    )
    db.add(doc)
    db.commit()

    # Запускаем обработку в фоне
    background_tasks.add_task(process_document_async, task_id, file_path, db)

    return {"task_id": task_id, "filename": file.filename, "status": "uploaded"}

async def process_document_async(task_id: str, file_path: str, db: Session):
    """Асинхронная обработка документа"""
    try:
        await document_processor.process_document(task_id, file_path, db)
    except Exception as e:
        logger.error(f"Ошибка фоновой обработки {task_id}: {e}")

@app.get("/status/{task_id}")
async def get_status(task_id: str, db: Session = Depends(get_db)):
    """Получение статуса обработки"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    return DocumentResponse(
        task_id=doc.task_id,
        filename=doc.filename,
        status=doc.status,
        confidence_score=doc.confidence_score,
        created_at=doc.created_at,
        processed_at=doc.processed_at
    )

@app.get("/results/{task_id}")
async def get_results(task_id: str, db: Session = Depends(get_db)):
    """Получение результатов распознавания"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()

    if not doc or doc.status != "completed":
        raise HTTPException(status_code=404, detail="Результаты не готовы")

    # Получаем сегменты
    segments = db.query(TextSegmentModel).filter(TextSegmentModel.document_id == doc.id).all()
    segments_data = [TextSegmentResponse(
        id=seg.id,
        segment_id=seg.segment_id,
        raw_text=seg.raw_text,
        corrected_text=seg.corrected_text,
        bbox=seg.bbox,
        confidence=seg.confidence,
        is_verified=seg.is_verified,
        is_corrected=seg.is_corrected
    ) for seg in segments]

    # Получаем атрибуты
    attributes = db.query(AttributeModel).filter(AttributeModel.document_id == doc.id).all()
    attributes_data = [AttributeResponse(
        id=attr.id,
        attr_type=attr.attr_type,
        attr_value=attr.attr_value,
        confidence=attr.confidence,
        bbox=attr.bbox,
        is_verified=attr.is_verified
    ) for attr in attributes]

    return {
        "task_id": task_id,
        "confidence_score": doc.confidence_score,
        "low_confidence_count": doc.low_confidence_count,
        "segments": segments_data,
        "attributes": attributes_data
    }

@app.post("/correct/{task_id}")
async def correct_text(
    task_id: str,
    correction: CorrectionRequest,
    db: Session = Depends(get_db)
):
    """Коррекция распознанного текста"""
    doc = db.query(DocumentModel).filter(DocumentModel.task_id == task_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    segment = db.query(TextSegmentModel).filter(
        TextSegmentModel.document_id == doc.id,
        TextSegmentModel.segment_id == correction.segment_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Сегмент не найден")

    # Сохраняем коррекцию
    segment.corrected_text = correction.corrected_text
    segment.is_corrected = True
    segment.is_verified = True
    db.commit()

    return {"status": "corrected", "segment_id": correction.segment_id}

@app.post("/export")
async def export_data(export_request: ExportRequest, db: Session = Depends(get_db)):
    """Экспорт данных по критериям"""

    # Базовый запрос
    query = db.query(DocumentModel)

    if export_request.document_ids:
        query = query.filter(DocumentModel.task_id.in_(export_request.document_ids))

    if export_request.min_confidence > 0:
        query = query.filter(DocumentModel.confidence_score >= export_request.min_confidence)

    documents = query.all()

    # Собираем данные для экспорта
    export_data = []
    for doc in documents:
        segments = db.query(TextSegmentModel).filter(TextSegmentModel.document_id == doc.id).all()
        attributes = db.query(AttributeModel).filter(AttributeModel.document_id == doc.id).all()

        doc_data = {
            "task_id": doc.task_id,
            "filename": doc.filename,
            "confidence_score": doc.confidence_score,
            "segments": [{"text": seg.corrected_text, "confidence": seg.confidence} for seg in segments],
            "attributes": {attr.attr_type: attr.attr_value for attr in attributes if attr.attr_type in export_request.include_attributes}
        }
        export_data.append(doc_data)

    # Формируем ответ в зависимости от формата
    if export_request.export_format == "json":
        return JSONResponse(content=export_data)
    elif export_request.export_format == "csv":
        # Упрощенная CSV генерация
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["task_id", "filename", "confidence", "text", "attributes"])

        for doc in export_data:
            text = " ".join([seg["text"] for seg in doc["segments"]])
            attrs = json.dumps(doc["attributes"])
            writer.writerow([doc["task_id"], doc["filename"], doc["confidence_score"], text, attrs])

        response = StreamingResponse(io.StringIO(output.getvalue()), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        return response

    return JSONResponse(content={"error": "Неподдерживаемый формат"})

@app.get("/export/{task_id}")
async def export_single_document(
    task_id: str, 
    format: str = Query(default="json"),
    db: Session = Depends(get_db)
):
    """Экспорт одного документа"""
    export_request = ExportRequest(
        document_ids=[task_id],
        export_format=format,
        include_attributes=["name", "date", "address", "archive_code"]
    )
    return await export_data(export_request, db)

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Статистика обработки"""
    total_docs = db.query(DocumentModel).count()
    processed_docs = db.query(DocumentModel).filter(DocumentModel.status == "completed").count()
    avg_confidence = db.query(DocumentModel).filter(DocumentModel.status == "completed").all()

    if avg_confidence:
        avg_conf = sum(doc.confidence_score for doc in avg_confidence) / len(avg_confidence)
    else:
        avg_conf = 0.0

    return {
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "average_confidence": avg_conf,
        "success_rate": processed_docs / total_docs if total_docs > 0 else 0.0
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "web_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
