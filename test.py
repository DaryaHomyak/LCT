from integrated_archive_processor import ArchiveDocumentProcessor

# Создание процессора
processor = ArchiveDocumentProcessor(
    ocr_engine='paddle',  # tesseract, paddle, easy
    use_postprocessing=True
)

# Обработка документа
results = processor.process_document('00000084.jpg', 'output_folder')

# Сохранение результатов
processor.save_results(results, 'results.txt')