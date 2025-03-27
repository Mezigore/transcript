import os
from typing import Optional
from config import API_KEYS, INPUT_DIR
from src.utils.filesystem import find_media_files, check_if_processed, cleanup_temp_files
from src.cli.cli import (
    select_operation_mode,
    select_media_file,
    print_processing_result,
    print_batch_statistics
)
from src.pipelines.audio_processing_pipeline import AudioProcessingPipeline, ProcessingResult

def format_time(seconds: float) -> str:
    """Форматирует время в минуты и секунды."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}м {seconds}с"

def process_media(media_path: str, hf_token: Optional[str] = None) -> ProcessingResult:
    processor = AudioProcessingPipeline(hf_token or API_KEYS['huggingface'])
    return processor.process(media_path)

def handle_single_file():
    media_path = select_media_file()
    if not media_path:
        print("[ОШИБКА] Файл не выбран")
        return False

    if check_if_processed(media_path):
        if input("[ВВОД] Файл уже обработан. Повторить? (y/n): ").lower() != 'y':
            print("[INFO] Отмена обработки")
            return True

    result = process_media(media_path)
    print_processing_result(result, os.path.basename(media_path))
    return result.success

def handle_batch_processing():
    files = find_media_files()
    if not files:
        print("[ОШИБКА] Файлы не найдены")
        return False

    print("[INFO] Найдены файлы:")
    for i, f in enumerate(files, 1):
        print(f" {i}. {f} {'[обработан]' if check_if_processed(f) else ''}")

    skip_processed = input("[ВВОД] Пропустить обработанные? (y/n): ").lower() != 'n'
    
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    for file in files:
        full_path = os.path.join(INPUT_DIR, file)
        if skip_processed and check_if_processed(full_path):
            stats['skipped'] += 1
            continue
            
        print(f"\n[INFO] Обработка {file}...")
        result = process_media(full_path)
        if result.success:
            stats['processed'] += 1
        else:
            stats['errors'] += 1
        print_processing_result(result, file)
    
    print_batch_statistics(len(files), **stats)
    return stats['errors'] == 0

def main():
    try:
        mode = select_operation_mode()
        success = handle_single_file() if mode == 1 else handle_batch_processing()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
        cleanup_temp_files()
        exit(0)

if __name__ == "__main__":
    main()
