import os
import argparse
import subprocess
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

def process_media(media_path: str, hf_token: Optional[str] = None, emotion_engine: str = "wav2vec", skip_emotions: bool = False) -> ProcessingResult:
    processor = AudioProcessingPipeline(hf_token or API_KEYS['huggingface'])
    return processor.process(media_path, skip_emotion_analysis=skip_emotions, emotion_engine=emotion_engine)

def post_process_transcripts(no_timestamps: bool = False):
    """Запускает скрипт постобработки транскриптов"""
    try:
        print("[INFO] Запуск постобработки транскриптов...")
        cmd = ["python3", "process_all_transcripts.py"]
        if no_timestamps:
            cmd.append("--no-timestamps")
        subprocess.run(cmd, check=True)
        print("[INFO] Постобработка завершена")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ОШИБКА] Ошибка при постобработке: {e}")
        return False

def handle_single_file(emotion_engine: str, skip_emotions: bool, post_process: bool, no_timestamps: bool = False):
    media_path = select_media_file()
    if not media_path:
        print("[ОШИБКА] Файл не выбран")
        return False

    if check_if_processed(media_path):
        if input("[ВВОД] Файл уже обработан. Повторить? (y/n): ").lower() != 'y':
            print("[INFO] Отмена обработки")
            return True

    result = process_media(media_path, emotion_engine=emotion_engine, skip_emotions=skip_emotions)
    print_processing_result(result, os.path.basename(media_path))
    
    if post_process and result.success:
        post_process_transcripts(no_timestamps)
        
    return result.success

def handle_batch_processing(emotion_engine: str, skip_emotions: bool, post_process: bool, no_timestamps: bool = False):
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
        result = process_media(full_path, emotion_engine=emotion_engine, skip_emotions=skip_emotions)
        if result.success:
            stats['processed'] += 1
        else:
            stats['errors'] += 1
        print_processing_result(result, file)
    
    print_batch_statistics(len(files), **stats)
    
    if post_process and stats['processed'] > 0:
        post_process_transcripts(no_timestamps)
        
    return stats['errors'] == 0

def parse_args():
    parser = argparse.ArgumentParser(description='Обработка аудио и видео файлов с транскрипцией и анализом эмоций')
    parser.add_argument('--emotion-engine', choices=['wavlm', 'wav2vec'], default='wav2vec',
                       help='Выбор движка анализа эмоций: wavlm или wav2vec (по умолчанию)')
    parser.add_argument('--skip-emotions', action='store_true',
                       help='Пропустить анализ эмоций')
    parser.add_argument('--post-process', action='store_true',
                       help='Запустить постобработку транскриптов после завершения')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Удалить временные метки из результатов постобработки (не влияет на основной транскрипт)')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        mode = select_operation_mode()
        success = handle_single_file(args.emotion_engine, args.skip_emotions, args.post_process, args.no_timestamps) if mode == 1 else handle_batch_processing(args.emotion_engine, args.skip_emotions, args.post_process, args.no_timestamps)
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
        cleanup_temp_files()
        exit(0)

if __name__ == "__main__":
    main()
