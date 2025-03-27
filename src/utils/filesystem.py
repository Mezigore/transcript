import os
import shutil
from typing import List
from config import FILES, INPUT_DIR, OUTPUT_DIR, TEMP_DIR

def find_media_files() -> List[str]:
    """Находит все медиа-файлы в директории INPUT_DIR"""
    video_extensions = FILES['video_extensions']
    audio_extensions = FILES['audio_extensions']
    media_extensions = video_extensions + audio_extensions
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    
    return [f for f in os.listdir(INPUT_DIR) 
            if os.path.isfile(os.path.join(INPUT_DIR, f)) and
            any(f.lower().endswith(ext) for ext in media_extensions)]

def check_if_processed(media_path: str) -> bool:
    """Проверяет, был ли файл уже обработан"""
    base_output = os.path.splitext(os.path.basename(media_path))[0]
    output_conversation = f"{base_output}_transcript.txt"
    return os.path.exists(os.path.join(OUTPUT_DIR, output_conversation))

def get_output_filename(media_path: str) -> str:
    """Генерирует имя выходного файла"""
    base = os.path.splitext(os.path.basename(media_path))[0]
    return f"{base}_transcript.txt"

def ensure_directories():
    """Создает необходимые директории, если они не существуют"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files():
    """Удаляет временные файлы"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def save_transcript_to_file(transcript_text: str, output_file: str) -> str:
    """Сохраняет текст транскрипции в файл"""
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_file))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    
    return output_path 