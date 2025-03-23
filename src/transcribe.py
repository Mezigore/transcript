from config import TRANSCRIPTION, API_KEYS, TEMP_DIR
from src.chunking import chunking_audio
from tqdm import tqdm
import ffmpeg
import numpy as np
import mlx_whisper
import os


# Функция для транскрипции с помощью Whisper
def transcribe(audio_path):
    print("[INFO] Начало транскрипции аудио...")
    
    # Получаем информацию о длительности аудио файла
    probe = ffmpeg.probe(audio_path)
    duration = float(probe['format']['duration'])
    print(f"[INFO] Длительность аудио: {duration:.2f} сек")
    
    # Размер сегмента в секундах
    segment_size = TRANSCRIPTION['segment_size']
      
    # Разбиваем аудио на сегменты
    print("[INFO] Разбиение аудио на сегменты...")
    segment_files, sample_rate = chunking_audio(audio_path, segment_size)
    
    
    # Транскрибируем каждый сегмент
    print("[INFO] Выполнение транскрипции...")
    all_segments = []
    
    prompt = TRANSCRIPTION['initial_prompt']
    
    for segment_file, start_time in tqdm(segment_files, desc="Транскрипция", unit="сегмент"):
        try:
            # Транскрибируем сегмент
            result = mlx_whisper.transcribe(
                segment_file, 
                path_or_hf_repo=TRANSCRIPTION['model_path'], 
                initial_prompt=prompt,
                condition_on_previous_text=True,
                **TRANSCRIPTION['decode_options']
            )
            prompt = result["text"]
            
            # Корректируем временные метки с учетом начала сегмента
            for segment in result["segments"]:
                segment["start"] += start_time
                segment["end"] += start_time
                all_segments.append(segment)
            
        except Exception as e:
            print(f"[ОШИБКА] Не удалось транскрибировать сегмент {segment_file}: {e}")
    
    # Сортируем сегменты по времени начала
    all_segments.sort(key=lambda x: x["start"])
    
    # Создаем результат в формате, совместимом с оригинальным выводом
    result = {
        "language": "ru",  # Предполагаем русский язык
        "segments": all_segments
    }
    
    # Очищаем временные файлы
    print("[INFO] Очистка временных файлов...")
    for segment_file, _ in segment_files:
        try:
            os.remove(segment_file)
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить файл {segment_file}: {e}")
    
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить временную директорию: {e}")
    
    print(f"[INFO] Язык транскрипции: {result['language']}")
    
    print("[INFO] Результат транскрипции:")
    # Добавляем прогресс-бар для вывода сегментов транскрипции
    for segment in tqdm(result["segments"], desc="Обработка сегментов", unit="сегмент"):
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
    
    return result["language"], result
