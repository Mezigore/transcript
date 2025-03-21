import os
import mlx_whisper
from files import *
from diarization import diarize
from emotion import analyze_emotions
from tqdm import tqdm
import ffmpeg
import numpy as np
import tempfile

# Функция для транскрипции с помощью Whisper
def transcribe(audio_path):
    print("Запуск транскрипции с помощью Whisper...")
    
    # Получаем информацию о длительности аудио файла
    probe = ffmpeg.probe(audio_path)
    duration = float(probe['format']['duration'])
    print(f"Длительность аудио: {duration:.2f} секунд")
    
    # Размер сегмента в секундах
    segment_size = 30.0
    
    # Вычисляем количество сегментов
    num_segments = int(np.ceil(duration / segment_size))
    
    # Создаем временную директорию для сегментов
    temp_dir = tempfile.mkdtemp()
    
    # Разбиваем аудио на сегменты
    print("Разбиение аудио на сегменты...")
    segment_files = []
    
    for i in tqdm(range(num_segments), desc="Подготовка сегментов аудио", unit="сегмент"):
        start_time = i * segment_size
        # Для последнего сегмента берем оставшуюся длительность
        if i == num_segments - 1:
            segment_duration = duration - start_time
        else:
            segment_duration = segment_size
        
        segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
        
        try:
            (
                ffmpeg
                .input(audio_path)
                .output(segment_file, ss=start_time, t=segment_duration, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            segment_files.append((segment_file, start_time))
        except Exception as e:
            print(f"Ошибка при создании сегмента {i}: {e}")
    
    # Транскрибируем каждый сегмент
    print("Транскрипция сегментов...")
    all_segments = []
    
    for segment_file, start_time in tqdm(segment_files, desc="Транскрипция сегментов", unit="сегмент"):
        try:
            # Транскрибируем сегмент
            result = mlx_whisper.transcribe(
                segment_file, 
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx", 
                initial_prompt="Запись интервью с представителем строительной сферы"
            )
            
            # Корректируем временные метки с учетом начала сегмента
            for segment in result["segments"]:
                segment["start"] += start_time
                segment["end"] += start_time
                all_segments.append(segment)
            
        except Exception as e:
            print(f"Ошибка при транскрипции сегмента {segment_file}: {e}")
    
    # Сортируем сегменты по времени начала
    all_segments.sort(key=lambda x: x["start"])
    
    # Создаем результат в формате, совместимом с оригинальным выводом
    result = {
        "language": "ru",  # Предполагаем русский язык
        "segments": all_segments
    }
    
    # Очищаем временные файлы
    print("Очистка временных файлов...")
    for segment_file, _ in segment_files:
        try:
            os.remove(segment_file)
        except Exception as e:
            print(f"Ошибка при удалении временного файла {segment_file}: {e}")
    
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Ошибка при удалении временной директории: {e}")
    
    print(f"Язык транскрипции: {result['language']}")
    
    print("Транскрипция:")
    # Добавляем прогресс-бар для вывода сегментов транскрипции
    for segment in tqdm(result["segments"], desc="Обработка сегментов транскрипции", unit="сегмент"):
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
    
    return result["language"], result

# Функция для объединения результатов транскрипции и диаризации
def merge_transcription_with_diarization(whisper_segments, speaker_segments_with_emotions):
    print("Объединение транскрипции с диаризацией и эмоциями...")
    merged_segments = []
    
    # Используем segments из результата Whisper
    for whisper_segment in whisper_segments["segments"]:
        # Находим спикера, который говорит в этом временном интервале
        # (используем наибольшее перекрытие)
        max_overlap = 0
        best_match = None
        
        whisper_start = float(whisper_segment["start"])
        whisper_end = float(whisper_segment["end"])
        
        for speaker_segment in speaker_segments_with_emotions:
            # Вычисляем перекрытие
            overlap_start = max(whisper_start, float(speaker_segment["start"]))
            overlap_end = min(whisper_end, float(speaker_segment["end"]))
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = speaker_segment
        
        if best_match:
            merged_segments.append({
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"],
                "speaker": best_match["speaker"],
                "emotion": best_match.get("emotion", "unknown"),
                "emotion_confidence": best_match.get("emotion_confidence", 0.0)
            })
        else:
            merged_segments.append({
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"],
                "speaker": "unknown",
                "emotion": "unknown",
                "emotion_confidence": 0.0
            })
    
    return merged_segments



# Основная функция
def process_video(video_path, hf_token):
    # Извлечение аудио
    audio_result = extract_and_process_audio(video_path)
    audio_path = audio_result
    
    if audio_path is None:
        return "Ошибка при извлечении аудио"
    
    # Транскрипция
    language, whisper_segments = transcribe(audio_path)
    
    # Диаризация
    speaker_segments = diarize(audio_path, hf_token)
    
    # Анализ эмоций
    speaker_segments_with_emotions = analyze_emotions(audio_path, speaker_segments)
    
    # Объединение результатов
    merged_segments = merge_transcription_with_diarization(whisper_segments, speaker_segments_with_emotions)
    
    # Базовое имя для выходных файлов
    base_output = os.path.splitext(video_path)[0]
    output_csv = f"{base_output}_transcript.csv"
    output_conversation = f"{base_output}_transcript.txt"
    
    # Создание CSV файла (для совместимости)
    csv_file = create_csv_file(merged_segments, output_csv)
    
    # Создание файла в формате разговора (основной формат для LLM)
    conversation_file = create_conversation_file(merged_segments, output_conversation)
    
    # Очистка временных файлов
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return output_conversation

if __name__ == "__main__":
    # Загрузка токена
    hf_token = "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"
    
    # Выбор файла
    video_path = select_mp4_file()
    if not video_path:
        exit(1)
    
    # Обработка видео
    result = process_video(video_path, hf_token)
    if result:
        print(f"Обработка завершена. Результат: {result}")
    else:
        print("Ошибка при обработке видео.")
