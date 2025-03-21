from pyannote.audio import Pipeline
from tqdm import tqdm
import ffmpeg
import os
import tempfile
import numpy as np
from config import DIARIZATION, API_KEYS


# Функция для диаризации с помощью Pyannote
def diarize(audio_path, hf_token=None):
    # Если токен не передан, используем из конфигурации
    if hf_token is None:
        hf_token = API_KEYS['huggingface']
        
    print("[INFO] Начало диаризации аудио...")
    
    # Получаем информацию о длительности аудио файла
    probe = ffmpeg.probe(audio_path)
    duration = float(probe['format']['duration'])
    print(f"[INFO] Длительность аудио: {duration:.2f} сек")
    
    # Размер сегмента в секундах
    segment_size = DIARIZATION['segment_size']
    
    # Вычисляем количество сегментов
    num_segments = int(np.ceil(duration / segment_size))
    
    # Создаем временную директорию для сегментов
    temp_dir = tempfile.mkdtemp()
    
    # Разбиваем аудио на сегменты
    print("[INFO] Разбиение аудио на сегменты...")
    segment_files = []
    
    for i in tqdm(range(num_segments), desc="Подготовка сегментов", unit="сегмент"):
        start_time = i * segment_size
        # Для последнего сегмента берем оставшуюся длительность
        if i == num_segments - 1:
            segment_duration = duration - start_time
        else:
            segment_duration = segment_size
        
        segment_file = os.path.join(temp_dir, f"diar_segment_{i}.wav")
        
        try:
            (
                ffmpeg
                .input(audio_path)
                .output(segment_file, ss=start_time, t=segment_duration, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            segment_files.append((segment_file, start_time))
        except Exception as e:
            print(f"[ОШИБКА] Не удалось создать сегмент {i}: {e}")
    
    # Загружаем модель диаризации один раз
    print("[INFO] Загрузка модели диаризации...")
    pipeline = Pipeline.from_pretrained(DIARIZATION['model_path'], use_auth_token=hf_token)
    
    # Диаризация каждого сегмента
    print("[INFO] Выполнение диаризации...")
    all_speaker_segments = []
    
    for segment_file, start_time in tqdm(segment_files, desc="Диаризация", unit="сегмент"):
        try:
            # Диаризация сегмента
            with tqdm(total=0, desc=f"Обработка сегмента {os.path.basename(segment_file)}", 
                     bar_format='{desc}: {elapsed}', position=1, leave=False) as pbar:
                diarization = pipeline(segment_file, 
                                      min_speakers=DIARIZATION['min_speakers'], 
                                      max_speakers=DIARIZATION['max_speakers'])
                pbar.set_description(f"Сегмент {os.path.basename(segment_file)} обработан")
            
            # Преобразование результатов диаризации в удобный формат
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Корректируем временные метки с учетом начала сегмента
                all_speaker_segments.append({
                    "start": turn.start + start_time,
                    "end": turn.end + start_time,
                    "speaker": speaker
                })
            
        except Exception as e:
            print(f"[ОШИБКА] Не удалось выполнить диаризацию сегмента {segment_file}: {e}")
    
    # Сортируем сегменты по времени начала
    all_speaker_segments.sort(key=lambda x: x["start"])
    
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
    
    print(f"[INFO] Найдено {len(all_speaker_segments)} сегментов с {len(set([s['speaker'] for s in all_speaker_segments]))} спикерами")
    return all_speaker_segments
