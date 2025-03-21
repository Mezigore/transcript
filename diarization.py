from pyannote.audio import Pipeline
from tqdm import tqdm
import ffmpeg
import os
import tempfile
import numpy as np


# Функция для диаризации с помощью Pyannote
def diarize(audio_path, hf_token):
    print("Запуск диаризации с помощью Pyannote...")
    
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
    print("Разбиение аудио на сегменты для диаризации...")
    segment_files = []
    
    for i in tqdm(range(num_segments), desc="Подготовка сегментов аудио для диаризации", unit="сегмент"):
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
            print(f"Ошибка при создании сегмента {i}: {e}")
    
    # Загружаем модель диаризации один раз
    print("Загрузка модели диаризации...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    
    # Диаризация каждого сегмента
    print("Диаризация сегментов...")
    all_speaker_segments = []
    
    for segment_file, start_time in tqdm(segment_files, desc="Диаризация сегментов", unit="сегмент"):
        try:
            # Диаризация сегмента
            with tqdm(total=0, desc=f"Диаризация сегмента {os.path.basename(segment_file)}", 
                     bar_format='{desc}: {elapsed}', position=1, leave=False) as pbar:
                diarization = pipeline(segment_file, min_speakers=2, max_speakers=3)
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
            print(f"Ошибка при диаризации сегмента {segment_file}: {e}")
    
    # Сортируем сегменты по времени начала
    all_speaker_segments.sort(key=lambda x: x["start"])
    
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
    
    print(f"\nВсего найдено {len(all_speaker_segments)} сегментов с {len(set([s['speaker'] for s in all_speaker_segments]))} спикерами")
    return all_speaker_segments
