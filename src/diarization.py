from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm
import ffmpeg
import os
import tempfile
import numpy as np
import torchaudio
import torch
from config import DIARIZATION, API_KEYS
from src.chunking import chunking_audio


# Функция для диаризации с помощью Pyannote
def diarize(audio_path, hf_token=None):
    # Если токен не передан, используем из конфигурации
    if hf_token is None:
        hf_token = API_KEYS['huggingface']
        
    print("[INFO] Начало диаризации аудио...")
    
    # Получаем информацию о длительности аудио файла
    # probe = ffmpeg.probe(audio_path)
    # duration = float(probe['format']['duration'])
    # print(f"[INFO] Длительность аудио: {duration:.2f} сек")
    
    # Размер сегмента в секундах
    segment_size = DIARIZATION['segment_size']
    
    # Создаем временную директорию для сегментов
    temp_dir = tempfile.mkdtemp()
    
    # Разбиваем аудио на сегменты
    # print("[INFO] Разбиение аудио на сегменты...")
    # segment_files, sample_rate = chunking_audio(audio_path, max_segment_length=segment_size)
    
    # Загружаем модель диаризации один раз
    print("[INFO] Загрузка модели диаризации...")
    pipeline = Pipeline.from_pretrained(DIARIZATION['model_path'], use_auth_token=hf_token)
    pipeline.to(torch.device("mps"))
    
    # # Диаризация каждого сегмента
    # print("[INFO] Выполнение диаризации...")
    all_speaker_segments = []
    audio_file, sample_rate = torchaudio.load(audio_path)
    
    try:
        with ProgressHook() as hook:
            # Get the raw diarization results
            diarization_results = pipeline({"waveform": audio_file, "sample_rate": sample_rate}, 
                                          min_speakers=DIARIZATION['min_speakers'], 
                                          max_speakers=DIARIZATION['max_speakers'],
                                          num_speakers=DIARIZATION['num_speakers'],
                                          hook=hook,
                                      )
            
            # Process the results into a list of segments
            for turn, _, speaker in diarization_results.itertracks(yield_label=True):
                all_speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
    except Exception as e:
        print(f"[ОШИБКА] Не удалось выполнить диаризацию: {e}")
    
    print(f"[INFO] Найдено {len(all_speaker_segments)} сегментов с {len(set([s['speaker'] for s in all_speaker_segments]))} спикерами")
    return all_speaker_segments
