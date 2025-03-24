import torch
import torchaudio
import numpy as np
from typing import List, Tuple

# Загрузка SileroVAD
def get_vad_model():
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, save_audio, read_audio, 
         VADIterator, collect_chunks) = utils
        return model, get_speech_timestamps, collect_chunks
    except Exception as e:
        print(f"[ERROR] Failed to load SileroVAD model: {str(e)}")
        raise

# Функция для нарезки аудио на фрагменты с речью
def chunking_audio(audio_path: str, min_segment_length: float = 5.0, max_segment_length: float = 30.0) -> List[Tuple[np.ndarray, float]]:
    """
    Разделяет аудиофайл на сегменты с речью, длиной от min_segment_length до max_segment_length секунд.
    
    Args:
        audio_path: Путь к аудиофайлу
        min_segment_length: Минимальная длина сегмента в секундах (5 секунд)
        max_segment_length: Максимальная длина сегмента в секундах (30 секунд)
        
    Returns:
        List[Tuple[np.ndarray, float]]: Список кортежей (аудио_данные, начальное_время)
    """
    # Загружаем модель VAD
    model, get_speech_timestamps, collect_chunks = get_vad_model()
    
    # Загружаем аудио
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Приводим к моно, если нужно
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Приводим к float32
    waveform = waveform.float()
    
    # Получаем временные метки речи
    speech_timestamps = get_speech_timestamps(
        waveform[0], 
        model, 
        sampling_rate=sample_rate,
        min_speech_duration_ms=250,
        max_speech_duration_s=float('inf'),
        speech_pad_ms=300
    )
    
    # Собираем сегменты речи
    speech_chunks = collect_chunks(speech_timestamps, waveform)
    
    # Создаем список кортежей (аудио_данные, начальное_время)
    segments = []
    for chunk, ts in zip(speech_chunks, speech_timestamps):
        audio_data = chunk.numpy()
        start_time = ts['start'] / 1000  # Преобразуем из миллисекунд в секунды
        duration = audio_data.shape[0] / sample_rate
        
        # Если длительность больше максимальной, разделяем на подсегменты
        if duration > max_segment_length:
            samples_per_segment = int(max_segment_length * sample_rate)
            num_segments = int(np.ceil(audio_data.shape[0] / samples_per_segment))
            
            for i in range(num_segments):
                segment_start = i * samples_per_segment
                segment_end = min((i + 1) * samples_per_segment, audio_data.shape[0])
                segment_data = audio_data[segment_start:segment_end]
                
                # Добавляем только если длина сегмента достаточная
                if segment_data.shape[0] / sample_rate >= min_segment_length:
                    segment_start_time = start_time + (segment_start / sample_rate)
                    segments.append((segment_data, segment_start_time))
        else:
            # Если длительность в пределах нормы, добавляем как есть
            if duration >= min_segment_length:
                segments.append((audio_data, start_time))
    
    return segments, sample_rate
