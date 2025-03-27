import os
import numpy as np
import torch
from torch import Tensor
import torchaudio
from typing import List, Tuple, Dict

def segment_audio(audio_tensor: Tensor, sample_rate: int, segment_length_sec: int = 30) -> List[Tensor]:
    """Разделение аудио на сегменты указанной длины.
    
    Args:
        audio_tensor: Аудио в виде тензора
        sample_rate: Частота дискретизации
        segment_length_sec: Длина сегмента в секундах
        
    Returns:
        List[Tensor]: Список аудио-сегментов
    """
    # Преобразуем тензор в numpy массив для упрощения работы
    audio_np = audio_tensor.squeeze().numpy()
    
    # Рассчитываем количество сегментов
    total_samples = len(audio_np)
    samples_per_segment = sample_rate * segment_length_sec
    num_segments = int(np.ceil(total_samples / samples_per_segment))
    
    segments = []
    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = min((i + 1) * samples_per_segment, total_samples)
        
        segment_np = audio_np[start_sample:end_sample]
        segment_tensor = torch.tensor(segment_np).unsqueeze(0)  # Добавляем размерность канала
        segments.append(segment_tensor)
    
    return segments

def vad_filter(input_tensor: Tensor, sample_rate: int, aggressiveness: int = 3) -> Tensor:
    """Фильтрация с использованием Voice Activity Detection (VAD)
    для удаления тишины и фоновых шумов.
    
    Перенесено из pre_processing.py и адаптировано для работы с тензорами.
    
    Args:
        input_tensor: Аудио-тензор
        sample_rate: Частота дискретизации
        aggressiveness: Агрессивность VAD (0-3)
        
    Returns:
        Tensor: Отфильтрованный аудио-тензор
    """
    import webrtcvad
    import struct
    
    vad = webrtcvad.Vad(aggressiveness)
    
    # Преобразуем тензор в 16-битный PCM
    audio_np = input_tensor.squeeze().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # Параметры для обработки
    frame_duration_ms = 30  # длина фрейма в миллисекундах
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    # Разделяем на фреймы и проверяем каждый на наличие речи
    frames = []
    for i in range(0, len(audio_int16), frame_size):
        frame = audio_int16[i:i+frame_size]
        if len(frame) < frame_size:
            # Дополняем последний фрейм тишиной, если нужно
            frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
        
        frame_bytes = struct.pack('h' * len(frame), *frame)
        if vad.is_speech(frame_bytes, sample_rate):
            frames.append(frame)
    
    if not frames:
        return input_tensor  # Возвращаем исходный тензор, если речь не обнаружена
    
    # Собираем финальное аудио только с речью
    combined = np.concatenate(frames)
    filtered_tensor = torch.tensor(combined / 32767.0).unsqueeze(0)  # Нормализуем обратно и добавляем размерность канала
    
    return filtered_tensor 