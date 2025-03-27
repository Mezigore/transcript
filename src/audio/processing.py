import os
import ffmpeg
from typing import Tuple, Optional
import torch
from torch import Tensor
import torchaudio
import numpy as np

def process_audio(input_audio: str, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Обработка аудио для улучшения качества распознавания речи.
    
    Args:
        input_audio: Путь к аудио-файлу
        output_path: Путь для сохранения обработанного аудио (опционально)
        
    Returns:
        Tuple[Tensor, int]: (аудио тензор, частота дискретизации)
    """
    try:
        # Проверяем существование входного файла
        if not os.path.exists(input_audio):
            raise FileNotFoundError(f"Аудио файл не найден: {input_audio}")
            
        # Загружаем аудио с помощью torchaudio
        waveform, sample_rate = torchaudio.load(input_audio)
        
        # Преобразуем в моно, если файл стерео
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ресэмплируем до 16кГц если нужно
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Нормализация
        normalized_tensor = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Сохраняем результат только если указан output_path
        if output_path:
            # Создаем директорию для выходного файла, если её нет
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Сохраняем в формате WAV
            try:
                torchaudio.save(
                    output_path,
                    normalized_tensor,
                    sample_rate,
                    format="wav"
                )
            except Exception as e:
                raise RuntimeError(f"Ошибка при сохранении аудио в файл: {str(e)}")
        
        return normalized_tensor, sample_rate
        
    except Exception as e:
        # В случае ошибки пытаемся вернуть хотя бы пустой тензор правильной формы
        print(f"Ошибка при обработке аудио: {str(e)}")
        
        # Попробуем напрямую извлечь аудио через ffmpeg
        try:
            audio_data, _ = (
                ffmpeg
                .input(input_audio)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k')
                .run(capture_stdout=True, quiet=True)
            )
            
            audio_array = np.frombuffer(audio_data, np.float32)
            audio_tensor = torch.from_numpy(audio_array.copy()).unsqueeze(0)
            return audio_tensor, 16000
        except:
            raise RuntimeError(f"Не удалось обработать аудио: {str(e)}")

def extract_and_process_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Извлечение и обработка аудио из медиа-файла."""
    from .extraction import extract_audio
    
    # Генерируем временный путь для извлеченного файла
    temp_audio_path = output_path
    
    # Извлекаем аудио
    audio_tensor, sample_rate, _ = extract_audio(input_media, temp_audio_path)
    
    # Проверяем, был ли создан файл
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            # Обрабатываем извлеченное аудио
            return process_audio(temp_audio_path)
        except Exception as e:
            print(f"Предупреждение: обработка аудио не удалась: {str(e)}")
            # Возвращаем оригинальный тензор
            return audio_tensor, sample_rate
    else:
        # Если файл не был создан, просто возвращаем тензор
        return audio_tensor, sample_rate