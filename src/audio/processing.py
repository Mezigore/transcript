import os
import ffmpeg
from typing import Tuple, Optional
import torch
from torch import Tensor
import torchaudio
import numpy as np

def process_audio(input_audio: str , output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Обработка аудио для улучшения качества распознавания речи.
    
    Применяет: удаление тишины, фильтрацию, нормализацию и другие улучшения.
    
    Args:
        input_audio: Путь к аудио-файлу
        output_path: Путь для сохранения обработанного аудио (опционально)
        
    Returns:
        Tuple[Tensor, int]: (аудио тензор, частота дискретизации)
    """
    try:
        # Читаем аудио напрямую в numpy массив
        audio_data, _ = (
                ffmpeg
                .input(input_audio)
                .output('pipe:', 
                    format='f32le',
                    acodec='pcm_f32le',
                    ac=1,
                    ar='16k',
                    af='silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=1:start_duration=0:stop_duration=3:detection=peak,highpass=200,lowpass=3000,afftdn,volume=12dB,dynaudnorm')
                .run(capture_stdout=True, quiet=True)
                )
        
        
        # Преобразуем в тензор PyTorch
        audio_array = np.frombuffer(audio_data, np.float32)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        
        # Нормализация
        normalized_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
        
        # Сохраняем результат только если указан output_path
        if output_path:
            # Создаем директорию для выходного файла, если её нет
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Сохраняем в файл
            torch.save(normalized_tensor, output_path)
        
        return normalized_tensor, 16000  # Фиксированная частота дискретизации
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке аудио: {str(e)}")

def extract_and_process_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Извлечение и обработка аудио из медиа-файла.
    
    Объединяет функции извлечения и обработки аудио в один удобный метод.
    
    Args:
        input_media: Путь к медиа-файлу
        
    Returns:
        Tuple[Tensor, int]: (аудио тензор, частота дискретизации)
    """
    from .extraction import extract_audio
    
    _, _, output_path = extract_audio(input_media, output_path)
    if output_path:
        return process_audio(output_path) 
    else:
        raise RuntimeError("Не удалось получить путь к извлеченному аудио")