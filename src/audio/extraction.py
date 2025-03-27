import os
import ffmpeg
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor

def extract_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Извлечение аудио из медиа-файла напрямую в память.
    
    Args:
        input_media: Путь к медиа-файлу
        output_path: Путь для сохранения аудио (опционально)
        
    Returns:
        Tuple[Tensor, int]: (аудио тензор, частота дискретизации)
    """
    try:
        # Читаем аудио напрямую в numpy массив
        audio_data, _ = (
            ffmpeg
            .input(input_media)
            .output('pipe:', 
                   format='f32le',
                   acodec='pcm_f32le',
                   ac=1,
                   ar='16k')
            .run(capture_stdout=True, quiet=True)
        )
        
        # Преобразуем в тензор PyTorch
        audio_array = np.frombuffer(audio_data, np.float32)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        
        # Сохраняем результат только если указан output_path
        if output_path:
            # Создаем директорию для выходного файла, если её нет
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Сохраняем в файл
            torch.save(audio_tensor, output_path)
        
        return audio_tensor, 16000  # Фиксированная частота дискретизации
        
    except ffmpeg.Error as e:
        raise RuntimeError(f"Ошибка FFmpeg при извлечении аудио: {e.stderr.decode()}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при извлечении аудио: {str(e)}") 