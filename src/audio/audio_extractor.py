import os
import ffmpeg
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
import torchaudio

def extract_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int, Optional[str]]:
    """Извлечение аудио из медиа-файла напрямую в память без обработки.
    
    Args:
        input_media: Путь к медиа-файлу
        output_path: Путь для сохранения аудио (опционально)
        
    Returns:
        Tuple[Tensor, int, Optional[str]]: (аудио тензор, частота дискретизации, путь к сохраненному файлу)
    """
    try:
        print(f"Извлечение аудио из файла: {input_media}")
        if not os.path.exists(input_media):
            raise FileNotFoundError(f"Медиа-файл не найден: {input_media}")
            
        # Извлекаем аудио без обработки
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
        audio_tensor = torch.from_numpy(audio_array.copy()).unsqueeze(0)
        
        # Сохраняем если нужно
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            torchaudio.save(
                output_path,
                audio_tensor,
                16000,
                format="wav"
            )
        
        return audio_tensor, 16000, output_path
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        raise RuntimeError(f"Ошибка FFmpeg при извлечении аудио: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при извлечении аудио: {str(e)}") 