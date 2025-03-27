import os
import ffmpeg
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
import torchaudio

def extract_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int, Optional[str]]:
    """Извлечение аудио из медиа-файла напрямую в память.
    
    Args:
        input_media: Путь к медиа-файлу
        output_path: Путь для сохранения аудио (опционально)
        
    Returns:
        Tuple[Tensor, int]: (аудио тензор, частота дискретизации)
    """
    try:
        print(f"Извлечение аудио из файла: {input_media}")
        if not os.path.exists(input_media):
            raise FileNotFoundError(f"Медиа-файл не найден: {input_media}")
            
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
        print(f"FFmpeg успешно извлек аудио из: {input_media}")
        
        # Преобразуем в тензор PyTorch
        audio_array = np.frombuffer(audio_data, np.float32)
        audio_tensor = torch.from_numpy(audio_array.copy()).unsqueeze(0)
        
        # Сохраняем результат только если указан output_path
        if output_path:
            # Создаем директорию для выходного файла, если её нет
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Сохраняем в формате WAV, а не как тензор PyTorch
            try:
                torchaudio.save(
                    output_path,
                    audio_tensor,
                    16000,
                    format="wav"
                )
            except Exception as e:
                raise RuntimeError(f"Ошибка при сохранении аудио в файл: {str(e)}")
        
        return audio_tensor, 16000, output_path
        
    except ffmpeg.Error as e:
        raise RuntimeError(f"Ошибка FFmpeg при извлечении аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
    except Exception as e:
        print(f"Ошибка при извлечении аудио: {str(e)}")
        raise RuntimeError(f"Ошибка при извлечении аудио: {str(e)}") 