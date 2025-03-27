import os
import ffmpeg
from typing import Optional, Tuple
from tqdm import tqdm
from torch import Tensor
import torchaudio
import numpy as np
import torch
from config import TEMP_DIR

def extract_and_process_audio(input_media: str) -> Tuple[Optional[Tensor], Optional[int]]:
    """Извлекает аудио из медиа-файла и предобрабатывает его напрямую в памяти"""
    file_ext = os.path.splitext(input_media)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    is_audio_file = file_ext in audio_extensions
    
    if is_audio_file:
        print(f"[INFO] Обработка аудиофайла {input_media}...")
    else:
        print(f"[INFO] Извлечение аудио из {input_media}...")
    
    with tqdm(total=0, desc="Обработка медиа", bar_format='{desc}: {elapsed}') as pbar:
        try:
            # Читаем аудио напрямую в numpy массив
            stream = ffmpeg.input(input_media)
            audio_data, _ = (
                ffmpeg
                .input(input_media)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k',
                       af='silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=1:start_duration=0:stop_duration=3:detection=peak,highpass=200,lowpass=3000,afftdn,volume=12dB,dynaudnorm')
                .run(capture_stdout=True, quiet=True)
            )
            
            # Преобразуем в тензор PyTorch
            audio_array = np.frombuffer(audio_data, np.float32)
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            pbar.set_description("Обработка аудио завершена")
            return audio_tensor, 16000  # Фиксированная частота дискретизации
            
        except ffmpeg.Error as e:
            pbar.set_description("Ошибка обработки медиа")
            print(f"[ОШИБКА] Не удалось обработать медиафайл: {e}")
            return None, None
        except Exception as e:
            pbar.set_description("Ошибка обработки медиа")
            print(f"[ОШИБКА] Не удалось обработать медиафайл: {e}")
            return None, None 