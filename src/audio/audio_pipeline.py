import os
from typing import Optional, Tuple
from tqdm import tqdm
from torch import Tensor
import torch
from config import TEMP_DIR

def extract_and_process_audio(input_media: str, 
                             output_path: Optional[str] = None,
                             temp_dir: str = TEMP_DIR) -> Tuple[Optional[Tensor], Optional[int]]:
    """Извлекает аудио из медиа-файла и предобрабатывает его с оптимизацией для Zoom-звонков.
    
    Args:
        input_media: Путь к медиа-файлу
        output_path: Путь для сохранения обработанного аудио (опционально)
        temp_dir: Директория для временных файлов
    
    Returns:
        Tuple[Optional[Tensor], Optional[int]]: (аудио тензор, частота дискретизации)
    """
    from .audio_extractor import extract_audio
    from .audio_processor import process_audio
    
    file_ext = os.path.splitext(input_media)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    is_audio_file = file_ext in audio_extensions
    
    if is_audio_file:
        print(f"[INFO] Обработка аудиофайла {input_media}...")
    else:
        print(f"[INFO] Извлечение аудио из {input_media}...")
    
    # Добавляем 2 этапа обработки с визуальным индикатором прогресса
    with tqdm(total=2, desc="Обработка медиа", bar_format='{desc}: |{bar}| {percentage:3.0f}% {elapsed}') as pbar:
        try:
            # Шаг 1: Извлечение аудио
            pbar.set_description("⏳ Извлечение аудио")
            audio_tensor, sample_rate, _ = extract_audio(input_media)
            pbar.update(1)  # Увеличиваем счетчик после извлечения
            
            # Шаг 2: Обработка аудио с оптимизированными фильтрами
            pbar.set_description("⚙️ Обработка аудио")
            processed_tensor, processed_rate = process_audio(audio_tensor, sample_rate, output_path)
            
            pbar.update(1)  # Увеличиваем счетчик после обработки
            pbar.set_description("✅ Обработка аудио завершена")
            return processed_tensor, processed_rate
            
        except Exception as e:
            pbar.set_description("❌ Ошибка обработки медиа")
            print(f"[ОШИБКА] Не удалось обработать медиафайл: {e}")
            return None, None 