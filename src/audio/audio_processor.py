import os
import ffmpeg
from typing import Tuple, Optional
import torch
from torch import Tensor
import torchaudio
import numpy as np

def process_audio(input_audio: str | Tensor, sample_rate: int = 16000, 
                  output_path: Optional[str] = None, normalize_to_lufs: Optional[float] = None) -> Tuple[Tensor, int]:
    """Обработка аудио с оптимизацией для Zoom-звонков.
    
    Args:
        input_audio: Путь к аудио-файлу или аудио тензор
        sample_rate: Частота дискретизации входного тензора (если передан тензор)
        output_path: Путь для сохранения обработанного аудио (опционально)
        normalize_to_lufs: Если указано, нормализует аудио до заданного уровня LUFS после обработки
        
    Returns:
        Tuple[Tensor, int]: (обработанный аудио тензор, частота дискретизации)
    """
    try:
        if isinstance(input_audio, str):
            # Проверяем существование входного файла
            if not os.path.exists(input_audio):
                raise FileNotFoundError(f"Аудио файл не найден: {input_audio}")
            
            # Применяем оптимизированные FFmpeg-фильтры для Zoom-звонков
            audio_data, _ = (
                ffmpeg
                .input(input_audio)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k',
                       af=',highpass=150,lowpass=7500,acompressor=threshold=-50dB:ratio=15:attack=50:release=1000,silenceremove=start_periods=1:start_threshold=-40dB:stop_threshold=-45dB:start_silence=0.5:stop_duration=2:detection=rms,afftdn=nr=10:nf=-25:tn=1,volume=150dB,arnndn,dynaudnorm=f=100:g=5:p=0.95')
                .run(capture_stdout=True, quiet=True)
            )
            
            # Преобразуем в тензор PyTorch
            audio_array = np.frombuffer(audio_data, np.float32)
            processed_tensor = torch.from_numpy(audio_array.copy()).unsqueeze(0)
            output_sample_rate = 16000
            
        else:
            # Обрабатываем уже имеющийся тензор
            # Здесь можно добавить PyTorch-обработку, если нужно
            processed_tensor = input_audio
            output_sample_rate = sample_rate
        
        # Применяем LUFS нормализацию, если указано
        if normalize_to_lufs is not None:
            # Временный путь для промежуточного файла
            temp_path = None
            if output_path:
                temp_path = output_path + ".temp.wav"
            
            # Нормализуем до указанного уровня LUFS
            # processed_tensor, output_sample_rate = normalize_lufs(
            #     processed_tensor, 
            #     target_lufs=normalize_to_lufs,
            #     sample_rate=output_sample_rate,
            #     output_path=temp_path
            # )
            
            # Если был создан временный файл и его путь отличается от выходного, удаляем его
            if temp_path and output_path != temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Сохраняем результат
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            torchaudio.save(
                output_path,
                processed_tensor,
                output_sample_rate,
                format="wav"
            )
        
        return processed_tensor, output_sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке аудио: {str(e)}")

def extract_and_process_audio(input_media: str, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Извлечение и обработка аудио из медиа-файла."""
    from .audio_extractor import extract_audio
    
    # Генерируем временный путь для извлеченного файла
    temp_audio_path = output_path
    
    # Извлекаем аудио
    audio_tensor, sample_rate, _ = extract_audio(input_media, temp_audio_path)
    
    # Проверяем, был ли создан файл
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            # Обрабатываем извлеченное аудио
            return process_audio(temp_audio_path, normalize_to_lufs=-16)
        except Exception as e:
            print(f"Предупреждение: обработка аудио не удалась: {str(e)}")
            # Возвращаем оригинальный тензор
            return audio_tensor, sample_rate
    else:
        # Если файл не был создан, просто возвращаем тензор
        return audio_tensor, sample_rate 

def normalize_lufs(input_audio: str | Tensor, target_lufs: float = -16.0, 
                  sample_rate: int = 16000, output_path: Optional[str] = None) -> Tuple[Tensor, int]:
    """Нормализация аудио до заданного уровня LUFS, оптимизированного для голосового аудио.
    
    Args:
        input_audio: Путь к аудио-файлу или аудио тензор
        target_lufs: Целевой уровень LUFS (по умолчанию -16 LUFS, стандарт для голосового аудио)
        sample_rate: Частота дискретизации входного тензора (если передан тензор)
        output_path: Путь для сохранения обработанного аудио (опционально)
        
    Returns:
        Tuple[Tensor, int]: (нормализованный аудио тензор, частота дискретизации)
    """
    temp_file = None
    try:
        if isinstance(input_audio, Tensor):
            # Если входные данные - тензор, сохраняем во временный файл для обработки через ffmpeg
            temp_file = os.path.join(os.getcwd(), "temp_audio_for_lufs.wav")
            torchaudio.save(temp_file, input_audio, sample_rate, format="wav")
            audio_path = temp_file
        else:
            # Если входные данные - путь к файлу
            if not os.path.exists(input_audio):
                raise FileNotFoundError(f"Аудио файл не найден: {input_audio}")
            audio_path = input_audio
        
        # Формируем команду для loudnorm фильтра ffmpeg
        # I - целевая интегрированная громкость (LUFS)
        # LRA - целевой диапазон громкости (по умолчанию 7 для голоса)
        # TP - максимальный допустимый уровень True Peak (по умолчанию -1 dB)
        loudnorm_filter = f"loudnorm=I={target_lufs}:LRA=7:TP=-1"
        
        # Применяем нормализацию LUFS через ffmpeg
        audio_data, _ = (
            ffmpeg
            .input(audio_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k',
                   af=loudnorm_filter)
            .run(capture_stdout=True, quiet=True)
        )
        
        # Преобразуем в тензор PyTorch
        audio_array = np.frombuffer(audio_data, np.float32)
        normalized_tensor = torch.from_numpy(audio_array.copy()).unsqueeze(0)
        output_sample_rate = 16000
        
        # Удаляем временный файл, если он был создан
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Сохраняем результат, если указан путь
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            torchaudio.save(
                output_path,
                normalized_tensor,
                output_sample_rate,
                format="wav"
            )
        
        return normalized_tensor, output_sample_rate
        
    except Exception as e:
        # Удаляем временный файл, если он был создан
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"Ошибка при нормализации LUFS: {str(e)}") 