import sys
from typing import List, Any, Dict
from torch import Tensor
import torch
from config import TRANSCRIPTION

def transcribe_audio(audio_tensor: Tensor, sample_rate: int) -> List[dict[str, Any]]:
    print("[INFO] Начало транскрипции аудио...")

    try:
        if sys.platform == "darwin":  # macOS
            try:
                import mlx_whisper
            except ImportError:
                raise ImportError("Пакет mlx_whisper необходим для macOS, но не установлен")
            
            try:
                result: Dict[str, Any] = mlx_whisper.transcribe(
                    audio_tensor.numpy()[0],
                    path_or_hf_repo=TRANSCRIPTION['model_path_mlx'],
                    initial_prompt=TRANSCRIPTION['initial_prompt'],
                    verbose=True,
                    condition_on_previous_text=True,
                    word_timestamps=True,
                    hallucination_silence_threshold=2.0,
                    compression_ratio_threshold=2.0,
                    logprob_threshold=-0.8,
                    no_speech_threshold=1.0,
                    temperature=(0.0, 0.2, 0.4),
                    **{'language': 'ru'}
                )
            except Exception as e:
                raise RuntimeError(f"Ошибка транскрипции MLX Whisper: {str(e)}")
        else:
            try:
                import whisper
            except ImportError:
                raise ImportError("Пакет whisper необходим, но не установлен")
            
            try:
                model = whisper.load_model(TRANSCRIPTION['model'], device='cuda' if torch.cuda.is_available() else 'cpu')
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить модель Whisper: {str(e)}")
            
            try:
                result: Dict[str, Any] = model.transcribe(
                    audio_tensor[0],
                    verbose=True,
                    initial_prompt=TRANSCRIPTION['initial_prompt'],
                    condition_on_previous_text=True,
                    word_timestamps=True,
                    hallucination_silence_threshold=2.0,
                    **TRANSCRIPTION['decode_options']
                )
            except Exception as e:
                raise RuntimeError(f"Ошибка транскрипции Whisper: {str(e)}")
    except Exception as e:
        print(f"[ОШИБКА] Транскрипция не удалась: {str(e)}")
        raise
    
    # print(f"[INFO] Транскрипция завершена")
    # print(f"[INFO] {len(result['segments'])} сегментов найдено")
    # print(f"[INFO] Первый сегмент: {result['segments'][0]}")

    # Преобразуем результаты в удобный формат
    return [{
        'start': float(segment['start']),
        'end': float(segment['end']),
        'text': segment['text']
    } for segment in result['segments']]