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
                    verbose=False,
                    # temperature=(0.0, 0.2, 0.5),
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    hallucination_silence_threshold=2.0,
                    no_speech_threshold=1.0,
                    compression_ratio_threshold=None,
                    logprob_threshold=None,
                    language="ru"
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
    
    print(f"[DEBUG] Whisper вернул {len(result['segments'])} сегментов")
    if 'segments' in result and len(result['segments']) > 0:
        for i, segment in enumerate(result['segments'][:3]):  # Показываем первые 3 сегмента
            print(f"[DEBUG] Whisper сегмент {i}: start={segment['start']}, end={segment['end']}, text={segment['text'][:50]}...")

    # Преобразуем результаты в удобный формат
    segments = [{
        'start': float(segment['start']),
        'end': float(segment['end']),
        'text': segment['text']
    } for segment in result['segments']]
    
    # Проверяем на наличие дубликатов в исходных данных Whisper
    texts = [(s['start'], s['end'], s['text']) for s in segments]
    duplicates = []
    for i, item in enumerate(texts):
        if texts.count(item) > 1:
            duplicates.append((i, item))
    if duplicates:
        print(f"[DEBUG] Найдено {len(duplicates)} дубликатов прямо из Whisper!")
        for idx, (start, end, text) in duplicates[:3]:
            print(f"[DEBUG] Дубликат из Whisper {idx}: [{start}-{end}] {text[:50]}...")
    
    return segments