import mlx_whisper
from tqdm import tqdm
import soundfile as sf
from config import TRANSCRIPTION

def transcribe_audio(audio_path):
    print("[INFO] Начало транскрипции аудио...")
    
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=TRANSCRIPTION['model_path'],
        initial_prompt=TRANSCRIPTION['initial_prompt'],
        condition_on_previous_text=True,
        **TRANSCRIPTION['decode_options']
    )
    
    if result is None:
        print(f"[ERROR] Failed to transcribe audio: {audio_path}")
        return []
    
    print(f"[INFO] Транскрипция завершена: {audio_path}")
    print(f"[INFO] {len(result['segments'])} сегментов найдено")
    print(f"[INFO] Первый сегмент: {result['segments'][0]}")

    # Преобразуем результаты в удобный формат
    return [{
        'file': audio_path,
        'start': float(segment['start']),
        'end': float(segment['end']),
        'text': segment['text']
    } for segment in result['segments']]