import mlx_whisper
from tqdm import tqdm
import soundfile as sf
import tempfile
import os
import shutil
from config import TRANSCRIPTION
from src.chunking import chunking_audio

def transcribe_audio(audio_path):
    # Разделяем аудио на фрагменты
    chunked_audios, sample_rate = chunking_audio(audio_path, max_segment_length=TRANSCRIPTION['segment_size'])
    print(f"[DEBUG] Created {len(list(chunked_audios))} audio segments")
    
    print("[INFO] Начало транскрипции аудио...")
    
    # Подготавливаем все сегменты
    segment_files = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Сохраняем сегменты на диск
        for idx, (audio_data, start_time) in enumerate(chunked_audios):
            segment_file = os.path.join(temp_dir, f"segment_{idx:04d}_{start_time:.2f}.wav")
            sf.write(segment_file, audio_data, sample_rate)
            segment_files.append((segment_file, start_time))
        
        # Транскрибируем все сегменты
        all_transcriptions = []
        prompt = TRANSCRIPTION['initial_prompt']
        
        for segment_file, start_time in tqdm(segment_files, desc="Транскрипция сегментов"):
            result = mlx_whisper.transcribe(
                segment_file,
                path_or_hf_repo=TRANSCRIPTION['model_path'],
                initial_prompt=prompt,
                condition_on_previous_text=True,
                **TRANSCRIPTION['decode_options']
            )
            
            # Сохраняем результаты транскрипции с информацией о времени начала
            for segment in result['segments']:
                all_transcriptions.append({
                    'file': segment_file,
                    'start': float(segment['start']) + start_time,
                    'end': float(segment['end']) + start_time,
                    'text': segment['text']
                })
        
        return all_transcriptions
        
    finally:
        # Временные файлы будут очищены после завершения всех операций в run.py
        pass