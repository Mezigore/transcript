import torch
from src.emo_recognizer import EmotionRecognizer
from config import FILES, TRANSCRIPTION
from src.chunking import chunking_audio
import mlx_whisper
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import tempfile
import os
import shutil

def analyze_emotions(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Разделяем аудио на фрагменты
    chunked_audios, sample_rate = chunking_audio(audio_path, max_segment_length=TRANSCRIPTION['segment_size'])
    print(f"[DEBUG] Created {len(list(chunked_audios))} audio segments")
    
    # === ЭТАП 1: Транскрипция всех сегментов ===
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
        
        # Очищаем память от модели транскрипции
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # === ЭТАП 2: Анализ эмоций ===
        print("[INFO] Начало анализа эмоций...")
        
        # Инициализируем модель распознавания эмоций
        emotion_recognizer = EmotionRecognizer(device=device)
        
        # Обрабатываем эмоции в многопоточном режиме
        def process_segment_emotion(segment):
            emotions = emotion_recognizer.recognize(segment['file'], segment['text'])
            emotion_name = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_confidence = emotions[emotion_name] * 100
            emotion_name = FILES['emotion_translations'].get(emotion_name.lower(), emotion_name)
            
            return {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'emotion': emotion_name,
                'confidence': emotion_confidence
            }
        
        # Обработка в пуле потоков с правильным отображением прогресса
        emotion_text_segments = []
        with ThreadPoolExecutor(max_workers=min(8, len(all_transcriptions))) as executor:
            futures = [executor.submit(process_segment_emotion, segment) for segment in all_transcriptions]
            for future in tqdm(futures, total=len(futures), desc="Анализ эмоций"):
                emotion_text_segments.append(future.result())
        
        print(f"[INFO] Найдено {len(emotion_text_segments)} сегментов с эмоциями")
        return emotion_text_segments
        
    finally:
        # Очистка временных файлов
        shutil.rmtree(temp_dir)
