import torch
from aniemore.recognizers.multimodal import VoiceTextRecognizer
from aniemore.models import HuggingFaceModel
from config import  FILES, TRANSCRIPTION
from src.chunking import chunking_audio
import mlx_whisper

def analyze_emotions(audio_path: str):
    model = HuggingFaceModel.MultiModal.WavLMBertFusion
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vt_recognizer = VoiceTextRecognizer(model=model, device=device)
    
    chunked_audios, sample_rate = chunking_audio(audio_path, max_segment_length=TRANSCRIPTION['segment_size'])
    
    print("[INFO] Анализ эмоций в аудио...")
    
    emotion_text_segments = []
    prompt = TRANSCRIPTION['initial_prompt']
    
    # Создаем временную директорию для сегментов
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Обрабатываем каждый сегмент аудио
        for audio_data, start_time in chunked_audios:
            # Создаем временный файл для сегмента
            segment_file = os.path.join(temp_dir, f"segment_{start_time:.2f}.wav")
            
            # Сохраняем аудио данные во временный файл
            import soundfile as sf
            sf.write(segment_file, audio_data, sample_rate)
            
            # Получаем транскрипцию с помощью Whisper
            transcription_result = mlx_whisper.transcribe(
                segment_file, 
                path_or_hf_repo=TRANSCRIPTION['model_path'], 
                initial_prompt=prompt,
                condition_on_previous_text=True,
                **TRANSCRIPTION['decode_options']
            )
            transcription = transcription_result['text']
            
            # Распознаем эмоции, передавая пару аудио-текст
            emotions = vt_recognizer.recognize((segment_file, transcription))
            
            # Обрабатываем сегменты из транскрипции
            for segment in transcription_result['segments']:
                # Определяем эмоцию для этого сегмента
                emotion_name = vt_recognizer._get_single_label(emotions)
                
                # Получаем уверенность для определенной эмоции
                emotion_confidence = emotions.get(emotion_name.lower(), 0.0) * 100  # Переводим в проценты
                
                # Преобразуем эмоцию в соответствии с маппингом из конфигурации
                emotion_name = FILES['emotion_translations'].get(emotion_name.lower(), emotion_name)
                
                # Корректируем временные метки с учетом начала сегмента
                segment_start = float(segment['start']) + start_time
                segment_end = float(segment['end']) + start_time
                
                emotion_text_segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'text': segment['text'],
                    'emotion': emotion_name,
                    'confidence': emotion_confidence
                })
    
    finally:
        # Удаляем временные файлы
        import shutil
        shutil.rmtree(temp_dir)
    
    print(f"[INFO] Найдено {len(emotion_text_segments)} сегментов с эмоциями")
    return emotion_text_segments
