import torch
from src.emo_recognizer import EmotionRecognizer
from config import FILES
from typing import List, Dict
import torchaudio

def analyze_emotions(audio_path: str, transcription_segments: List[Dict]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[INFO] Начинаем анализ эмоций...")

    # Get audio with torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Проверяем размерность waveform и преобразуем если нужно
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Добавляем размерность для каналов
    
    # Объединяем короткие сегменты
    combined_segments = []
    current_segment = None
    
    for segment in transcription_segments:
        segment_duration = segment['end'] - segment['start']
        
        if current_segment is None:
            current_segment = segment
            continue
            
        if segment_duration < 5.0:  # Если текущий сегмент короткий
            # Объединяем с предыдущим сегментом
            current_segment['end'] = segment['end']
            current_segment['text'] += f" {segment['text']}"
        else:
            combined_segments.append(current_segment)
            current_segment = segment
            
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    print(f"[INFO] Объединено {len(transcription_segments)} коротких сегментов в {len(combined_segments)} сегментов")
    
    emotion_recognizer = EmotionRecognizer(device=device)
    audio_text_pairs = [(waveform[0, int(segment['start']*sample_rate):int(segment['end']*sample_rate)], segment['text']) for segment in combined_segments]
    
    emotion_results = emotion_recognizer.batch_recognize(audio_text_pairs, sample_rate)
    
    # Process results
    emotion_text_segments = []
    for segment, emotions in zip(combined_segments, emotion_results):
        emotion_name = max(emotions.items(), key=lambda x: x[1])[0]
        emotion_confidence = emotions[emotion_name] * 100
        emotion_name = FILES['emotion_translations'].get(emotion_name.lower(), emotion_name)
        
        emotion_text_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'emotion': emotion_name,
            'confidence': emotion_confidence
        })
    
    print(f"[INFO] Найдено {len(emotion_text_segments)} сегментов с эмоциями")
    return emotion_text_segments
