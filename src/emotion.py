import torch
from src.emo_recognizer import EmotionRecognizer
from config import FILES
import gc

def analyze_emotions(audio_path, transcription_segments):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[INFO] Starting emotion analysis...")
    
    emotion_recognizer = EmotionRecognizer(device=device)
    audio_text_pairs = [(segment['file'], segment['text']) for segment in transcription_segments]
    
    emotion_results = emotion_recognizer.batch_recognize(audio_text_pairs)
    
    # Process results
    emotion_text_segments = []
    for segment, emotions in zip(transcription_segments, emotion_results):
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
    
    print(f"[INFO] Found {len(emotion_text_segments)} segments with emotions")
    return emotion_text_segments
