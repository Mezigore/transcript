import torch
import torchaudio
import logging
from src.analysis.emo_recognizer import EmotionRecognizer
from src.analysis.emotion_wav2vec import Wav2VecEmotionRecognizer
from config import FILES
from typing import List, Dict, Optional

# Настройка логирования вместо print
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def analyze_emotions(
    audio_file: torch.Tensor, 
    sample_rate: int, 
    transcription_segments: List[Dict],
    emotion_engine: str = "wavlm"  # "wavlm" или "wav2vec"
) -> List[Dict]:
    """
    Анализ эмоций в аудио с использованием выбранного движка.
    
    Args:
        audio_file: Тензор аудио
        sample_rate: Частота дискретизации
        transcription_segments: Сегменты транскрипции
        emotion_engine: Выбор движка эмоций ("wavlm" или "wav2vec")
    """
    device = 'cpu'  # Default to CPU to prevent CUDA memory issues
    
    logger.info(f"Starting emotion analysis with {emotion_engine} engine...")
    
    # Ensure correct audio tensor dimensions
    if len(audio_file.shape) == 1:
        audio_file = audio_file.unsqueeze(0)  # Add channel dimension
    
    # Combine short segments
    combined_segments = []
    current_segment = None
    
    for segment in transcription_segments:
        segment_duration = segment['end'] - segment['start']
        
        if current_segment is None:
            current_segment = segment.copy()
            continue
        
        # Разная логика объединения для разных движков
        if emotion_engine == "wavlm":
            if segment_duration < 7.0:  # If current segment is short
                # Merge with previous segment
                current_segment['end'] = segment['end']
                current_segment['text'] += f" {segment['text']}"
            else:
                combined_segments.append(current_segment)
                current_segment = segment.copy()
        else:  # wav2vec
            if segment_duration < 2.0:  # Если сегмент слишком короткий
                # Объединяем с предыдущим
                current_segment['end'] = segment['end']
                current_segment['text'] += f" {segment['text']}"
            elif segment_duration > 7.0:  # Если сегмент слишком длинный
                # Разбиваем на части по 7 секунд
                if current_segment is not None:
                    combined_segments.append(current_segment)
                start_time = segment['start']
                while start_time < segment['end']:
                    end_time = min(start_time + 7.0, segment['end'])
                    new_segment = {
                        'start': start_time,
                        'end': end_time,
                        'text': segment['text'],
                        'speaker': segment.get('speaker', 'Unknown')  # Сохраняем информацию о спикере
                    }
                    combined_segments.append(new_segment)
                    start_time = end_time
                current_segment = None
            else:
                if current_segment is not None:
                    combined_segments.append(current_segment)
                current_segment = segment.copy()
    
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    logger.info(f"Combined {len(transcription_segments)} short segments into {len(combined_segments)} segments")
    
    # Инициализация соответствующего движка эмоций
    if emotion_engine == "wavlm":
        emotion_recognizer = EmotionRecognizer(device=device)
    else:
        emotion_recognizer = Wav2VecEmotionRecognizer(device=device)
    
    # Process segments in smaller batches to manage memory
    max_segments_per_batch = 10
    emotion_text_segments = []
    
    for i in range(0, len(combined_segments), max_segments_per_batch):
        batch_segments = combined_segments[i:i+max_segments_per_batch]
        audio_text_pairs = []
        
        for segment in batch_segments:
            if 'start' not in segment or 'end' not in segment:
                logger.warning(f"Skipping segment without start/end times: {segment}")
                continue
                
            try:
                start_idx = max(0, int(segment['start'] * sample_rate))
                end_idx = min(int(segment['end'] * sample_rate), audio_file.shape[1])
                
                if end_idx <= start_idx:
                    logger.warning(f"Skipping invalid segment: start={segment['start']}, end={segment['end']}")
                    continue
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing segment {segment}: {e}")
                continue
                
            audio_segment = audio_file[0, start_idx:end_idx]
            audio_text_pairs.append((audio_segment, segment['text']))
        
        if not audio_text_pairs:
            continue
            
        try:
            emotion_results = emotion_recognizer.batch_recognize(audio_text_pairs, sample_rate)
            
            # Process results
            for segment, emotions in zip(batch_segments, emotion_results):
                if not emotions:
                    continue
                    
                if emotion_engine == "wavlm":
                    emotion_name = max(emotions.items(), key=lambda x: x[1])[0]
                    emotion_confidence = emotions[emotion_name] * 100
                    emotion_name = FILES['emotion_translations'].get(emotion_name.lower(), emotion_name)
                    
                    emotion_text_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'],
                        'emotion': emotion_name,
                        'confidence': emotion_confidence,
                        'speaker': segment.get('speaker', 'Unknown')  # Сохраняем информацию о спикере
                    })
                else:  # wav2vec
                    emotion_text_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'],
                        'emotions': emotions,  # Словарь с тремя эмоциями: возбуждение, доминирование, валентность
                        'speaker': segment.get('speaker', 'Unknown')  # Сохраняем информацию о спикере
                    })
                
        except Exception as e:
            logger.error(f"Batch emotion recognition failed: {e}")
    
    logger.info(f"Found {len(emotion_text_segments)} emotion segments")
    return emotion_text_segments


