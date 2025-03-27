import torch
import torchaudio
import logging
from src.analysis.emo_recognizer import EmotionRecognizer
from config import FILES
from typing import List, Dict

# Настройка логирования вместо print
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def analyze_emotions(audio_file: torch.Tensor, sample_rate: int, transcription_segments: List[Dict]):
    device = 'cpu'  # Default to CPU to prevent CUDA memory issues
    
    logger.info("Starting emotion analysis...")
    
    # Ensure correct audio tensor dimensions
    if len(audio_file.shape) == 1:
        audio_file = audio_file.unsqueeze(0)  # Add channel dimension
    
    # Downsample very long audio files to reduce memory usage
    max_audio_length = 1_000_000  # Adjust as needed
    if audio_file.shape[1] > max_audio_length:
        new_sample_rate = sample_rate // 2
        # Исправлена скобка в конце вызова функции
        audio_file = torchaudio.functional.resample(
            audio_file,
            orig_freq=sample_rate,
            new_freq=new_sample_rate
        )  
        sample_rate = new_sample_rate
        logger.info(f"Downsampled audio to {new_sample_rate} Hz")
    
    # Combine short segments
    combined_segments = []
    current_segment = None
    
    for segment in transcription_segments:
        segment_duration = segment['end'] - segment['start']
        
        if current_segment is None:
            # Используем копию словаря для предотвращения мутации оригинала
            current_segment = segment.copy()
            continue
        
        if segment_duration < 7.0:  # If current segment is short
            # Merge with previous segment
            current_segment['end'] = segment['end']
            current_segment['text'] += f" {segment['text']}"
        else:
            combined_segments.append(current_segment)
            current_segment = segment.copy()
    
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    logger.info(f"Combined {len(transcription_segments)} short segments into {len(combined_segments)} segments")
    
    emotion_recognizer = EmotionRecognizer(device=device)
    
    # Process segments in smaller batches to manage memory
    max_segments_per_batch = 10
    emotion_text_segments = []
    
    for i in range(0, len(combined_segments), max_segments_per_batch):
        batch_segments = combined_segments[i:i+max_segments_per_batch]
        audio_text_pairs = []
        
        for segment in batch_segments:
            # Проверка наличия необходимых ключей и их типов
            if 'start' not in segment or 'end' not in segment:
                logger.warning(f"Skipping segment without start/end times: {segment}")
                continue
                
            try:
                # Более безопасное вычисление индексов
                start_idx = max(0, int(segment['start'] * sample_rate))
                end_idx = min(int(segment['end'] * sample_rate), audio_file.shape[1])
                
                # Проверка на корректность сегмента
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
                if not emotions:  # Проверка на пустой словарь результатов
                    continue
                    
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
                
        except Exception as e:
            logger.error(f"Batch emotion recognition failed: {e}")
    
    logger.info(f"Found {len(emotion_text_segments)} emotion segments")
    return emotion_text_segments


