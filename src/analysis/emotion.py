import torch
import torchaudio
import logging
import tqdm
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
            # Минимальная длина для wav2vec - 2 секунды (из-за ограничений модели)
            MIN_DURATION = 2.0
            
            if segment_duration < MIN_DURATION:  # Если сегмент слишком короткий
                # Если текущий сегмент не инициализирован или его длительность будет меньше минимальной
                if current_segment is None or (segment['end'] - current_segment['start']) < MIN_DURATION:
                    # Сохраняем как новый текущий сегмент
                    current_segment = segment.copy()
                else:
                    # Объединяем с предыдущим
                    current_segment['end'] = segment['end']
                    current_segment['text'] += f" {segment['text']}"
            elif segment_duration > 7.0:  # Если сегмент слишком длинный
                # Добавляем предыдущий сегмент, если он был
                if current_segment is not None and (current_segment['end'] - current_segment['start']) >= MIN_DURATION:
                    combined_segments.append(current_segment)
                
                # Разбиваем на части по 5-7 секунд (с запасом от минимальной длины)
                start_time = segment['start']
                while start_time < segment['end']:
                    end_time = min(start_time + 5.0, segment['end'])
                    
                    # Проверяем, что получившийся сегмент достаточной длины
                    if (end_time - start_time) >= MIN_DURATION:
                        new_segment = {
                            'start': start_time,
                            'end': end_time,
                            'text': segment['text'],
                            'speaker': segment.get('speaker', 'Unknown')
                        }
                        combined_segments.append(new_segment)
                    
                    start_time = end_time
                
                # Сбрасываем текущий сегмент
                current_segment = None
            else:
                # Обычный сегмент нормальной длины
                if current_segment is not None and (current_segment['end'] - current_segment['start']) >= MIN_DURATION:
                    combined_segments.append(current_segment)
                current_segment = segment.copy()
    
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    logger.info(f"Combined {len(transcription_segments)} short segments into {len(combined_segments)} segments")
    
    # Инициализация соответствующего движка эмоций
    if emotion_engine == "wavlm":
        emotion_recognizer = EmotionRecognizer(device=device)
    else:
        emotion_recognizer = Wav2VecEmotionRecognizer(device_name=device)
    
    # Process segments in smaller batches to manage memory
    max_segments_per_batch = 10
    emotion_text_segments = []
    
    # Добавляем индикатор прогресса
    total_batches = (len(combined_segments) + max_segments_per_batch - 1) // max_segments_per_batch
    with tqdm.tqdm(total=total_batches, desc=f"Анализ эмоций ({emotion_engine})") as progress_bar:
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
                    
                    # Минимальная длина для wav2vec - 2 сэмпла
                    if emotion_engine == "wav2vec":
                        min_samples = 2 * sample_rate  # минимум 2 секунды
                        if (end_idx - start_idx) < min_samples:
                            logger.warning(f"Сегмент слишком короткий для wav2vec: {(end_idx - start_idx)/sample_rate:.2f}с ({end_idx - start_idx} сэмплов)")
                            continue
                    
                    if end_idx <= start_idx:
                        logger.warning(f"Skipping invalid segment: start={segment['start']}, end={segment['end']}")
                        continue
                except (TypeError, ValueError) as e:
                    logger.error(f"Error processing segment {segment}: {e}")
                    continue
                    
                audio_segment = audio_file[0, start_idx:end_idx]
                audio_text_pairs.append((audio_segment, segment['text']))
            
            if not audio_text_pairs:
                progress_bar.update(1)
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
            
            # Обновляем индикатор прогресса
            progress_bar.update(1)
    
    logger.info(f"Found {len(emotion_text_segments)} emotion segments")
    return emotion_text_segments


