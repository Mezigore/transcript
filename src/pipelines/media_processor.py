import os
import time
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from config import OUTPUT_DIR, TEMP_DIR
from src.audio.processing import extract_and_process_audio
from src.audio.segmentation import segment_audio
from src.analysis.transcription import transcribe_audio
from src.analysis.diarization import diarize
from src.analysis.emotion import analyze_emotions
from src.utils.segment_merger import create_transcript
from src.utils.filesystem import ensure_directories, cleanup_temp_files

@dataclass
class ProcessingResult:
    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None
    execution_times: Dict[str, str] = field(default_factory=dict)

def format_time(seconds: float) -> str:
    """Форматирует время в удобочитаемый формат"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}м {seconds}с"

class MediaProcessor:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        ensure_directories()
        
    def process(self, media_path: str) -> ProcessingResult:
        """Основной метод обработки медиа-файла"""
        try:
            start_time = time.time()
            
            # Извлечение и обработка аудио
            audio_start = time.time()
            audio_tensor, sample_rate = extract_and_process_audio(media_path)
            audio_time = format_time(time.time() - audio_start)
            
            # Сегментация аудио
            segment_start = time.time()
            segments = segment_audio(audio_tensor, sample_rate)
            segment_time = format_time(time.time() - segment_start)
            
            # Транскрипция аудио
            transcribe_start = time.time()
            transcription = transcribe_audio(segments, self.hf_token)
            transcribe_time = format_time(time.time() - transcribe_start)
            
            # Диаризация (определение говорящих)
            diarize_start = time.time()
            speakers = diarize(audio_tensor, sample_rate, self.hf_token)
            diarize_time = format_time(time.time() - diarize_start)
            
            # Анализ эмоций
            emotion_start = time.time()
            emotions = analyze_emotions(segments, self.hf_token)
            emotion_time = format_time(time.time() - emotion_start)
            
            # Создание итогового файла
            output_file = os.path.splitext(os.path.basename(media_path))[0] + ".txt"
            result_file = create_transcript(
                transcription, speakers, emotions, output_file
            )
            
            # Очистка временных файлов
            cleanup_temp_files()
            
            total_time = format_time(time.time() - start_time)
            
            return ProcessingResult(
                success=True,
                file_path=result_file,
                execution_times={
                    "Обработка аудио": audio_time,
                    "Сегментация": segment_time,
                    "Транскрипция": transcribe_time,
                    "Диаризация": diarize_time,
                    "Анализ эмоций": emotion_time,
                    "Общее время": total_time
                }
            )
        except Exception as e:
            cleanup_temp_files()
            return ProcessingResult(
                success=False,
                error=str(e)
            ) 