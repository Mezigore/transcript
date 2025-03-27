from config import TEMP_DIR, OUTPUT_DIR
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from torch import Tensor
from ..utils.segment_merger import merge_transcription_with_diarization
from ..utils.text_formatter import format_transcript
from ..audio.processing import extract_and_process_audio
from ..audio.segmentation import segment_audio, vad_filter
from ..analysis.transcription import transcribe_audio
from ..analysis.diarization import diarize
from ..analysis.emotion import analyze_emotions
from ..utils.filesystem import  ensure_directories, cleanup_temp_files

@dataclass
class ProcessingResult:
    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None
    execution_times: Dict[str, str] = field(default_factory=dict)

def format_time(seconds: float) -> str:
    """Форматирование времени выполнения."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}м {seconds}с"

def timed_execution(log_key: str):
    """Декоратор для замера времени выполнения функции."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                self.execution_times[log_key] = format_time(time.time() - start_time)
                return result
            except Exception as e:
                raise RuntimeError(f"{func.__name__} failed: {str(e)}")
        return wrapper
    return decorator

class AudioProcessingPipeline:
    """Главный класс для обработки аудио и медиа-файлов."""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.execution_times = {}
        ensure_directories()
    
    @timed_execution('audio_extraction')
    def _extract_audio(self, media_path: str) -> Tuple[Tensor, int]:
        """Извлечение и обработка аудио."""
        audio_tensor, sample_rate = extract_and_process_audio(media_path, output_path=TEMP_DIR + "_temp_audio.wav")
        # Дополнительная обработка VAD для удаления тишины
        filtered_tensor = vad_filter(audio_tensor, sample_rate)
        return filtered_tensor, sample_rate
    
    @timed_execution('segmentation')
    def _segment_audio(self, audio_tensor: Tensor, sample_rate: int) -> List[Tensor]:
        """Сегментация аудио на части для обработки."""
        if audio_tensor.size(1) > sample_rate * 60:  # Если длиннее 1 минуты
            return segment_audio(audio_tensor, sample_rate)
        else:
            return [audio_tensor]
    
    @timed_execution('transcription')
    def _transcribe(self, audio_segments: List[Tensor], sample_rate: int) -> List[Dict[str, Any]]:
        """Транскрибирование аудио."""
        all_transcriptions = []
        for i, segment_tensor in enumerate(audio_segments):
            segment_transcription = transcribe_audio(segment_tensor, sample_rate)
            # Корректируем временные метки
            segment_start = i * 30  # 30 секунд на сегмент
            for trans in segment_transcription:
                trans["start"] += segment_start
                trans["end"] += segment_start
            all_transcriptions.extend(segment_transcription)
        return all_transcriptions
    
    @timed_execution('diarization')
    def _diarize(self, audio_tensor: Tensor, sample_rate: int) -> List[Dict[str, Any]]:
        """Определение говорящих."""
        return diarize(audio_tensor, sample_rate, self.hf_token)
    
    @timed_execution('emotion_analysis')
    def _analyze_emotions(self, audio_tensor: Tensor, sample_rate: int, transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Анализ эмоций в речи."""
        return analyze_emotions(audio_tensor, sample_rate, transcription_segments)
    
    @timed_execution('merging')
    def _merge_results(self, diarized_segments: List[Dict[str, Any]], emotion_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Объединение результатов диаризации и эмоций."""
        return merge_transcription_with_diarization(diarized_segments, emotion_segments, 100)
    
    @timed_execution('file_creation')
    def _create_output_file(self, merged_segments: List[Dict[str, Any]], media_path: str) -> str:
        """Создание файла с результатами транскрипции."""
        base_output = os.path.splitext(os.path.basename(media_path))[0]
        output_path = f"{OUTPUT_DIR}/{base_output}_transcript.txt"
        text = format_transcript(merged_segments, 100)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return output_path
    
    def process(self, media_path: str, skip_emotion_analysis: bool = False) -> ProcessingResult:
        """Обработка медиа-файла через полный пайплайн.
        
        Args:
            media_path: Путь к медиа-файлу для обработки
            skip_emotion_analysis: Если True, шаг анализа эмоций будет пропущен
        """
        start_time = time.time()
        try:
            if not os.path.exists(media_path):
                return ProcessingResult(success=False, error="[ОШИБКА] Файл медиа не найден")

            # Подготовка директорий
            ensure_directories()

            # Основные шаги обработки
            audio_tensor, sample_rate = self._extract_audio(media_path)
            audio_segments = self._segment_audio(audio_tensor, sample_rate)
            transcription = self._transcribe(audio_segments, sample_rate)
            diarization = self._diarize(audio_tensor, sample_rate)
            
            # Пропускаем анализ эмоций, если указано
            if skip_emotion_analysis:
                emotions = []  # Пустой список вместо результатов анализа эмоций
                self.execution_times['emotion_analysis'] = "Пропущено"
            else:
                emotions = self._analyze_emotions(audio_tensor, sample_rate, transcription)
            
            merged = self._merge_results(diarization, emotions)
            output_path = self._create_output_file(merged, media_path)
            print(f"[INFO] Результаты транскрипции сохранены в {output_path}")
            # Добавляем общее время выполнения
            self.execution_times['Общее время'] = format_time(time.time() - start_time)

            # Очистка временных файлов
            cleanup_temp_files()

            return ProcessingResult(
                success=True,
                file_path=output_path,
                execution_times=self.execution_times
            )
        except Exception as e:
            cleanup_temp_files()
            return ProcessingResult(success=False, error=str(e)) 