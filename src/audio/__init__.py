# Инициализация модуля аудио
from .audio_pipeline import extract_and_process_audio
from .audio_extractor import extract_audio
from .audio_processor import process_audio

__all__ = ['extract_and_process_audio', 'extract_audio', 'process_audio'] 