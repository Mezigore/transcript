from typing import Dict, Any, List
import textwrap
from config import OUTPUT_FORMAT

def get_friendly_speaker_name(speaker_label):
    """Преобразует метку спикера в более дружественное имя"""
    try:
        number = int(speaker_label.split('_')[-1])
        # SPEAKER_00 -> 0 + 1 = "1 Спикер"
        # SPEAKER_01 -> 1 + 1 = "2 Спикер"
        # Сохраняем для отладки исходный идентификатор
        print(f"[DEBUG] Преобразование спикера: {speaker_label} -> {number + 1} Спикер")
        return f"{number + 1} Спикер"
    except:
        return speaker_label

def format_segment_text(text, max_line_length=None, indent=" "):
    """Форматирует текст с учетом переноса строк"""
    if not max_line_length:
        return text
        
    return textwrap.fill(
        text,
        width=max_line_length - len(indent),
        initial_indent="",
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False
    )

def format_segment_with_metadata(segment: Dict[str, Any], max_line_length=None) -> str:
    """Форматирует сегмент с метаданными (спикер, время, эмоции)"""
    avg_confidence = segment["emotion_confidence"] if "emotion_confidence" in segment else 0.0
    
    start_time = f"{int(segment['start'] // 60)}:{int(segment['start'] % 60):02d}"
    end_time = f"{int(segment['end'] // 60)}:{int(segment['end'] % 60):02d}"
    
    timestamp_info = f"[{start_time} - {end_time}] " if OUTPUT_FORMAT['include_timestamps'] else ""
    emotion_info = f" ({segment['emotion']}, {min(avg_confidence, 100):.1f}%)" if avg_confidence >= OUTPUT_FORMAT['min_confidence_threshold'] else ""
    
    header = f"{timestamp_info}{segment['speaker']}{emotion_info}:"
    formatted_text = format_segment_text(segment['text'], max_line_length=max_line_length)
    
    return f"{header}\n{formatted_text}"

def format_time(seconds: float) -> str:
    """Форматирование времени в формат MM:SS."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_transcript(segments: List[Dict], confidence_threshold: float = 100) -> str:
    """Форматирование транскрипции в текстовый формат."""
    # Используем порог из конфигурации вместо жестко заданного значения
    use_threshold = OUTPUT_FORMAT.get('min_confidence_threshold', 0.5) * 100
    
    formatted_text = []
    
    for segment in segments:
        # Форматирование времени
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        
        # Получаем информацию о спикере
        speaker = get_friendly_speaker_name(segment.get('speaker', 'Unknown'))
        
        # Форматирование текста с эмоциями
        text = segment.get('text', '').strip()
        if not text:
            continue  # Пропускаем пустые сегменты
            
        emotion_info = ""
        # Проверяем наличие эмоций и форматируем их в зависимости от модели
        if OUTPUT_FORMAT.get('include_emotions', True):  # Проверяем, включены ли эмоции в вывод
            # Проверяем тип эмоций (старый формат для wavlm или новый A,D,V для wav2vec)
            if 'emotion' in segment:
                # Формат для модели wavlm с одной эмоцией
                emotion = segment['emotion']
                confidence = segment.get('confidence', 0)
                if confidence >= use_threshold:  # Используем порог из конфигурации
                    emotion_info = f"[{emotion} ({confidence:.1f}%)] "
            elif 'emotions' in segment and segment['emotions']:
                # Формат для модели wav2vec с тремя параметрами (A,D,V)
                emotions = segment['emotions']
                if emotions:
                    arousal = emotions.get('возбуждение', 0)
                    dominance = emotions.get('доминирование', 0)
                    valence = emotions.get('валентность', 0)
                    # Форматируем значения с двумя знаками после запятой
                    emotion_info = f"[A:{arousal:.2f}, D:{dominance:.2f}, V:{valence:.2f}] "
        
        # Добавляем временные метки и информацию о спикере
        formatted_text.append(f"[{start_time} -> {end_time}] {speaker}: {emotion_info}{text}")
    
    return "\n".join(formatted_text) 