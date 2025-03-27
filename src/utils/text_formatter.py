from typing import Dict, Any, List
import textwrap
from config import OUTPUT_FORMAT

def get_friendly_speaker_name(speaker_label):
    """Преобразует метку спикера в более дружественное имя"""
    try:
        number = int(speaker_label.split('_')[-1])
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
        initial_indent=indent,
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

def format_transcript(merged_segments: List[Dict[str, Any]], max_line_length=None) -> str:
    """Форматирует объединенные сегменты в текст"""
    formatted_segments = []
    
    for segment in sorted(merged_segments, key=lambda x: x['start']):
        if 'formatted' not in segment:
            segment['formatted'] = format_segment_with_metadata(segment, max_line_length)
        formatted_segments.append(segment['formatted'])
    
    return "\n\n".join(formatted_segments) 