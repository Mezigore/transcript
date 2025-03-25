# Функция для объединения результатов транскрипции и диаризации
from config import OUTPUT_FORMAT

def get_friendly_speaker_name(speaker_label):
    """Convert technical speaker label (e.g., SPEAKER_00) to a more friendly name (e.g., Speaker 1)"""
    try:
        # Extract the number from the speaker label (e.g., "SPEAKER_00" -> "00")
        number = int(speaker_label.split('_')[-1])
        return f"Speaker {number + 1}"
    except:
        # If the label is not in expected format, return it as is
        return speaker_label

def format_segment_text(text, max_line_length=None, indent="                        "):
    """
    Форматирует текст сегмента с учетом переноса строк
    
    Args:
        text (str): Исходный текст
        max_line_length (int): Максимальная длина строки (None - без ограничений)
        indent (str): Отступ для всех строк текста
        
    Returns:
        str: Отформатированный текст
    """
    if max_line_length is None:
        return text
        
    import textwrap
    # Уменьшаем доступную ширину на размер отступа
    available_width = max_line_length - len(indent)
    # Форматируем текст с отступом для каждой строки
    wrapped_text = textwrap.fill(text, 
                               width=available_width,
                               initial_indent=indent,
                               subsequent_indent=indent,
                               break_long_words=False,
                               break_on_hyphens=False)
    return wrapped_text

def merge_transcription_with_diarization(diarized_segments, text_and_emotion_segments, max_line_length=None):
    print("[INFO] Объединение транскрипции с диаризацией и эмоциями...")
    merged_segments = []
    temp_segment = None
    MAX_GAP = 5.0  # Максимальный промежуток между сегментами в секундах для объединения
    
    print(f"[INFO] Получено {len(diarized_segments)} сегментов диаризации")
    print(f"[INFO] Получено {len(text_and_emotion_segments)} сегментов с эмоциями")
    
    # Сортируем сегменты по времени начала
    text_and_emotion_segments.sort(key=lambda x: x["start"])
    diarized_segments.sort(key=lambda x: x["start"])
    
    # Используем segments из результата Whisper
    for whisper_segment in text_and_emotion_segments:
        # Находим спикера, который говорит в этом временном интервале
        max_overlap = 0
        best_match_diarized = None
        
        whisper_start = float(whisper_segment["start"])
        whisper_end = float(whisper_segment["end"])
        
        # Поиск наилучшего совпадения в diarized_segments
        for speaker_segment in diarized_segments:
            # Пропускаем если сегменты не перекрываются
            if speaker_segment["end"] < whisper_start or speaker_segment["start"] > whisper_end:
                continue
                
            # Вычисляем перекрытие
            overlap_start = max(whisper_start, float(speaker_segment["start"]))
            overlap_end = min(whisper_end, float(speaker_segment["end"]))
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_diarized = speaker_segment
        
        # Определяем спикера из результатов диаризации
        if best_match_diarized:
            speaker = get_friendly_speaker_name(best_match_diarized["speaker"])
            # Add debug logging
            # print(f"[DEBUG] Found speaker {speaker} for segment [{whisper_start:.2f}-{whisper_end:.2f}] with overlap {max_overlap:.2f}")
        else:
            # Log a warning if no speaker is found
            print(f"[WARNING] No speaker found for segment [{whisper_start:.2f}-{whisper_end:.2f}]")
            # Use the speaker from the previous segment if available
            if merged_segments:
                speaker = merged_segments[-1]["speaker"]
            else:
                # If it's the first segment and no speaker is found, raise an error
                raise ValueError(f"No speaker found for first segment [{whisper_start:.2f}-{whisper_end:.2f}]")
        
        # Извлекаем эмоцию и уверенность из whisper_segment
        emotion = whisper_segment.get("emotion", "")
        emotion_confidence = whisper_segment.get("confidence", 0.0)
        
        # Проверяем порог уверенности для эмоции
        if emotion_confidence < OUTPUT_FORMAT['min_confidence_threshold']:
            emotion = None  
            emotion_confidence = None
        
        # Если это первый сегмент
        if temp_segment is None:
            temp_segment = {
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"].strip(),
                "speaker": speaker,
                "emotion": emotion,
                "confidence_sum": emotion_confidence,
                "segments_count": 1
            }
        # Если тот же спикер, та же эмоция и пауза не превышает MAX_GAP
        elif (temp_segment["speaker"] == speaker and 
              temp_segment["emotion"] == emotion and 
              whisper_start - temp_segment["end"] <= MAX_GAP):
            # Объединяем сегменты
            temp_segment["end"] = whisper_end
            # Добавляем пробел только если предыдущий текст не заканчивается знаком пунктуации
            if temp_segment["text"][-1] not in ".!?":
                temp_segment["text"] += " "
            temp_segment["text"] += whisper_segment["text"].strip()
            temp_segment["confidence_sum"] += emotion_confidence
            temp_segment["segments_count"] += 1
        else:
            # Завершаем текущий объединенный сегмент
            avg_confidence = temp_segment["confidence_sum"] / temp_segment["segments_count"]
            start_time = f"{int(temp_segment['start'] // 60)}:{int(temp_segment['start'] % 60):02d}"
            end_time = f"{int(temp_segment['end'] // 60)}:{int(temp_segment['end'] % 60):02d}"
            
            # Форматируем заголовок и текст отдельно
            header = f"[{start_time} - {end_time}] {temp_segment['speaker']} ({temp_segment['emotion']}, {avg_confidence:.1f}%):"
            if temp_segment['emotion'] is None:
                header = f"[{start_time} - {end_time}] {temp_segment['speaker']}:"
            formatted_text = format_segment_text(temp_segment['text'], max_line_length=max_line_length)
            formatted_segment = f"{header}\n{formatted_text}"
            
            merged_segments.append({
                "start": temp_segment["start"],
                "end": temp_segment["end"],
                "text": temp_segment["text"],
                "speaker": temp_segment["speaker"],
                "emotion": temp_segment["emotion"],
                "emotion_confidence": avg_confidence,
                "formatted": formatted_segment
            })
            
            # Начинаем новый временный сегмент
            temp_segment = {
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"].strip(),
                "speaker": speaker,
                "emotion": emotion,
                "confidence_sum": emotion_confidence,
                "segments_count": 1
            }
    
    # Добавляем последний временный сегмент
    if temp_segment is not None:
        avg_confidence = temp_segment["confidence_sum"] / temp_segment["segments_count"]
        start_time = f"{int(temp_segment['start'] // 60)}:{int(temp_segment['start'] % 60):02d}"
        end_time = f"{int(temp_segment['end'] // 60)}:{int(temp_segment['end'] % 60):02d}"
        
        # Форматируем заголовок и текст отдельно
        header = f"[{start_time} - {end_time}] {temp_segment['speaker']} ({temp_segment['emotion']}, {avg_confidence:.1f}%):"
        if temp_segment['emotion'] is None:
            header = f"[{start_time} - {end_time}] {temp_segment['speaker']}:"
        formatted_text = format_segment_text(temp_segment['text'], max_line_length=max_line_length)
        formatted_segment = f"{header}\n{formatted_text}"
        
        merged_segments.append({
            "start": temp_segment["start"],
            "end": temp_segment["end"],
            "text": temp_segment["text"],
            "speaker": temp_segment["speaker"],
            "emotion": temp_segment["emotion"],
            "emotion_confidence": avg_confidence,
            "formatted": formatted_segment
        })
    
    print(f"[INFO] Создано {len(merged_segments)} объединенных сегментов")
    return merged_segments

# Функция для преобразования объединенных сегментов в текстовый формат
def format_transcript(merged_segments, max_line_length=None):
    """
    Форматирует объединенные сегменты в текстовый формат для LLM
    
    Args:
        merged_segments (list): Список объединенных сегментов
        max_line_length (int): Максимальная длина строки (None - без ограничений)
        
    Returns:
        str: Отформатированный текст транскрипции
    """
    return "\n\n".join(segment["formatted"] for segment in merged_segments)