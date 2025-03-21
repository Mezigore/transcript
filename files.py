import os
import pandas as pd
import ffmpeg
import textwrap
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
from preprocessor import AudioPreprocessor

# Словарь для перевода эмоций с английского на русский
EMOTION_TRANSLATIONS = {
    "neutral": "нейтральность",
    "happiness": "радость",
    "positive": "радость",
    "sadness": "грусть",
    "sad": "грусть",
    "anger": "гнев",
    "angry": "гнев",
    "злость": "гнев",
    "fear": "страх",
    "disgust": "отвращение",
    "surprise": "удивление",
    "enthusiasm": "энтузиазм",
    "disappointment": "разочарование",
    "other": "другое",
    "unknown": "неизвестно"
}


def find_media_files() -> List[str]:
    """
    Находит все аудио и видео файлы в директории input.
    
    Returns:
        List[str]: Список имен аудио и видео файлов
    """
    # Поддерживаемые форматы видео
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    # Поддерживаемые форматы аудио
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    # Все поддерживаемые форматы
    media_extensions = video_extensions + audio_extensions
    
    # Проверяем существование директории input
    input_dir = 'input'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    media_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  any(f.lower().endswith(ext) for ext in media_extensions)]
    return media_files


def select_media_file() -> Optional[str]:
    """
    Позволяет пользователю выбрать аудио или видео файл из директории input.
    
    Returns:
        Optional[str]: Путь к выбранному файлу или None, если файлы не найдены
    """
    media_files = find_media_files()
    
    if not media_files:
        print("[ОШИБКА] Аудио или видео файлы не найдены в директории input.")
        return None
    
    if len(media_files) == 1:
        selected_file = media_files[0]
        print(f"[INFO] Найден медиа файл: {selected_file}")
    else:
        print("[INFO] Найдено несколько медиа файлов:")
        for i, file in enumerate(media_files, 1):
            print(f"  {i}. {file}")
        
        try:
            choice = int(input("[ВВОД] Выберите номер файла (или Enter для первого файла): ") or "1")
            if 1 <= choice <= len(media_files):
                selected_file = media_files[choice - 1]
            else:
                print("[ПРЕДУПРЕЖДЕНИЕ] Неверный выбор. Используется первый файл.")
                selected_file = media_files[0]
        except ValueError:
            print("[ПРЕДУПРЕЖДЕНИЕ] Неверный ввод. Используется первый файл.")
            selected_file = media_files[0]
    
    # Возвращаем полный путь к файлу
    return os.path.join('input', selected_file)


def extract_and_process_audio(input_media: str) -> Optional[str]:
    """
    Извлекает аудио из видеофайла или обрабатывает аудиофайл напрямую.
    
    Args:
        input_media (str): Имя входного аудио или видеофайла
        
    Returns:
        Optional[str]: Путь к обработанному аудиофайлу или None в случае ошибки
    """
    # Получаем имя файла без расширения
    input_name = os.path.splitext(os.path.basename(input_media))[0]
    output_audio = f"audio-input-{input_name}.wav"
    processor = AudioPreprocessor()
    
    # Определяем тип файла по расширению
    file_ext = os.path.splitext(input_media)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    is_audio_file = file_ext in audio_extensions
    
    if is_audio_file:
        print(f"[INFO] Обработка аудиофайла {input_media}...")
    else:
        print(f"[INFO] Извлечение аудио из {input_media}...")
    
    # Создаем прогресс-бар для процесса извлечения/обработки аудио
    with tqdm(total=0, desc="Обработка медиа", bar_format='{desc}: {elapsed}') as pbar:
        try:
            # Для аудиофайлов, которые не в формате WAV 16kHz mono, конвертируем их
            # Для видеофайлов извлекаем аудиодорожку
            (
                ffmpeg
                .input(input_media)
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            
            if is_audio_file:
                pbar.set_description("Обработка аудио завершена")
                print(f"[INFO] Аудио обработано: {output_audio}")
            else:
                pbar.set_description("Извлечение аудио завершено")
                print(f"[INFO] Аудио извлечено: {output_audio}")
                
            output_audio = processor.process_audio(output_audio, visualize=False)
            return output_audio
        except ffmpeg.Error as e:
            if is_audio_file:
                pbar.set_description("Ошибка обработки аудио")
                print(f"[ОШИБКА] Не удалось обработать аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            else:
                pbar.set_description("Ошибка извлечения аудио")
                print(f"[ОШИБКА] Не удалось извлечь аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
        except Exception as e:
            pbar.set_description("Ошибка обработки медиа")
            print(f"[ОШИБКА] Не удалось обработать медиафайл: {e}")
            return None


def create_conversation_file(merged_segments: List[Dict[str, Any]], output_file: str) -> str:
    """
    Создает файл в формате разговора (conversation format) из сегментов транскрипции.
    Формат: [start_time - end_time] Speaker (emotion): text
    
    Args:
        merged_segments (List[Dict[str, Any]]): Список сегментов с транскрипцией
        output_file (str): Имя выходного файла
        
    Returns:
        str: Путь к созданному файлу разговора
    """
    # Проверяем существование директории output
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем только имя файла без пути
    output_filename = os.path.basename(output_file)
    # Формируем полный путь в директории output
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Создание файла в формате разговора: {output_path}")
    
    # Создаем маппинг спикеров на буквы (A, B, C...)
    unique_speakers: Set[str] = set(segment['speaker'] for segment in merged_segments)
    speaker_mapping = {speaker: chr(65 + i) for i, speaker in enumerate(sorted(list(unique_speakers)))}
    
    # Объединяем последовательные сегменты с одинаковым спикером и эмоцией
    combined_segments = []
    current_segment = None
    
    with tqdm(total=len(merged_segments), desc="Объединение сегментов", unit="сегмент") as pbar:
        for segment in merged_segments:
            # Заменяем SPEAKER_XX на соответствующую букву
            segment['speaker'] = speaker_mapping[segment['speaker']]
            
            if current_segment is None:
                current_segment = segment.copy()
            elif (segment['speaker'] == current_segment['speaker'] and 
                  segment.get('emotion', 'unknown') == current_segment.get('emotion', 'unknown')):
                # Объединяем сегменты с тем же спикером и эмоцией
                current_segment['end'] = segment['end']
                current_segment['text'] += " " + segment['text']
                # Если есть уверенность в эмоции, берем среднее значение
                if 'emotion_confidence' in segment and 'emotion_confidence' in current_segment:
                    current_segment['emotion_confidence'] = (current_segment['emotion_confidence'] + segment['emotion_confidence']) / 2
            else:
                combined_segments.append(current_segment)
                current_segment = segment.copy()
            
            pbar.update(1)
    
    # Добавляем последний сегмент
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    # Формируем строки в формате [start_time - end_time] Speaker (emotion): с переносами строк
    conversation_lines = []
    
    with tqdm(total=len(combined_segments), desc="Форматирование сегментов", unit="сегмент") as pbar:
        for segment in combined_segments:
            start_time = f"{float(segment['start']):.2f}"
            end_time = f"{float(segment['end']):.2f}"
            speaker = segment['speaker']
            emotion_eng = segment.get('emotion', 'unknown')
            # Переводим эмоцию на русский язык
            emotion_rus = EMOTION_TRANSLATIONS.get(emotion_eng.lower(), emotion_eng)
            
            # Добавляем процент уверенности, если он доступен
            confidence = segment.get('emotion_confidence', 0)
            if confidence > 0:
                emotion_display = f"{emotion_rus} {confidence:.0f}%"
            else:
                emotion_display = emotion_rus
                
            text = segment['text']
            
            # Создаем заголовок сегмента с переведенной эмоцией и уверенностью
            header = f"[{start_time} - {end_time}] {speaker} ({emotion_display}):"
            
            # Добавляем заголовок как отдельную строку
            conversation_lines.append(header)
            
            # Форматируем текст с переносами строк (ширина 80 символов)
            wrapped_lines = textwrap.wrap(text, width=80, initial_indent='    ', subsequent_indent='    ')
            
            # Добавляем текст с отступом
            conversation_lines.extend(wrapped_lines)
            
            # Добавляем пустую строку после сегмента для лучшей читаемости
            conversation_lines.append("")
            
            pbar.update(1)
    
    # Записываем в файл
    output_path = output_path.replace('.csv', '.txt')
    with tqdm(total=1, desc="Сохранение файла", unit="файл") as pbar:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conversation_lines))
        pbar.update(1)
    
    print(f"[INFO] Файл разговора создан: {output_path}")
    return output_path
