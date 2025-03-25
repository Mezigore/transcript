import os
import ffmpeg
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from config import FILES, INPUT_DIR, OUTPUT_DIR

# Словарь для перевода эмоций с английского на русский
EMOTION_TRANSLATIONS = FILES['emotion_translations']

def find_media_files() -> List[str]:
    """
    Находит все аудио и видео файлы в директории input.
    
    Returns:
        List[str]: Список имен аудио и видео файлов
    """
    video_extensions = FILES['video_extensions']
    audio_extensions = FILES['audio_extensions']
    media_extensions = video_extensions + audio_extensions
    
    input_dir = INPUT_DIR
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    media_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and
                  any(f.lower().endswith(ext) for ext in media_extensions)]
    
    return media_files

def check_if_processed(media_path: str) -> bool:
    """
    Проверяет, был ли файл уже обработан (есть ли соответствующий выходной файл).
    
    Args:
        media_path (str): Путь к медиа файлу
        
    Returns:
        bool: True, если файл уже обработан, False в противном случае
    """
    base_output = os.path.splitext(os.path.basename(media_path))[0]
    output_conversation = f"{base_output}_transcript.txt"
    output_path = os.path.join(OUTPUT_DIR, output_conversation)
    
    return os.path.exists(output_path)

def select_operation_mode() -> int:
    """
    Позволяет пользователю выбрать режим работы программы.
    
    Returns:
        int: 1 - обработка одного файла, 2 - обработка всех файлов
    """
    print("\n[МЕНЮ] Выберите режим работы:")
    print(" 1. Обработать один файл")
    print(" 2. Обработать все файлы в папке input")
    
    try:
        choice = int(input("[ВВОД] Введите номер выбранного режима (или Enter для режима 1): ") or "1")
        if choice not in [1, 2]:
            print("[ПРЕДУПРЕЖДЕНИЕ] Неверный выбор. Используется режим 1.")
            return 1
        return choice
    except ValueError:
        print("[ПРЕДУПРЕЖДЕНИЕ] Неверный ввод. Используется режим 1.")
        return 1

def select_media_file() -> Optional[str]:
    """
    Позволяет пользователю выбрать аудио или видео файл из директории input.
    Отмечает уже обработанные файлы.
    
    Returns:
        Optional[str]: Путь к выбранному файлу или None, если файлы не найдены
    """
    media_files = find_media_files()
    
    if not media_files:
        print("[ОШИБКА] Аудио или видео файлы не найдены в директории input.")
        return None
    
    if len(media_files) == 1:
        selected_file = media_files[0]
        is_processed = check_if_processed(os.path.join(INPUT_DIR, selected_file))
        status = " [обработан]" if is_processed else ""
        print(f"[INFO] Найден медиа файл: {selected_file}{status}")
    else:
        print("[INFO] Найдено несколько медиа файлов:")
        for i, file in enumerate(media_files, 1):
            is_processed = check_if_processed(os.path.join(INPUT_DIR, file))
            status = " [обработан]" if is_processed else ""
            print(f" {i}. {file}{status}")
            
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
    
    return os.path.join(INPUT_DIR, selected_file)

def extract_and_process_audio(input_media: str) -> Optional[str]:
    """
    Извлекает аудио из видеофайла или обрабатывает аудиофайл напрямую.
    
    Args:
        input_media (str): Имя входного аудио или видеофайла
        
    Returns:
        Optional[str]: Путь к обработанному аудиофайлу или None в случае ошибки
    """
    input_name = os.path.splitext(os.path.basename(input_media))[0]
    output_audio = f"audio-input-{input_name}.wav"
    
    file_ext = os.path.splitext(input_media)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    is_audio_file = file_ext in audio_extensions
    
    if is_audio_file:
        print(f"[INFO] Обработка аудиофайла {input_media}...")
    else:
        print(f"[INFO] Извлечение аудио из {input_media}...")
    
    with tqdm(total=0, desc="Обработка медиа", bar_format='{desc}: {elapsed}') as pbar:
        try:
            ffmpeg\
                .input(input_media)\
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')\
                .run(quiet=True, overwrite_output=True)
            
            if is_audio_file:
                pbar.set_description("Обработка аудио завершена")
            else:
                pbar.set_description("Извлечение аудио завершено")
            
            print(f"[INFO] Аудио {'обработано' if is_audio_file else 'извлечено'}: {output_audio}")
            
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

def create_conversation_file(merged_segments: List[Dict[str, Any]], output_file: str, max_line_length: int = 100) -> str:
    """
    Создает файл в формате разговора с учетом приоритетов разделения блоков.
    Формат: время - говорящий - эмоция и процент уверенности - текст
    
    Args:
        merged_segments (List[Dict[str, Any]]): Список сегментов с транскрипцией
        output_file (str): Имя выходного файла
        max_line_length (int): Максимальная длина строки для переноса текста
        
    Returns:
        str: Путь к созданному файлу разговора
    """
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.basename(output_file)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Создание файла в формате разговора: {output_path}")
    
    sorted_segments = sorted(merged_segments, key=lambda x: x['start'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        from src.combine import format_transcript
        formatted_text = format_transcript(sorted_segments, max_line_length=max_line_length)
        f.write(formatted_text)
    
    print(f"[INFO] Файл в формате разговора создан: {output_path}")
    
    return output_path
