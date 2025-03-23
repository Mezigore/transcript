import os
import shutil
from src.files import *
from src.transcribe import transcribe
from src.diarization import diarize
from src.emotion import analyze_emotions
from config import API_KEYS, TEMP_DIR
from src.combine import merge_transcription_with_diarization
from typing import Dict, Union

# Основная функция
def process_media(media_path: str, hf_token: Union[str, None] = None) -> Dict[str, Union[bool, str]]:
    # Validate input path
    if not os.path.exists(media_path):
        return {"success": False, "error": "[ERROR] Media file does not exist"}
    
    # Если токен не передан, используем из конфигурации
    if hf_token is None:
        hf_token = API_KEYS['huggingface']
    
    # Извлечение аудио
    audio_result = extract_and_process_audio(media_path)
    audio_path = audio_result
    
    if audio_path is None:
        return {"success": False, "error": "[ОШИБКА] Не удалось извлечь аудио"}
    
    # Диаризация
    try:
        diarized_segments = diarize(audio_path, hf_token)
    except Exception as e:
        return {"success": False, "error": f"[ERROR] Diarization failed: {str(e)}"}
    
    # Анализ эмоций
    try:
        text_and_emotion_segments = analyze_emotions(audio_path)
    except Exception as e:
        return {"success": False, "error": f"[ERROR] Emotion analysis failed: {str(e)}"}
    
    # Объединение результатов
    merged_segments = merge_transcription_with_diarization(diarized_segments, text_and_emotion_segments, max_line_length=100)
    
    # Базовое имя для выходных файлов
    base_output = os.path.splitext(os.path.basename(media_path))[0]
    output_conversation = f"{base_output}_transcript.txt"
    
    # Создание файла в формате разговора (основной формат для LLM)
    conversation_file = create_conversation_file(merged_segments, output_conversation, max_line_length=100)
    
    # Очистка временных файлов
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    return {"success": True, "file_path": conversation_file}

if __name__ == "__main__":
    # Загрузка токена из конфигурации
    hf_token = API_KEYS['huggingface']
    
    # Выбор режима работы
    mode = select_operation_mode()
    
    if mode == 1:
        # Режим обработки одного файла
        media_path = select_media_file()
        
        if not media_path:
            print("[ОШИБКА] Файл не выбран. Поместите медиафайлы в директорию 'input'.")
            exit(1)
        
        # Проверка, был ли файл уже обработан
        if check_if_processed(media_path):
            should_process = input("[ВВОД] Этот файл уже был обработан. Обработать снова? (y/n, по умолчанию n): ").lower() == 'y'
            if not should_process:
                print("[INFO] Обработка файла отменена пользователем.")
                exit(0)
        
        # Обработка медиафайла
        result = process_media(media_path)
        
        if result["success"]:
            print(f"[INFO] Обработка завершена. Результат сохранен в: {result['file_path']}")
            print(f"[INFO] Полный путь: {os.path.abspath(str(result['file_path']))}")
        else:
            print(f"[ERROR] Ошибка обработки медиафайла: {result['error']}")
    
    else:
        # Режим обработки всех файлов
        media_files = find_media_files()
        
        if not media_files:
            print("[ОШИБКА] Медиафайлы не найдены в директории input.")
            exit(1)
        
        # Отображаем информацию о всех файлах
        print("[INFO] Найдены следующие медиафайлы:")
        for i, file in enumerate(media_files, 1):
            full_path = os.path.join(INPUT_DIR, file)
            is_processed = check_if_processed(full_path)
            status = " [обработан]" if is_processed else ""
            print(f" {i}. {file}{status}")
        
        # Предлагаем пропустить уже обработанные файлы
        skip_processed = input("[ВВОД] Пропустить уже обработанные файлы? (y/n, по умолчанию y): ").lower() != 'n'
        
        # Счетчики для статистики
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        # Обрабатываем каждый файл
        for file in media_files:
            full_path = os.path.join(INPUT_DIR, file)
            is_processed = check_if_processed(full_path)
            
            # Пропускаем уже обработанные файлы, если пользователь указал это
            if skip_processed and is_processed:
                print(f"[INFO] Пропуск уже обработанного файла: {file}")
                skipped_count += 1
                continue
            
            print(f"\n[INFO] Обработка файла: {file}")
            
            # Обработка медиафайла
            result = process_media(full_path)
            
            if result["success"]:
                print(f"[INFO] Обработка завершена. Результат сохранен в: {result['file_path']}")
                processed_count += 1
            else:
                print(f"[ERROR] Ошибка обработки медиафайла: {result['error']}")
                error_count += 1
        
        # Отображаем итоговую статистику
        print(f"\n[INFO] Итоговая статистика обработки:")
        print(f" - Всего файлов: {len(media_files)}")
        print(f" - Успешно обработано: {processed_count}")
        print(f" - Пропущено (уже обработанные): {skipped_count}")
        print(f" - Ошибки обработки: {error_count}")
