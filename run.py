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
    
    # Выбор файла
    media_path = select_media_file()
    if not media_path:
        print("[ОШИБКА] Файл не выбран. Поместите медиафайлы в директорию 'input'.")
        exit(1)
    
    # Обработка медиафайла
    result = process_media(media_path)
    if result["success"]:
        print(f"[INFO] Processing completed. Result saved to: {result['file_path']}")
        print(f"[INFO] Full path: {os.path.abspath(str(result['file_path']))}")
    else:
        print(f"[ERROR] Error processing media file: {result['error']}")
