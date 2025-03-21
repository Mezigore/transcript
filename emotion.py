import torch
import ffmpeg
import os
import concurrent.futures
import numpy as np
import time
import torchaudio
import threading
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# Глобальные объекты для кэширования моделей
_emotion_model = None
_feature_extractor = None
# Расширенный словарь эмоций с русскими названиями
_num2emotion = {
    0: 'нейтральный',  # neutral
    1: 'злость',       # angry
    2: 'радость',      # positive
    3: 'грусть',       # sad
    4: 'другое'        # other
}
# Добавляем блокировку для безопасной инициализации модели в многопоточной среде
_model_lock = threading.Lock()

def get_emotion_recognizer():
    """Функция для ленивой инициализации и кэширования модели распознавания эмоций"""
    global _emotion_model, _feature_extractor
    
    # Используем блокировку для безопасного доступа к глобальным переменным
    with _model_lock:
        if _emotion_model is None or _feature_extractor is None:
            try:
                # Проверяем доступность CUDA, затем MPS, иначе используем CPU
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available() and torch.backends.mps.is_macos13_or_newer():
                    device = 'mps'
                else:
                    device = 'cpu'
                
                print(f"[INFO] Инициализация модели анализа эмоций (устройство: {device})")
                
                _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
                _emotion_model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
                
                # Перемещаем модель на нужное устройство
                _emotion_model = _emotion_model.to(device)
                
            except Exception as e:
                print(f"[ОШИБКА] Не удалось инициализировать модель анализа эмоций: {e}")
                return None, None
    
    return _emotion_model, _feature_extractor

# Функция для анализа эмоций в одном сегменте
def analyze_segment_emotion(segment_data):
    """Анализирует эмоции для одного аудио сегмента"""
    segment_path, segment = segment_data
    
    # Максимальное количество попыток анализа
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Получаем кэшированные модели
            model, feature_extractor = get_emotion_recognizer()
            
            # Если модель не инициализирована, возвращаем сегмент без эмоции
            if model is None or feature_extractor is None:
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Модель не инициализирована, пропускаем анализ эмоций")
                segment_with_emotion = segment.copy()
                segment_with_emotion["emotion"] = "нейтральный"  # Используем нейтральную эмоцию по умолчанию
                segment_with_emotion["emotion_confidence"] = 0.0  # Уверенность 0%
                return segment_with_emotion
            
            # Загружаем и подготавливаем аудио
            waveform, sample_rate = torchaudio.load(segment_path, normalize=True)
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)
            
            # Подготавливаем входные данные для модели
            inputs = feature_extractor(
                waveform, 
                sampling_rate=feature_extractor.sampling_rate, 
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,  # Максимальная длина 10 секунд
                truncation=True
            )
            
            # Перемещаем входные данные на то же устройство, что и модель
            device = next(model.parameters()).device
            input_values = inputs['input_values'][0].to(device)
            
            # Получаем предсказание модели
            with torch.no_grad():
                outputs = model(input_values)
                logits = outputs.logits
                
                # Применяем softmax для получения вероятностей
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Получаем индекс максимальной вероятности и саму вероятность
                max_prob, predictions = torch.max(probs, dim=-1)
                predicted_emotion = _num2emotion[predictions.cpu().numpy()[0]]
                confidence = max_prob.cpu().numpy()[0] * 100  # Переводим в проценты
            
            # Создаем копию сегмента с добавленной эмоцией и уверенностью
            segment_with_emotion = segment.copy()
            segment_with_emotion["emotion"] = predicted_emotion
            segment_with_emotion["emotion_confidence"] = confidence
            
            print(f"[INFO] Спикер {segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s, эмоция: {predicted_emotion} ({confidence:.1f}%)")
            return segment_with_emotion
                
        except Exception as e:
            print(f"[ОШИБКА] Ошибка при анализе эмоций: {e}")
            retry_count += 1
            
            if retry_count <= max_retries:
                print(f"[INFO] Повторная попытка {retry_count}/{max_retries}...")
                time.sleep(1)  # Небольшая пауза перед повторной попыткой
            else:
                # Если все попытки исчерпаны, возвращаем сегмент с нейтральной эмоцией
                segment_with_emotion = segment.copy()
                segment_with_emotion["emotion"] = "нейтральный"
                segment_with_emotion["emotion_confidence"] = 0.0  # Уверенность 0%
                return segment_with_emotion
    
    # Этот код не должен выполняться, но на всякий случай
    segment_with_emotion = segment.copy()
    segment_with_emotion["emotion"] = "нейтральный"
    segment_with_emotion["emotion_confidence"] = 0.0  # Уверенность 0%
    return segment_with_emotion

# Функция для анализа эмоций
def analyze_emotions(audio_path, speaker_segments):
    print("[INFO] Начало анализа эмоций...")
    
    # Минимальная длительность сегмента для надежного анализа эмоций (в секундах)
    MIN_SEGMENT_DURATION = 5.0
    
    # Объединяем короткие сегменты от одного спикера
    merged_segments = []
    current_segment = None
    
    for segment in speaker_segments:
        if current_segment is None:
            current_segment = segment.copy()
        elif segment['speaker'] == current_segment['speaker'] and segment['start'] - current_segment['end'] < 1.0:
            # Объединяем сегменты одного спикера, если они близко друг к другу
            current_segment['end'] = segment['end']
        else:
            merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    # Отфильтровываем слишком короткие сегменты
    filtered_segments = [s for s in merged_segments if (s['end'] - s['start']) >= MIN_SEGMENT_DURATION]
    
    if len(filtered_segments) < len(merged_segments):
        print(f"[INFO] Пропущено {len(merged_segments) - len(filtered_segments)} сегментов короче {MIN_SEGMENT_DURATION} сек")
    
    # Создаем временную директорию для сегментов
    os.makedirs("temp_segments", exist_ok=True)
    
    # Подготавливаем задачи для параллельной обработки
    segment_tasks = []
    
    # Добавляем прогресс-бар для подготовки сегментов
    print("[INFO] Подготовка сегментов для анализа...")
    for i, segment in tqdm(enumerate(filtered_segments), total=len(filtered_segments), desc="Подготовка сегментов"):
        start_time = segment["start"]
        duration = segment["end"] - segment["start"]
        
        try:
            # Если длина сегмента больше 10 секунд, то делим его на части
            if duration > 10:
                segments_parts = []
                # Вычисляем количество полных 10-секундных сегментов
                full_segments_count = int(duration // 10)
                # Вычисляем остаток
                remainder = duration % 10
                
                # Если остаток меньше MIN_SEGMENT_DURATION и у нас есть хотя бы один полный сегмент,
                # то последний полный сегмент будет длиннее на этот остаток
                if remainder < MIN_SEGMENT_DURATION and full_segments_count > 0:
                    for x in range(full_segments_count - 1):
                        segment_path = f"temp_segments/segment_{i}_part_{x}.wav"
                        (
                            ffmpeg
                            .input(audio_path)
                            .output(segment_path, ss=start_time + x * 10, t=10, acodec='pcm_s16le', ac=1, ar='16k')
                            .run(quiet=True, overwrite_output=True)
                        )
                        segments_parts.append((segment_path, segment))
                    
                    # Последний сегмент с остатком
                    last_segment_duration = 10 + remainder
                    segment_path = f"temp_segments/segment_{i}_part_{full_segments_count - 1}.wav"
                    (
                        ffmpeg
                        .input(audio_path)
                        .output(segment_path, ss=start_time + (full_segments_count - 1) * 10, t=last_segment_duration, acodec='pcm_s16le', ac=1, ar='16k')
                        .run(quiet=True, overwrite_output=True)
                    )
                    segments_parts.append((segment_path, segment))
                else:
                    # Иначе обрабатываем все сегменты как обычно
                    for x in range(full_segments_count):
                        segment_path = f"temp_segments/segment_{i}_part_{x}.wav"
                        (
                            ffmpeg
                            .input(audio_path)
                            .output(segment_path, ss=start_time + x * 10, t=10, acodec='pcm_s16le', ac=1, ar='16k')
                            .run(quiet=True, overwrite_output=True)
                        )
                        segments_parts.append((segment_path, segment))
                    
                    # Если есть остаток и он >= MIN_SEGMENT_DURATION, создаем дополнительный сегмент
                    if remainder >= MIN_SEGMENT_DURATION:
                        segment_path = f"temp_segments/segment_{i}_part_{full_segments_count}.wav"
                        (
                            ffmpeg
                            .input(audio_path)
                            .output(segment_path, ss=start_time + full_segments_count * 10, t=remainder, acodec='pcm_s16le', ac=1, ar='16k')
                            .run(quiet=True, overwrite_output=True)
                        )
                        segments_parts.append((segment_path, segment))
                
                # Добавляем все части сегмента в задачи
                segment_tasks.extend(segments_parts)
            else:
                segment_path = f"temp_segments/segment_{i}.wav"
                (
                    ffmpeg
                    .input(audio_path)
                    .output(segment_path, ss=start_time, t=duration, acodec='pcm_s16le', ac=1, ar='16k')
                    .run(quiet=True, overwrite_output=True)
                )
                segment_tasks.append((segment_path, segment))
                
        except Exception as e:
            print(f"[ОШИБКА] Не удалось подготовить сегмент {i}: {e}")
    
    print(f"[INFO] Подготовлено {len(segment_tasks)} сегментов для анализа")
    
    # Предварительно инициализируем модель до запуска многопоточной обработки
    # Это гарантирует, что модель будет загружена только один раз
    print("[INFO] Предварительная инициализация модели анализа эмоций...")
    get_emotion_recognizer()
    
    # Обработка сегментов в многопоточном режиме с прогресс-баром
    segments_with_emotions = []
    
    # Определяем максимальное количество потоков
    max_workers = min(os.cpu_count() or 4, 8)  # Не более 8 потоков
    
    print(f"[INFO] Запуск анализа эмоций в {max_workers} потоках...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем задачи и создаем прогресс-бар
        futures = {executor.submit(analyze_segment_emotion, task): task for task in segment_tasks}
        
        # Используем tqdm для отображения прогресса выполнения задач
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Анализ эмоций"):
            try:
                segment_with_emotion = future.result()
                if segment_with_emotion:
                    segments_with_emotions.append(segment_with_emotion)
            except Exception as e:
                print(f"[ОШИБКА] Ошибка при обработке сегмента: {e}")
    
    # Сортировка результатов по времени начала
    segments_with_emotions.sort(key=lambda x: x["start"])
    
    # Очистка временных файлов
    print("[INFO] Очистка временных файлов...")
    for segment_path, _ in segment_tasks:
        try:
            if os.path.exists(segment_path):
                os.remove(segment_path)
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить файл {segment_path}: {e}")
    
    print(f"[INFO] Анализ эмоций завершен для {len(segments_with_emotions)} сегментов")
    return segments_with_emotions
