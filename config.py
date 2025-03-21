"""
Конфигурационный файл для системы транскрипции и анализа эмоций.
Содержит все настройки и параметры для различных модулей системы.
"""

# Общие настройки
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
TEMP_DIR = 'temp_segments'  # Директория для временных аудио сегментов

# Настройки для транскрипции (ts.py)
TRANSCRIPTION = {
    'model_path': "mlx-community/whisper-large-v3-mlx",  # Путь к модели Whisper
    'segment_size': 30.0,  # Размер сегмента в секундах для транскрипции
    'initial_prompt': "Запись интервью с представителем строительной сферы"  # Начальный промпт для Whisper
}

# Настройки для диаризации (diarization.py)
DIARIZATION = {
    'model_path': "pyannote/speaker-diarization-3.1",  # Путь к модели диаризации
    'segment_size': 30.0,  # Размер сегмента в секундах для диаризации
    'min_speakers': 2,  # Минимальное количество спикеров
    'max_speakers': 3   # Максимальное количество спикеров
}

# Настройки для анализа эмоций (emotion.py)
EMOTION = {
    'feature_extractor': "facebook/hubert-large-ls960-ft",  # Путь к экстрактору признаков
    'model_path': "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned",  # Путь к модели эмоций
    'min_segment_duration': 5.0,  # Минимальная длительность сегмента для анализа эмоций (в секундах)
    'max_segment_duration': 10.0,  # Максимальная длительность сегмента для анализа эмоций (в секундах)
    'max_retries': 2,  # Максимальное количество попыток анализа
    'emotion_mapping': {  # Словарь для перевода эмоций
        0: 'нейтральный',  # neutral
        1: 'злость',       # angry
        2: 'радость',      # positive
        3: 'грусть',       # sad
        4: 'другое'        # other
    }
}

# Настройки для препроцессора аудио (preprocessor.py)
AUDIO_PREPROCESSOR = {
    'target_sr': 16000,  # Целевая частота дискретизации
    'mono': True,  # Преобразовывать ли в моно
    'silence_removal': {
        'enabled': False,  # Включить удаление тишины
        'top_db': 20  # Порог в дБ для определения тишины
    },
    'highpass_filter': {
        'enabled': True,  # Включить фильтр высоких частот
        'cutoff': 80  # Частота среза в Гц
    },
    'noise_reduction': {
        'enabled': False  # Включить подавление шума
    },
    'normalization': {
        'enabled': True  # Включить нормализацию амплитуды
    },
    'visualization': {
        'enabled': False  # Включить визуализацию аудио
    }
}

# Настройки для обработки файлов (files.py)
FILES = {
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'],
    'audio_extensions': ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma'],
    'emotion_translations': {
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
}

# Настройки для API ключей
API_KEYS = {
    'huggingface': "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"  # Токен Hugging Face
}

# Настройки для форматирования вывода
OUTPUT_FORMAT = {
    'conversation_format': True,  # Использовать формат разговора
    'wrap_width': 100,  # Ширина строки для переноса текста
    'include_emotions': True,  # Включать эмоции в вывод
    'include_confidence': False,  # Включать уверенность в вывод
    'min_confidence_threshold': 30.0  # Минимальный порог уверенности для отображения эмоции (в процентах)
}
