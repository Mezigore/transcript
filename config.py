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
    'model_path': "mlx-community/whisper-large-v3-turbo",  # Путь к модели Whisper
    'segment_size': 30.0,  # Увеличиваем размер сегмента для лучшего контекста в онлайн-встречах
    'initial_prompt': "Запись онлайн-встречи в Zoom/Google Meet с несколькими участниками. Технические термины, профессиональная лексика. Строительство, инфраструктура, проекты, проектирование",  # Специализированный промпт
    'decode_options': {
        'language': 'ru',
        'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Детерминированные результаты
        'best_of': 7,  # Уменьшаем для быстродействия
        'length_penalty': 1.0,  # Нейтральный штраф для длинных последовательностей
        'fp16': True,  # Использовать полуплавающую точность для ускорения
        'without_timestamps': False,  # Сохраняем временные метки для синхронизации
        'compression_ratio_threshold': 2.0,  # Оптимизировано для речи в онлайн-встречах
        'logprob_threshold': -0.8,  # Повышаем порог для лучшего качества
        'no_speech_threshold': 1.0,  # Снижаем порог для лучшей детекции речи в онлайн-встречах
    }
}

# Настройки для диаризации (diarization.py)
DIARIZATION = {
    'model_path': "pyannote/speaker-diarization-3.1",  # Путь к модели диаризации
    'segment_size': 30.0,  # Соответствует размеру сегмента для транскрипции
    'min_speakers': 2,  # Минимальное количество спикеров
    'max_speakers': 3,  # Оптимизировано для типичных онлайн-встреч
    'num_speakers': None,
    'diarization_params': {
        'threshold': 0.4,  # Снижаем порог для лучшего разделения спикеров в онлайн-встречах
        'min_duration_on': 0.3,  # Оптимизировано для речи в онлайн-встречах
        'min_duration_off': 0.1,  # Оптимизировано для быстрых переключений между говорящими
        'filter_speech': True,  # Фильтрация речи
        'speech_threshold': 0.4,  # Снижаем порог для лучшей детекции речи в онлайн-встречах
        'speech_duration': 0.4,  # Оптимизировано для речи в онлайн-встречах
        'speech_min_duration': 0.1,  # Оптимизировано для коротких реплик
        'speech_max_duration': 15.0  # Увеличиваем для длинных монологов в презентациях
    }
}

# Настройки для препроцессора аудио (preprocessor.py)
AUDIO_PREPROCESSOR = {
    'target_sr': 16000,  # Целевая частота дискретизации (оптимально для Whisper)
    'mono': True,  # Преобразовывать в моно (снижает шум)
    'silence_removal': {
        'enabled': True,  # Включить удаление тишины
        'top_db': 35  # Повышаем порог для лучшего определения тишины в онлайн-встречах
    },
    'highpass_filter': {
        'enabled': True,  # Включить фильтр высоких частот
        'cutoff': 120  # Повышаем частоту среза для удаления фонового шума онлайн-встреч
    },
    'noise_reduction': {
        'enabled': True,  # Включить подавление шума
        'nr_type': 'spectral',  # Использовать спектральное подавление шума
        'nr_strength': 0.3  # Увеличиваем силу подавления для онлайн-встреч
    },
    'normalization': {
        'enabled': True,  # Включить нормализацию амплитуды
        'target_level': -23  # Оптимизированный уровень громкости для речи в онлайн-встречах
    },
    'zoom_meet_optimization': {
        'enabled': True,  # Включить специальные оптимизации для Zoom/Meet
        'echo_reduction': True,  # Уменьшение эха
        'background_noise_suppression': True,  # Подавление фонового шума
        'voice_enhancement': True  # Улучшение разборчивости речи
    },
    'visualization': {
        'enabled': False  # Визуализация аудиофайла 
    }
}

# Настройки для обработки файлов
FILES = {
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'],
    'audio_extensions': ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma'],
    'emotion_translations': {
        'anger': 'гнев',
        'disgust': 'отвращение',
        'enthusiasm': 'энтузиазм',
        'fear': 'страх',
        'happiness': 'счастье',
        'neutral': 'нейтральность',
        'sadness': 'грусть'
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
    'include_confidence': True,  # Включать уверенность в вывод
    'min_confidence_threshold': 30.0  # Минимальный порог уверенности для отображения эмоции (в процентах)
}
