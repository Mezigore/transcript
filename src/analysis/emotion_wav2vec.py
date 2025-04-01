import numpy as np
from pyparsing import Any
import torch
import torch.nn as nn
import logging
import tqdm
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from typing import Dict, List, Tuple, Optional, Union

# Настройка логирования
logger = logging.getLogger(__name__)

class RegressionHead(nn.Module):
    """Классификационная головка для эмоций."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    """Классификатор эмоций на основе Wav2Vec2."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

class Wav2VecEmotionRecognizer:
    """Класс для распознавания эмоций с использованием Wav2Vec2."""
    
    def __init__(self, model_name: str = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device_name: Optional[str] = None):
        """
        Инициализирует распознаватель эмоций.
        
        Args:
            model_name: Название модели или путь к ней
            device_name: Имя устройства для вычислений ("cuda" или "cpu")
        """
        # Определяем устройство для вычислений
        if device_name is None:
            device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device: Any = torch.device(device_name)
        
        # Инициализируем извлекатель признаков
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Загружаем модель
        logger.info(f"Загрузка модели {model_name} на устройство {device_name}")
        self.model = EmotionModel.from_pretrained(model_name)
        # Игнорируем предупреждение линтера - это стандартный код PyTorch
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Словарь для перевода эмоций
        self.emotion_translations = {
            'arousal': 'возбуждение',
            'dominance': 'доминирование',
            'valence': 'валентность'
        }

    def process_audio(self, audio: Union[np.ndarray, torch.Tensor], sampling_rate: int, embeddings: bool = False) -> np.ndarray:
        """
        Обработка аудио для предсказания эмоций или извлечения эмбеддингов.
        
        Args:
            audio: Аудио-данные как numpy массив или PyTorch тензор
            sampling_rate: Частота дискретизации аудио
            embeddings: Если True, возвращает эмбеддинги вместо предсказаний эмоций
            
        Returns:
            np.ndarray: Массив с эмоциями или эмбеддингами
        """
        try:
            # Проверяем длину аудио - минимально допустимая для wav2vec - 2 сэмпла
            if isinstance(audio, np.ndarray) and len(audio) < 2:
                logger.error(f"Аудиосегмент слишком короткий для обработки: {len(audio)} сэмплов")
                return np.array([])
                
            # Нормализация сигнала через feature extractor
            inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)

            # Запуск через модель
            with torch.no_grad():
                output = self.model(input_values)
                y = output[0] if embeddings else output[1]

            # Конвертация в numpy
            return y.detach().cpu().numpy()
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            raise

    def analyze_segment(self, audio_segment: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> Dict[str, float]:
        """
        Анализ эмоций для одного сегмента аудио.
        
        Args:
            audio_segment: Аудио-сегмент как numpy массив или PyTorch тензор
            sampling_rate: Частота дискретизации аудио
            
        Returns:
            Dict[str, float]: Словарь с эмоциями и их значениями
        """
        try:
            # Проверка минимальной длины аудио (минимум 2 секунды)
            min_samples = 2 * sampling_rate
            
            # Проверяем длину в зависимости от типа данных
            if isinstance(audio_segment, np.ndarray) and len(audio_segment) < min_samples:
                logger.warning(f"Аудиосегмент слишком короткий для обработки wav2vec: {len(audio_segment)/sampling_rate:.2f}с")
                return {}
                
            # Если это тензор PyTorch
            if isinstance(audio_segment, torch.Tensor) and audio_segment.dim() > 0:
                if audio_segment.shape[-1] < min_samples:
                    logger.warning(f"Аудиосегмент-тензор слишком короткий: {audio_segment.shape[-1]/sampling_rate:.2f}с")
                    return {}
            
            # Обрабатываем аудио
            results = self.process_audio(audio_segment, sampling_rate)
            if results.size == 0:
                return {}
                
            # Преобразование результатов в словарь эмоций
            emotions = {
                'arousal': float(results[0][0]),
                'dominance': float(results[0][1]),
                'valence': float(results[0][2])
            }
            
            # Перевод названий эмоций
            return {self.emotion_translations.get(k, k): v for k, v in emotions.items()}
        except Exception as e:
            logger.error(f"Ошибка при анализе сегмента: {e}")
            return {}

    def batch_recognize(self, audio_text_pairs: List[Tuple[Union[np.ndarray, torch.Tensor], str]], sample_rate: int) -> List[Dict[str, float]]:
        """
        Пакетная обработка аудио сегментов.
        
        Args:
            audio_text_pairs: Список пар (аудио-сегмент, текст)
            sample_rate: Частота дискретизации аудио
            
        Returns:
            List[Dict[str, float]]: Список словарей с эмоциями для каждого сегмента
        """
        results = []
        # Добавляем индикатор прогресса для более детальной информации
        with tqdm.tqdm(total=len(audio_text_pairs), desc="Анализ WAV2VEC сегментов", leave=False) as segment_progress:
            for audio, _ in audio_text_pairs:
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                result = self.analyze_segment(audio, sample_rate)
                results.append(result)
                segment_progress.update(1)
        return results 