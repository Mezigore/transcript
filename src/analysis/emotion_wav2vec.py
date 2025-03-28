import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from typing import Dict, List, Tuple, Optional

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
    
    def __init__(self, model_name: str = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device: Optional[str] = None):
        self.device = torch.device(device if device else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Словарь для перевода эмоций
        self.emotion_translations = {
            'arousal': 'возбуждение',
            'dominance': 'доминирование',
            'valence': 'валентность'
        }

    def process_audio(self, audio: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
        """Обработка аудио для предсказания эмоций или извлечения эмбеддингов."""
        try:
            # Нормализация сигнала через процессор
            y = self.processor(audio, sampling_rate=sampling_rate)
            y = y['input_values'][0]
            y = y.reshape(1, -1)
            y = torch.from_numpy(y).to(self.device)

            # Запуск через модель
            with torch.no_grad():
                y = self.model(y)[0 if embeddings else 1]

            # Конвертация в numpy
            return y.detach().cpu().numpy()
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            raise

    def analyze_segment(self, audio_segment: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """Анализ эмоций для одного сегмента аудио."""
        try:
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

    def batch_recognize(self, audio_text_pairs: List[Tuple[np.ndarray, str]], sample_rate: int) -> List[Dict[str, float]]:
        """Пакетная обработка аудио сегментов."""
        results = []
        for audio, _ in audio_text_pairs:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            result = self.analyze_segment(audio, sample_rate)
            results.append(result)
        return results 