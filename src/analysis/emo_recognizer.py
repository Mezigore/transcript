from importlib.machinery import ModuleSpec
import torch
import os
import importlib.util
import sys
import logging
from transformers import AutoTokenizer, AutoFeatureExtractor
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from contextlib import nullcontext
import tqdm
import torchaudio

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set PyTorch CUDA memory configuration
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_custom_model(model_name):
    """Загружает кастомную модель с оптимизациями"""
    try:
        from huggingface_hub import snapshot_download
        
        # Скачивание модели с кешированием
        model_path = snapshot_download(repo_id=model_name)
        
        # Загрузка кастомного модуля
        custom_module_path = os.path.join(model_path, "audio_text_multimodal.py")
        if not os.path.exists(custom_module_path):
            raise FileNotFoundError(f"Кастомный модуль не найден по пути {custom_module_path}")
            
        spec: ModuleSpec | None = importlib.util.spec_from_file_location("audio_text_multimodal", custom_module_path)
        if spec is None:
            raise ImportError(f"Ошибка загрузки спецификации модуля из {custom_module_path}")
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules["audio_text_multimodal"] = custom_module
        if spec.loader is None:
            raise ImportError(f"Ошибка загрузки загрузчика модуля из {custom_module_path}")
        spec.loader.exec_module(custom_module)
        
        # Получение классов из модуля
        WavLMBertConfig = custom_module.WavLMBertConfig
        WavLMBertForSequenceClassification = custom_module.WavLMBertForSequenceClassification
        
        # Загрузка конфигурации
        config = WavLMBertConfig.from_pretrained(model_path)
        
        # Загрузка модели
        model = WavLMBertForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            local_files_only=True
        )
        
        # Валидация модели
        if not hasattr(model, 'config') or not hasattr(model.config, 'id2label'):
            raise ValueError("Не удалось загрузить конфигурацию модели с эмоциональными метками")
            
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки кастомной модели: {e}")
        raise

class EmotionDataset(Dataset):
    def __init__(self, audio_text_pairs, processor, feature_extractor, sampling_rate: int):
        self.pairs = audio_text_pairs
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio, text = self.pairs[idx]
        
        # Добавлена обработка ошибок и проверка типов
        try:
            # Преобразование аудио
            if torch.is_tensor(audio):
                audio_numpy = audio.cpu().numpy()
            else:
                audio_numpy = audio
                
            audio_inputs = self.feature_extractor(
                audio_numpy,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Обработка текста
            text_inputs = self.processor(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Исправлена закрывающая скобка и добавлена проверка на token_type_ids
            result = {
                "input_values": audio_inputs["input_values"].squeeze(),
                "audio_attention_mask": audio_inputs["attention_mask"].squeeze().float(),
                "input_ids": text_inputs["input_ids"].squeeze(),
                "text_attention_mask": text_inputs["attention_mask"].squeeze().float()
            }
            
            if "token_type_ids" in text_inputs:
                result["token_type_ids"] = text_inputs["token_type_ids"].squeeze()
                
            return result
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise RuntimeError(f"Failed to process item {idx}: {e}")

class EmotionRecognizer:
    def __init__(self, model_name="aniemore/wavlm-bert-fusion-s-emotion-russian-resd", device=None):
        # Определение устройства с приоритетом CPU для избежания проблем с памятью
        self.device = torch.device("cpu") if device == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
        
        # Инициализация токенизатора и моделей
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = load_custom_model(model_name)
        
        # Безопасная квантизация
        if self.device.type == "cpu":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear, torch.nn.Conv1d},
                    dtype=torch.qint8
                )
                logger.info("Модель успешно квантизирована")
            except Exception as e:
                logger.warning(f"Ошибка квантизации: {e}")
        
        # Перенос модели на устройство
        self.model = self.model.to(self.device)
        self.model.eval()

    def batch_recognize(self, audio_text_pairs, sample_rate: int, use_fp16: bool = False, batch_size: int = 1) -> List[Dict]:
        # Проверка входных данных
        if not audio_text_pairs:
            logger.warning("Получен пустой список audio_text_pairs")
            return []
            
        # Ограничение размера батча
        batch_size = max(1, min(batch_size, 4))
        
        with torch.no_grad():
            try:
                dataset = EmotionDataset(
                    audio_text_pairs,
                    self.tokenizer,
                    self.feature_extractor,
                    sample_rate
                )
                
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False
                )
                
                results = []
                
                with tqdm.tqdm(total=len(dataloader), desc="Анализ эмоций") as pbar:
                    for batch in dataloader:
                        try:
                            # Перенос данных на устройство
                            batch = {k: (v.to(self.device) if v is not None and torch.is_tensor(v) else v) 
                                   for k, v in batch.items()}
                            
                            # Контекстный менеджер для mixed precision
                            context_mgr = torch.cuda.amp.autocast() if use_fp16 and self.device.type == 'cuda' else nullcontext()
                            
                            with context_mgr:
                                outputs = self.model(**{k: v for k, v in batch.items() if v is not None})
                                probs = torch.softmax(outputs.logits, dim=1)
                                
                                for prob in probs:
                                    result = {}
                                    for i, p in enumerate(prob.cpu().numpy()):
                                        if i in self.model.config.id2label:
                                            label = self.model.config.id2label[i]
                                            result[label] = float(p)
                                    results.append(result)
                            
                            pbar.update(1)
                            
                            # Очистка памяти
                            del batch
                            del outputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e:
                            logger.error(f"Ошибка обработки: {e}")
                            continue
                
                return results
                
            except Exception as e:
                logger.error(f"Ошибка пакетной обработки: {e}")
                return []

    def recognize(self, audio_path, text):
        """Обработка одиночного примера"""
        # Улучшенная обработка входных данных
        if isinstance(audio_path, str) and os.path.isfile(audio_path):
            # Если передан путь к файлу
            waveform, sr = torchaudio.load(audio_path)
            audio = waveform
        else:
            # Если передан тензор или массив
            audio = audio_path
            
        # Используйте переменную или константу вместо хардкодированного значения
        sample_rate = 16000  # или получать из конфигурации
        results = self.batch_recognize([(audio, text)], sample_rate=sample_rate)
        return results[0] if results else {}
