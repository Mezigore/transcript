import torch
from transformers import AutoTokenizer, AutoFeatureExtractor, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import importlib.util
import sys
import logging
from typing import List, Dict
from contextlib import nullcontext
import tqdm

# Настраиваем логирование с меньшей детализацией для производительности
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def load_custom_model(model_name):
    """Загрузка пользовательской модели с оптимизациями"""
    from huggingface_hub import snapshot_download
    
    # Скачиваем модель локально с кэшированием
    model_path = snapshot_download(repo_id=model_name)
    
    # Загружаем пользовательский модуль
    custom_module_path = os.path.join(model_path, "audio_text_multimodal.py")
    spec = importlib.util.spec_from_file_location("audio_text_multimodal", custom_module_path)
    custom_module = importlib.util.module_from_spec(spec)
    sys.modules["audio_text_multimodal"] = custom_module
    spec.loader.exec_module(custom_module)
    
    # Получаем классы из пользовательского модуля
    WavLMBertConfig = custom_module.WavLMBertConfig
    WavLMBertForSequenceClassification = custom_module.WavLMBertForSequenceClassification
    
    # Загружаем конфигурацию
    config = WavLMBertConfig.from_pretrained(model_path)
    
    # Загружаем модель
    model = WavLMBertForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    )
    
    # Проверяем модель
    if not hasattr(model, 'config') or not hasattr(model.config, 'id2label'):
        raise ValueError("Не удалось загрузить конфигурацию модели с метками эмоций")
    
    return model

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
        
        # Prepare inputs
        audio_inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        text_inputs = self.processor(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Convert all attention masks to float type for consistency
        return {
            "input_values": audio_inputs["input_values"].squeeze(),
            "audio_attention_mask": audio_inputs["attention_mask"].squeeze().float(),  # Changed to float
            "input_ids": text_inputs["input_ids"].squeeze(),
            "text_attention_mask": text_inputs["attention_mask"].squeeze().float(),    # Changed to float
            "token_type_ids": text_inputs["token_type_ids"].squeeze() if "token_type_ids" in text_inputs else None
        }

class EmotionRecognizer:
    def __init__(self, model_name="aniemore/wavlm-bert-fusion-s-emotion-russian-resd", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = load_custom_model(model_name)
        
        if device == 'cuda':
            self.model = self.model.cuda()
            self.use_fp16 = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        else:
            self.use_fp16 = False
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        
        self.model.eval()
    
    def batch_recognize(self, audio_text_pairs, sample_rate: int, use_fp16: bool = False, device: str = 'cpu', batch_size: int = 1) -> List[Dict]:
        with torch.no_grad():
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
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    with torch.cuda.amp.autocast('cuda') if self.use_fp16 else nullcontext():
                        outputs = self.model(**batch)
                    
                    probs = torch.softmax(outputs.logits, dim=1)
                    
                    for prob in probs:
                        result = {}
                        for i, p in enumerate(prob.cpu().numpy()):
                            if i in self.model.config.id2label:
                                label = self.model.config.id2label[i]
                                result[label] = float(p)
                        results.append(result)
                    
                    pbar.update(1)
                    del batch
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return results

    def recognize(self, audio_path, text):
        """Single sample recognition"""
        results = self.batch_recognize([(audio_path, text)])
        return results[0]
