import os
from typing import List
from langdetect import detect
from transformers import pipeline

from .common import OfflineTranslator



ISO_639_1_TO_M2M100 = {
    'zh': 'zh', 'cs': 'cs', 'nl': 'nl', 'en': 'en', 'fr': 'fr', 'de': 'de',
    'hu': 'hu', 'it': 'it', 'ja': 'ja', 'ko': 'ko', 'pl': 'pl', 'pt': 'pt',
    'ro': 'ro', 'ru': 'ru', 'es': 'es', 'tr': 'tr', 'uk': 'uk', 'vi': 'vi',
    'ar': 'ar', 'sr': 'sr', 'hr': 'hr', 'th': 'th', 'id': 'id'
}

class M2M100HFTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh', 'CHT': 'zh', 'CSY': 'cs', 'NLD': 'nl', 'ENG': 'en',
        'FRA': 'fr', 'DEU': 'de', 'HUN': 'hu', 'ITA': 'it', 'JPN': 'ja',
        'KOR': 'ko', 'PLK': 'pl', 'PTB': 'pt', 'ROM': 'ro', 'RUS': 'ru',
        'ESP': 'es', 'TRK': 'tr', 'UKR': 'uk', 'VIN': 'vi', 'ARA': 'ar',
        'SRP': 'sr', 'HRV': 'hr', 'THA': 'th', 'IND': 'id'
    }
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'm2m100')
    _TRANSLATOR_MODEL = "facebook/m2m100_418M"
    _last_detected_lang = "auto"
    _last_target_lang = "auto"


    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        if ':' not in device:
            device += ':0'
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self._TRANSLATOR_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            try:
                detected_lang = detect('\n'.join(queries))
                target_lang = self._map_detected_lang_to_translator(detected_lang)
            except Exception as e:
                self.logger.warn(f'Could not detect language from over all sentence: {e}. Will try per sentence.')
                target_lang = None

            if target_lang is None:
                from_lang = self._last_target_lang
            else:
                from_lang = target_lang
                self._last_target_lang = target_lang

        return [self._translate_sentence(from_lang, to_lang, query) for query in queries]

    def _translate_sentence(self, from_lang: str, to_lang: str, query: str) -> str:
        if not self.is_loaded():
            return ''

        if from_lang == 'auto':
            try:
                detected_lang = self._detect_language(query)
                from_lang = self._map_detected_lang_to_translator(detected_lang)
            except Exception as e:
                self.logger.warn(f'Could not detect language for text: {query}: {e}')
                from_lang = None

        if from_lang is None:
            self.logger.warn(f'M2M100 Translation Failed. Could not detect language (Or language not supported for text: {query})')
            result = self._manual_input_translation(query, from_lang, to_lang)
            return result

        translator = pipeline('translation',
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=from_lang,
            tgt_lang=to_lang,
            max_length = 512,
        )
        try:
            if from_lang in ["zh", "ja", "ko"]:
                lengh = 10
            else:
                lengh = 2
            self.logger.warn(f'from_lang {from_lang}')

            result = translator(query)[0]['translation_text']
            if len(result) > lengh * len(query):
                self.logger.warn(f'Suspiciously long translation: {result}')
                result = self._manual_input_translation(query, from_lang, to_lang)
            if len(result) < 1:
                self.logger.warn(f'Suspiciously short translation: {result}')
                result = self._manual_input_translation(query, from_lang, to_lang)
        except Exception as e:
            self.logger.error(f'Translation failed: {e}')
            result = self._manual_input_translation(query, from_lang, to_lang)

        return result

    def _map_detected_lang_to_translator(self, lang):
        if lang not in ISO_639_1_TO_M2M100:
            return None

        return ISO_639_1_TO_M2M100[lang]
    
    def _detect_language(self, text: str) -> str:
        lang_detector = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
        result = lang_detector(text)
        return result[0]['label']

    def _manual_input_translation(self, query: str, from_lang: str, to_lang: str) -> str:
        print(f"Could not translate the following text: {query}")
        manual_translation = input("Please enter the translation manually: ")
        return manual_translation


    async def _download(self):
        import huggingface_hub
        # do not download msgpack and h5 files as they are not needed to run the model
        huggingface_hub.snapshot_download(self._TRANSLATOR_MODEL, cache_dir=self._MODEL_SUB_DIR, ignore_patterns=["*.msgpack", "*.h5", '*.ot',".*", "*.safetensors"])

    def _check_downloaded(self) -> bool:
        import huggingface_hub
        return huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'pytorch_model.bin', cache_dir=self._MODEL_SUB_DIR) is not None

class M2M100HFBigTranslator(M2M100HFTranslator):
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'm2m100')
    _TRANSLATOR_MODEL = 'facebook/m2m100_1.2B'
