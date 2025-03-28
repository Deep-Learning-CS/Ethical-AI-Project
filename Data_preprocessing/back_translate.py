from deep_translator import GoogleTranslator
import random

class BackTranslator:
    """Handles back-translation augmentation using Google Translate"""
    
    def __init__(self):
        # Supported languages in your pipeline
        self.pivot_lang = "en"  # Best intermediate language
        self.supported_langs = {"uk": "ukrainian", "ru": "russian"}
        
    def translate(self, text, source_lang):
        """Single translation direction"""
        try:
            return GoogleTranslator(
                source=source_lang,
                target=self.pivot_lang
            ).translate(text[:5000])  # Character limit safeguard
        except Exception as e:
            print(f"Translation error ({source_lang}→en): {str(e)}")
            return None

    def backtranslate(self, text, source_lang):
        """Full back-translation cycle"""
        if not isinstance(text, str) or len(text) < 10:  # Skip short texts
            return text
            
        try:
            # First translate to English
            en_text = self.translate(text, source_lang)
            if not en_text:
                return text
                
            # Then translate back to original language
            return GoogleTranslator(
                source=self.pivot_lang,
                target=source_lang
            ).translate(en_text[:5000])
            
        except Exception as e:
            print(f"Back-translation failed: {str(e)}")
            return text