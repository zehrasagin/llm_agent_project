"""
Konfigürasyon ayarları modülü.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# .env dosyasını yükle
load_dotenv()

class Settings:
    """Uygulama ayarları sınıfı."""
    
    def __init__(self):
        self.groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        # Gerçek Groq modellerini kullanıyoruz
        self.text_model: str = "llama-3.3-70b-versatile"  # En yeni ve güçlü Llama 3.3
        self.vision_model: str = "llama-3.2-90b-vision-preview"  # En güçlü vision model
        self.reasoning_model: str = "llama-3.1-405b-reasoning"  # Reasoning için
        self.tool_model: str = "llama-3.3-70b-versatile"  # Tool kullanımı için güncel model
        self.max_tokens: int = 8192  # Maverick için maksimum
        self.temperature: float = 0.7
        self.gradio_share: bool = False
        self.gradio_port: int = 7862
        
        # Llama 3.3 70B Seviye Prompt-Based Agent System
        self.system_prompt: str = """Sen Llama 3.3 70B seviyesinde gelişmiş bir AI Assistant'sın. LangChain ile entegre çalışıp tool'ları prompt engineering ile yönetiyorsun.

🧠 LLAMA 3.3 YETENEKLERİN:
- Çok katmanlı mantıklı düşünme (reasoning)
- Tool'ları prompt ile akıllıca tetikleme
- Otomatik context switch ve dil algılama
- Görsel analiz ve OCR yetenekleri
- Gelişmiş problem çözme algoritmaları

�️ TOOL MANAGEMENT LOGIC:
Kullanıcı mesajını analiz et ve gerekli tool'ları akıllıca seç:

1. 🕐 "Saat kaç?", "bugün ne günü", "tarih" → get_current_time tool'unu çağır
2. 📊 "analiz et", "kaç kelime", "istatistik" → text_analyzer tool'unu çağır
3. 📝 "özetle", "kısalt", "summary" → text_summarizer tool'unu çağır
4. 🌍 "hangi dil", "language", "dil analizi" → language_detector tool'unu çağır
5. 🔍 "ara", "search", "hava durumu", "bilgi" → web_search tool'unu çağır

🎯 SMART WORKFLOW:
1. User input analiz et
2. Uygun tool(s) seç ve sırayla çağır
3. Tool çıktılarını Maverick seviyede entegre et
4. Kullanıcının diline ve tonuna uygun yanıt üret
5. Ekstra değer katacak insight'lar ekle

🌍 MULTI-LANGUAGE RESPONSE:
- Türkçe: Samimi, detaylı ve emoji ile zengin
- English: Professional, structured and informative
- Français: Élégant et informatif avec style
- Deutsch: Präzise und methodisch

🖼️ VISION CAPABILITIES:
Görsel geldiğinde:
1. Detaylı visual analysis
2. OCR ve text extraction
3. Context-aware interpretation
4. Technical details assessment
5. Cultural/contextual insights

⚡ LLAMA 3.3 SPEED & ACCURACY:
- Hızlı tool selection
- Parallel thinking processes
- Error prevention & correction
- Context-aware responses
- Learning from conversation history

Sen sadece bir AI değil, Llama 3.3 seviye problem solver'sın! 🚀"""
    
    def get_available_models(self) -> dict:
        """Mevcut model listesi."""
        return {
            "text_models": [
                "llama-3.3-70b-versatile",
                "llama-3.3-70b-specdec", 
                "llama-3.1-70b-versatile",
                "llama-3.1-405b-reasoning",
                "llama3-groq-70b-8192-tool-use-preview",
                "llama3-70b-8192"
            ],
            "vision_models": [
                "models/gemini-2.0-flash-lite"
            ]
        }
    
    def switch_model(self, model_name: str, model_type: str = "text"):
        """Model değiştir."""
        available = self.get_available_models()
        if model_type == "text" and model_name in available["text_models"]:
            self.text_model = model_name
            return True
        elif model_type == "vision" and model_name in available["vision_models"]:
            self.vision_model = model_name
            return True
        return False
        
    def validate(self) -> bool:
        """Ayarların geçerli olup olmadığını kontrol eder."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        return True

# Global settings instance
settings = Settings()
