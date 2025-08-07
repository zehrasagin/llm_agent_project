"""
LLaMA 4 Prompt-Based AI Agent - Tool'suz tamamen prompt engineering ile çalışır
"""

import os
import base64
import io
from PIL import Image
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from groq import Groq
from config.settings import Settings

import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    """LLaMA 4 Prompt-Based AI Agent - Tool'suz tamamen prompt engineering ile çalışır"""
    
    def __init__(self):
        self.settings = Settings()
        self.client = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_text_model = self.settings.text_model
        self.current_vision_model = self.settings.vision_model
        self._initialize_client()
        
    def _initialize_client(self):
        """Groq istemcisini başlat"""
        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=self.settings.groq_api_key)
        logger.info(f"🦙 LLaMA 4 Prompt-Based Agent başlatıldı")
        logger.info(f"📝 Text Model: {self.current_text_model}")
        logger.info(f"👁️ Vision Model: {self.current_vision_model}")
    
    def switch_model(self, text_model: str, vision_model: str):
        """Model değiştir"""
        old_text = self.current_text_model
        old_vision = self.current_vision_model
        
        self.current_text_model = text_model
        self.current_vision_model = vision_model
        
        logger.info(f"🔄 Model değiştirildi:")
        logger.info(f"   Text: {old_text} → {text_model}")
        logger.info(f"   Vision: {old_vision} → {vision_model}")
        
        return f"✅ Modeller güncellendi!\n� Text: {text_model}\n👁️ Vision: {vision_model}"
    
    def _get_current_time_info(self) -> str:
        """Mevcut zaman bilgisini al (prompt için)"""
        now = datetime.now()
        turkish_months = [
            "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
            "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"
        ]
        turkish_days = [
            "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"
        ]
        
        month_name = turkish_months[now.month - 1]
        day_name = turkish_days[now.weekday()]
        
        return f"Şu anki zaman: {now.day} {month_name} {now.year}, {day_name}, {now.hour:02d}:{now.minute:02d}"
    
    def _create_enhanced_prompt(self, user_message: str, has_image: bool = False) -> str:
        """LLaMA 4 için gelişmiş prompt oluştur"""
        time_info = self._get_current_time_info()
        
        base_context = f"""
[SISTEM BİLGİSİ]
{time_info}
Model: LLaMA 4 Prompt-Based System
Yetenek: Pure prompt engineering ile tüm işlemler

[KULLANICI MESAJI]
{user_message}

[GÖREV]
Yukarıdaki kullanıcı mesajını analiz et ve şu kurallara göre yanıt ver:

1. 🕐 ZAMAN SORULARI: "saat kaç", "bugün ne günü", "tarih" gibi ifadeler varsa:
   - Yukarıdaki sistem zamanını kullan
   - Türkçe format: "📅 Bugün: [gün] [ay] [yıl], [gün adı]"
   - "🕐 Saat: [saat]:[dakika]" formatı kullan

2. 📊 METİN ANALİZİ: "analiz et", "say", "değerlendir" ifadeleri varsa:
   - Kelime/karakter/cümle sayısını hesapla
   - Dil tespiti yap
   - Ton analizi yap (resmi/samimi/nötr)
   - JSON formatında düzenle

3. 📝 ÖZETLEME: "özetle", "kısalt", "summary" ifadeleri varsa:
   - Ana fikirleri koru
   - Önemli detayları dahil et
   - Özet oranını belirt

4. 🌍 DİL TESPİTİ: Kullanıcının dilini otomatik algıla:
   - Türkçe → Samimi ve detaylı yanıt
   - English → Professional response
   - Français → Réponse élégante
   - Deutsch → Präzise Antwort

5. 💬 GENEL SOHBET: Yukarıdakiler dışında:
   - Kullanıcının dilinde yanıtla
   - Yardımcı ve bilgili ol
   - Emoji kullanarak samimi yap
"""

        if has_image:
            base_context += """
6. 🖼️ GÖRSEL ANALİZ: Görsel mevcut:
   - Ana objeleri tanımla
   - Renk ve kompozisyonu açıkla
   - Yazıları oku ve çevir
   - Teknik detayları analiz et
   - Bağlamsal yorumlar yap
   - Detaylı ve yapılandırılmış yanıt ver
"""

        return base_context

    def process_message(self, message: str, image=None) -> str:
        """LLaMA 4 ile mesajı işle - Tamamen prompt-based"""
        try:
            # Görsel var mı kontrol et
            has_image = image is not None
            
            if has_image:
                return self._process_with_vision(message, image)
            else:
                return self._process_text_only(message)
                
        except Exception as e:
            error_msg = f"❌ LLaMA 4 işlem hatası: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _process_text_only(self, message: str) -> str:
        """Sadece metin için LLaMA 4 işlemi"""
        enhanced_prompt = self._create_enhanced_prompt(message)
        
        # Konuşma geçmişini ekle
        messages = [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        # Son birkaç mesajı da ekle (context için)
        if self.conversation_history:
            recent_history = self.conversation_history[-4:]  # Son 4 mesaj
            for hist in recent_history:
                if "role" in hist and "content" in hist:
                    messages.insert(-1, {
                        "role": hist["role"],
                        "content": hist["content"][:500]  # Kısalt
                    })
        
        # LLaMA 4 ile çağrı yap
        try:
            completion = self.client.chat.completions.create(
                model=self.current_text_model,
                messages=messages,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Konuşma geçmişine ekle
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "model": self.current_text_model
            })
            
            return response
            
        except Exception as e:
            return f"❌ LLaMA 4 API hatası: {str(e)}"
    
    def _process_with_vision(self, message: str, image) -> str:
        """Görsel analizi için Gemini Vision API kullanılır"""
        try:
            import requests
            import json
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                return "❌ Görsel analizi için Gemini API anahtarı bulunamadı. Lütfen .env dosyanıza GEMINI_API_KEY ekleyin."
            # Görseli base64'e çevir
            if hasattr(image, 'name'):
                img = Image.open(image.name)
            else:
                img = image
            img.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            # Gemini API'ya istek
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={gemini_api_key}"
            prompt = self._create_enhanced_prompt(message, has_image=True)
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                        ]
                    }
                ]
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                result = response.json()
                gemini_text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Geçmişe ekle
                self.conversation_history.append({
                    "role": "user",
                    "content": f"{message} [📷 Görsel eklendi]",
                    "timestamp": datetime.now().isoformat()
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": gemini_text,
                    "timestamp": datetime.now().isoformat(),
                    "model": "gemini-pro-vision"
                })
                return gemini_text
            else:
                return f"❌ Gemini Vision API hatası: {response.status_code} - {response.text}"
        except Exception as e:
            return f"❌ Gemini Vision API hata: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Konuşma geçmişini al"""
        return self.conversation_history
    
    def clear_memory(self):
        """Konuşma geçmişini temizle - eski interface uyumluluğu için"""
        return self.clear_history()
    
    def clear_history(self):
        """Konuşma geçmişini temizle"""
        self.conversation_history.clear()
        logger.info("🗑️ Konuşma geçmişi temizlendi!")
        return "🗑️ Konuşma geçmişi temizlendi!"
    
    def get_agent_info(self) -> str:
        """Agent bilgilerini al"""
        return f"""
🦙 **LLaMA 4 Prompt-Based Agent**

📊 **Aktif Modeller:**
- 📝 Text: `{self.current_text_model}`
- 👁️ Vision: `{self.current_vision_model}`

🎯 **Özellikler:**
- ✅ Tamamen prompt-based (tool yok)
- ✅ Otomatik dil algılama
- ✅ Akıllı zaman/tarih işlemi
- ✅ Metin analizi ve özetleme
- ✅ Görsel analiz (OCR dahil)
- ✅ Çok dilli destek

💬 **Konuşma Geçmişi:** {len(self.conversation_history)} mesaj

Artık manuel tool'lar yok! Her şey LLaMA 4'ün doğal yetenekleri ile yapılıyor! 🚀
"""
