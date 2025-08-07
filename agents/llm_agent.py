"""
Llama 3.3 70B + LangChain Prompt-Based AI Agent
Tool'lar Llama 3.3 ile prompt engineering entegrasyonu ile çalışır
"""

import os
import base64
import io
from PIL import Image
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from groq import Groq
from config.settings import Settings

# LangChain imports - geri eklendi!
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from agents.tools import create_tools

import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    """Llama 3.3 70B + LangChain Prompt-Based AI Agent"""
    
    def __init__(self):
        self.settings = Settings()
        self.groq_client = None
        self.langchain_llm = None
        self.agent = None
        self.memory = None
        self.current_text_model = self.settings.text_model
        self.current_vision_model = self.settings.vision_model
        self._initialize_systems()
        
    def _initialize_systems(self):
        """Groq ve LangChain sistemlerini Llama 3.3 ile başlat"""
        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Groq istemcisi (vision için)
        self.groq_client = Groq(api_key=self.settings.groq_api_key)
        
        # LangChain + Groq LLM (Llama 3.3 ile tool entegrasyonu)
        self.langchain_llm = ChatGroq(
            groq_api_key=self.settings.groq_api_key,
            model_name=self.current_text_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens
        )
        
        # Konuşma hafızası
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Son 10 mesajı hatırla
        )
        
        # Tool'ları yükle
        tools = create_tools()
        
        # LangChain agent'ını başlat - Meta-Llama Maverick seviyesi
        self.agent = initialize_agent(
            tools=tools,
            llm=self.langchain_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,  # Debug için
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": self.settings.system_prompt,
                "input_variables": ["input", "chat_history", "agent_scratchpad"]
            }
        )
        
        logger.info(f"🦙 Llama 3.3 + LangChain Agent başlatıldı")
        logger.info(f"📝 Text Model: {self.current_text_model}")
        logger.info(f"👁️ Vision Model: {self.current_vision_model}")
        logger.info(f"🛠️ Tool Model: {self.current_text_model}")
    
    def switch_model(self, text_model: str, vision_model: str):
        """Model değiştir"""
        old_text = self.current_text_model
        old_vision = self.current_vision_model
        
        self.current_text_model = text_model
        self.current_vision_model = vision_model
        
        # LangChain LLM'i yeniden başlat
        self.langchain_llm = ChatGroq(
            groq_api_key=self.settings.groq_api_key,
            model_name=text_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens
        )
        
        # Agent'ı yeniden başlat
        tools = create_tools()
        self.agent = initialize_agent(
            tools=tools,
            llm=self.langchain_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": self.settings.system_prompt,
                "input_variables": ["input", "chat_history", "agent_scratchpad"]
            }
        )
        
        logger.info(f"🔄 Maverick Model değiştirildi:")
        logger.info(f"   Text: {old_text} → {text_model}")
        logger.info(f"   Vision: {old_vision} → {vision_model}")
        
        return f"✅ Llama 3.3 Modeller güncellendi!\n📝 Text: {text_model}\n👁️ Vision: {vision_model}"
    
    def process_message(self, message: str, image=None) -> str:
        """Llama 3.3 ile mesajı işle - LangChain + prompt-based tool entegrasyonu"""
        try:
            # Görsel var mı kontrol et
            if image is not None:
                return self._process_with_vision(message, image)
            else:
                return self._process_with_langchain(message)
                
        except Exception as e:
            error_msg = f"❌ Llama 3.3 işlem hatası: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _process_with_langchain(self, message: str) -> str:
        """LangChain agent ile prompt-based tool entegrasyonu"""
        try:
            # LangChain agent'ını çağır - tool'lar otomatik olarak çağrılacak
            response = self.agent.run(input=message)
            
            logger.info(f"✅ Maverick LangChain yanıt alındı")
            return response
            
        except Exception as e:
            error_msg = f"❌ LangChain Agent hatası: {str(e)}"
            logger.error(error_msg)
            
            # Fallback: Direkt LLM çağrısı
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.current_text_model,
                    messages=[
                        {"role": "system", "content": self.settings.system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature
                )
                return f"⚠️ Fallback mode:\n\n{completion.choices[0].message.content}"
            except Exception as fallback_e:
                return f"❌ Sistem hatası: {str(fallback_e)}"
    
    def _process_with_vision(self, message: str, image) -> str:
        """Görsel analizi için Meta-Llama Maverick vision"""
        try:
            # Görseli işle
            if hasattr(image, 'name'):  # Gradio file objesi
                img = Image.open(image.name)
            else:
                img = image
            
            # Maverick için optimize et
            img.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Maverick Vision prompt
            enhanced_prompt = f"""
[META-LLAMA MAVERICK VISION ANALYSIS]

Kullanıcı mesajı: {message}

Bu görseli Maverick seviyede analiz et:

1. 🔍 DETAYLI ANALİZ:
   - Ana objeler ve kompozisyon
   - Renkler, ışık, perspektif
   - Teknik kalite ve stil

2. 📖 METIN OKUMA (OCR):
   - Görseldeki tüm yazıları oku
   - Farklı dillerdeki metinleri çevir
   - Tabela, logo, işaret vb. tanımla

3. 🧠 CONTEXT ANALYSİS:
   - Bu görsel hangi amaçla çekilmiş?
   - Hangi ortam, zaman, durum?
   - Kültürel ve sosyal context

4. 💡 MAVERICK İNSIGHT:
   - Dikkat çeken detaylar
   - Gizli anlamlar veya semboller
   - Profesyonel değerlendirme

Türkçe'de detaylı ve yapılandırılmış yanıt ver.
"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            completion = self.groq_client.chat.completions.create(
                model=self.current_vision_model,
                messages=messages,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature
            )
            
            response = completion.choices[0].message.content
            logger.info("✅ Maverick Vision analizi tamamlandı")
            return response
            
        except Exception as e:
            return f"❌ Meta-Llama Maverick Vision hatası: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Konuşma geçmişini al - LangChain memory'den"""
        if self.memory and hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            history = []
            for msg in messages:
                history.append({
                    "role": "human" if hasattr(msg, 'content') and msg.__class__.__name__ == "HumanMessage" else "assistant",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
            return history
        return []
    
    def clear_memory(self):
        """Konuşma geçmişini temizle - eski interface uyumluluğu için"""
        return self.clear_history()
    
    def clear_history(self):
        """LangChain memory'yi temizle"""
        if self.memory:
            self.memory.clear()
        logger.info("🗑️ Meta-Llama Maverick hafızası temizlendi!")
        return "🗑️ Meta-Llama Maverick hafızası temizlendi!"
    
    def get_agent_info(self) -> str:
        """Agent bilgilerini al"""
        tool_count = len(create_tools()) if create_tools() else 0
        memory_count = len(self.get_conversation_history())
        
        return f"""
🦙 **Meta-Llama Maverick + LangChain Agent**

📊 **Aktif Modeller:**
- 📝 Text: `{self.current_text_model}`
- 👁️ Vision: `{self.current_vision_model}`
- 🛠️ Tool: `{self.settings.tool_model}`

🎯 **Maverick Özellikleri:**
- ✅ LangChain framework entegrasyonu
- ✅ Prompt-based tool management
- ✅ Otomatik tool selection
- ✅ Gelişmiş reasoning capabilities
- ✅ Vision analysis + OCR
- ✅ Multi-language support
- ✅ Conversation memory ({memory_count} mesaj)

🛠️ **Mevcut Tool'lar:** {tool_count} adet
- ⏰ Akıllı zaman/tarih işlemi
- 📊 Gelişmiş metin analizi  
- 🌍 Dil tespiti ve kültürel analiz
- 📝 AI-powered özetleme
- 🔍 Bilgi arama ve analiz

Meta-Llama Maverick seviyede reasoning + LangChain tool entegrasyonu! 🚀
"""
