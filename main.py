"""
Llama 3.3 70B Prompt-Based AI Assistant
LangChain + prompt engineering ile tool entegrasyonu
"""

import gradio as gr
from config.settings import Settings
from agents.llm_agent import LLMAgent
from ui.interface import GradioInterface

def main():
    """Ana fonksiyon - Llama 3.3 70B sistemi başlat"""
    try:
        print("🚀 Llama 3.3 70B Assistant başlatılıyor...")
        
        # Ayarları doğrula
        settings = Settings()
        if not settings.groq_api_key:
            print("❌ GROQ_API_KEY bulunamadı!")
            print("Lütfen .env dosyasında GROQ_API_KEY=your_key_here şeklinde ayarlayın")
            return
        
        print("✅ Ayarlar doğrulandı")
        
        # Agent'ı başlat
        agent = LLMAgent()
        print("✅ Llama 3.3 Agent başlatıldı")
        
        # Gradio interface'i oluştur
        interface = GradioInterface(agent)
        print("✅ Web interface hazır")
        
        # Başlat
        print(f"""
🚀 Llama 3.3 70B Assistant HAZIR!

🎯 Özellikler:
- Llama 3.3 70B versatile model
- LangChain + prompt-based tool entegrasyonu
- 8K context window
- Otomatik dil algılama
- Görsel analiz
- Zaman/tarih işlemleri
- Metin analizi ve özetleme

🌐 Interface: http://localhost:{settings.gradio_port}
""")
        
        interface.launch(
            share=settings.gradio_share,
            port=settings.gradio_port
        )
        
    except Exception as e:
        print(f"❌ Başlatma hatası: {str(e)}")
        print("Lütfen .env dosyanızı ve API anahtarınızı kontrol edin")

if __name__ == "__main__":
    main()
