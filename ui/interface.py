"""
Gradio arayüz modülü.
"""

import gradio as gr
from typing import List, Tuple
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioInterface:
    """Gradio web arayüzü sınıfı."""
    
    def __init__(self, agent):
        """Arayüzü agent ile başlatır."""
        self.agent = agent
        self.interface = None
        self._create_interface()
    
    def _create_interface(self) -> None:
        """Gradio arayüzünü oluşturur."""
        try:
            # CSS stilleri
            css = """
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
            }
            .user-message {
                background-color: #e3f2fd;
                padding: 10px;
                margin: 5px;
                border-radius: 10px;
                text-align: right;
            }
            .bot-message {
                background-color: #f1f8e9;
                padding: 10px;
                margin: 5px;
                border-radius: 10px;
                text-align: left;
            }
            """
            
            with gr.Blocks(css=css, title="Llama 3.3 - AI Vision & Chat") as interface:
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1>🚀 CevapLlama</h1>
                    <p>Groq AI ile CevapLlama - gelişmiş görsel analiz ve akıllı sohbet</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Chat arayüzü
                        chatbot = gr.Chatbot(
                            label="Konuşma",
                            height=400,
                            show_label=True,
                            container=True,
                            bubble_full_width=False
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                user_input = gr.Textbox(
                                    label="Mesajınız",
                                    placeholder="Merhaba! Bir soru sorun veya görsel yükleyin...",
                                    lines=2
                                )
                            with gr.Column(scale=1):
                                image_input = gr.Image(
                                    label="Görsel Yükle",
                                    type="pil",
                                    sources=["upload", "webcam"],
                                    height=100
                                )
                        
                        with gr.Row():
                            send_btn = gr.Button("Gönder", variant="primary", scale=2)
                            clear_btn = gr.Button("Konuşmayı Temizle", variant="secondary", scale=1)
                    
                    with gr.Column(scale=1):
                        # Model seçici
                        with gr.Accordion("🚀 Model Ayarları", open=False):
                            text_model_dropdown = gr.Dropdown(
                                choices=[
                                    "llama-3.3-70b-versatile",  # En yeni Llama 3.3
                                    "llama-3.3-70b-specdec", 
                                    "llama-3.1-70b-versatile",
                                    "llama-3.1-405b-reasoning",
                                    "llama3-groq-70b-8192-tool-use-preview",
                                    "llama3-70b-8192"
                                ],
                                value="llama-3.3-70b-versatile",  # Default en yeni model
                                label="Text Model",
                                interactive=True
                            )
                            vision_model_dropdown = gr.Dropdown(
                                choices=[
                                    "models/gemini-2.0-flash-lite"
                                ],
                                value="models/gemini-2.0-flash-lite",
                                label="Vision Model (Gemini 2.0 Flash Lite)",
                                interactive=False
                            )
                        
                        # Bilgi paneli
                        gr.HTML("""
                        <div style="padding: 20px; background-color: #222; border-radius: 10px; color: #f5f5f5;">
                            <h3>🚀 Llama 3.3 70B Yetenekleri</h3>
                            <ul>
                                <li>� 17B parametre Maverick model</li>
                                <li>🧠 128K context window</li>
                                <li>�🖼️ 90B param görsel analiz</li>
                                <li>🌍 Çoklu dil desteği</li>
                                <li>🔗 Groq AI hızlı inference</li>
                                <li>🛠️ LangChain tool entegrasyonu</li>
                                <li>⏰ Zaman ve tarih bilgisi</li>
                                <li>📊 Gelişmiş metin analizi</li>
                                <li>📝 AI-powered özetleme</li>
                                <li>🔍 Otomatik dil algılama</li>
                            </ul>
                            
                            <h3>💡 Örnek Kullanım</h3>
                            <ul>
                                <li>"Bu görselde ne var?"</li>
                                <li>"Saat kaç?"</li>
                                <li>"Bu metni analiz et: ..."</li>
                                <li>"What time is it?" (İngilizce)</li>
                                <li>"Résume ce texte" (Fransızca)</li>
                                <li>Görsel + "Açıkla"</li>
                            </ul>
                            
                            <p><strong>🚀 Powered by Llama 3.3 70B!</strong></p>
                        </div>
                        """)
                
                # Event handlers
                def process_message(message: str, image, history: List[List[str]]) -> Tuple[List[List[str]], str, None]:
                    """
                    Kullanıcı mesajını işler. Eğer saat veya tarih soruluyorsa, local date/time bilgisini prompt'a ekler.
                    """
                    try:
                        import re
                        import datetime
                        if not message.strip() and image is None:
                            return history, "", None
                        
                        # Görsel yüklendiyse Gemini Vision API ile analiz et
                        if image is not None:
                            try:
                                import os
                                import requests
                                import json
                                import base64
                                import io
                                from PIL import Image
                                gemini_api_key = os.getenv("GEMINI_API_KEY")
                                if not gemini_api_key:
                                    error_response = "❌ Görsel analizi için Gemini API anahtarı bulunamadı. Lütfen .env dosyanıza GEMINI_API_KEY ekleyin."
                                    display_message = message if message.strip() else "Görsel analizi"
                                    history.append([display_message, error_response])
                                    return history, "", None
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
                                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={gemini_api_key}"
                                prompt = message or "Bu resmi açıkla"
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
                                    display_message = message if message.strip() else "Görsel analizi"
                                    history.append([display_message, gemini_text])
                                    return history, "", None
                                else:
                                    error_response = f"❌ Gemini Vision API hatası: {response.status_code} - {response.text}"
                                    display_message = message if message.strip() else "Görsel analizi"
                                    history.append([display_message, error_response])
                                    return history, "", None
                            except Exception as e:
                                error_response = f"❌ Görsel analizi hatası: {str(e)}"
                                display_message = message if message.strip() else "Görsel analizi"
                                history.append([display_message, error_response])
                                return history, "", None
                        
                        # Mesaj hazırla
                        display_message = message if message.strip() else "Görsel analizi"
                        
                        # Saat veya tarih sorusu mu?
                        time_patterns = [
                            r"saat[\s\?]*$", r"saat kaç", r"what time", r"current time", r"time now", r"kaç saat", r"zaman nedir", r"date", r"tarih", r"bugün ne", r"hangi gün"
                        ]
                        is_time_query = any(re.search(pat, message.lower()) for pat in time_patterns)
                        prompt = message
                        if is_time_query:
                            now = datetime.datetime.now()
                            local_time = now.strftime("%d.%m.%Y %A %H:%M:%S")
                            # Türkçe ve İngilizce açıklama ekle
                            ek = f"\nNot: Şu anki yerel saat ve tarih: {local_time} (Lütfen cevabında bu bilgiyi kullan.)"
                            prompt = f"{message}\n{ek}"
                        
                        # Agent'tan yanıt al
                        response = self.agent.process_message(prompt, image)
                        
                        # Geçmişe ekle
                        history.append([display_message, response])
                        
                        return history, "", None
                        
                    except Exception as e:
                        logger.error(f"Mesaj işleme hatası: {str(e)}")
                        error_response = f"Üzgünüm, bir hata oluştu: {str(e)}"
                        history.append([message or "Görsel", error_response])
                        return history, "", None
                
                def clear_conversation():
                    """Konuşmayı temizler."""
                    try:
                        self.agent.clear_memory()
                        return []
                    except Exception as e:
                        logger.error(f"Temizleme hatası: {str(e)}")
                        return []
                
                def change_text_model(model_name):
                    """Text modelini değiştirir."""
                    try:
                        from config.settings import settings
                        if settings.switch_model(model_name, "text"):
                            # Agent'ı yeniden başlat
                            self.agent._initialize_llms()
                            return f"✅ Text model değiştirildi: {model_name}"
                        return "❌ Model değiştirilemedi"
                    except Exception as e:
                        return f"❌ Hata: {str(e)}"
                
                def change_vision_model(model_name):
                    """Vision modelini değiştirir."""
                    try:
                        from config.settings import settings
                        if settings.switch_model(model_name, "vision"):
                            # Agent'ı yeniden başlat
                            self.agent._initialize_llms()
                            return f"✅ Vision model değiştirildi: {model_name}"
                        return "❌ Model değiştirilemedi"
                    except Exception as e:
                        return f"❌ Hata: {str(e)}"
                
                # Event bindings
                send_btn.click(
                    fn=process_message,
                    inputs=[user_input, image_input, chatbot],
                    outputs=[chatbot, user_input, image_input]
                )
                
                user_input.submit(
                    fn=process_message,
                    inputs=[user_input, image_input, chatbot],
                    outputs=[chatbot, user_input, image_input]
                )
                
                clear_btn.click(
                    fn=clear_conversation,
                    outputs=[chatbot]
                )
                
                # Model değiştirme event'leri
                text_model_dropdown.change(
                    fn=change_text_model,
                    inputs=[text_model_dropdown],
                    outputs=[]
                )
                
                vision_model_dropdown.change(
                    fn=change_vision_model,
                    inputs=[vision_model_dropdown],
                    outputs=[]
                )
            
            self.interface = interface
            logger.info("Gradio arayüzü başarıyla oluşturuldu")
            
        except Exception as e:
            logger.error(f"Arayüz oluşturma hatası: {str(e)}")
            raise
    
    def launch(self, share: bool = False, port: int = 7860, max_tries: int = 10) -> None:
        """Arayüzü başlatır. Port kullanımdaysa bir sonraki portu dener."""
        try:
            if not self.interface:
                raise ValueError("Arayüz oluşturulmamış")
            
            current_port = port
            for attempt in range(max_tries):
                try:
                    logger.info(f"Arayüz başlatılıyor - Port: {current_port}, Share: {share}")
                    self.interface.launch(
                        share=share,
                        server_port=current_port,
                        server_name="0.0.0.0",
                        show_error=True,
                        quiet=False
                    )
                    return
                except Exception as e:
                    if "address already in use" in str(e).lower() or "Cannot find empty port" in str(e):
                        logger.warning(f"Port {current_port} kullanılıyor, bir sonrakini deniyorum...")
                        current_port += 1
                    else:
                        logger.error(f"Arayüz başlatma hatası: {str(e)}")
                        raise
            logger.error(f"{max_tries} port denemesine rağmen boş port bulunamadı.")
            raise RuntimeError(f"{max_tries} port denemesine rağmen boş port bulunamadı.")
        except Exception as e:
            logger.error(f"Arayüz başlatma hatası: {str(e)}")
            raise
