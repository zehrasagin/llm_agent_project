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
            
            with gr.Blocks(css=css, title="CevapLlama 4 - AI Vision & Chat") as interface:
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1>🤖 CevapLlama 4 Vision</h1>
                    <p>Groq AI (LLaMA 4) ile gelişmiş görsel analiz ve akıllı sohbet</p>
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
                        # Bilgi paneli
                        gr.HTML("""
                        <div style="padding: 20px; background-color: #f5f5f5; border-radius: 10px; color: #222;">
                            <h3>🧠 LLaMA 4 Yetenekleri</h3>
                            <ul>
                                <li>🖼️ Gelişmiş görsel analiz</li>
                                <li>🌍 Çoklu dil desteği</li>
                                <li>🔗 Groq AI entegrasyonu</li>
                                <li>🛠️ Akıllı tool kullanımı</li>
                                <li>⏰ Zaman ve tarih bilgisi</li>
                                <li>📊 Metin analizi</li>
                                <li>📝 Özetleme</li>
                                <li>🔍 Dil algılama</li>
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
                        </div>
                        """)
                
                # Event handlers
                def process_message(message: str, image, history: List[List[str]]) -> Tuple[List[List[str]], str, None]:
                    """Kullanıcı mesajını işler."""
                    try:
                        if not message.strip() and image is None:
                            return history, "", None
                        
                        # Mesaj hazırla
                        display_message = message if message.strip() else "Görsel analizi"
                        
                        # Agent'tan yanıt al
                        response = self.agent.process_message(message, image)
                        
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
