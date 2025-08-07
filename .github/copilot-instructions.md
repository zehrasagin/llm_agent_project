<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# LLM Agent Project - Copilot Instructions

Bu proje aşağıdaki teknolojileri kullanır:
- **Gradio**: Web arayüzü geliştirme
- **Groq AI**: LLaMA 3 model entegrasyonu
- **LangChain**: Agent ve tool yönetimi
- **Python**: Ana programlama dili

## Kod Yazım Kuralları

1. **Type Hints**: Tüm fonksiyonlarda type hint kullan
2. **Docstrings**: Her fonksiyon için açıklayıcı docstring ekle
3. **Error Handling**: Uygun try-catch blokları kullan
4. **Modularity**: Kodu modüler şekilde organize et
5. **Environment Variables**: Hassas bilgileri .env dosyasında sakla

## Proje Yapısı

- `main.py`: Ana uygulama giriş noktası
- `agents/`: LangChain agent'ları ve tool'ları
- `ui/`: Gradio arayüz bileşenleri
- `config/`: Konfigürasyon ayarları

## Best Practices

- Asenkron programlama kullan (async/await)
- Logging ekle
- Unit test'ler yaz
- Code documentation'ı güncel tut
