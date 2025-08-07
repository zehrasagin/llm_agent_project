
# CevapLlama 3.3 Vision - Ultra Advanced AI


This project uses a Gradio interface to receive text and images from the user, leverages **LLaMA 3.3** series models via Groq AI for text, and integrates Gemini Vision API for advanced image analysis. The system is orchestrated with a LangChain agent structure for modular and intelligent tool use.


## 🚀 LLaMA 3.3 & Gemini Vision Models

- **🧠 llama-3.3-70b-versatile**: En güçlü genel amaçlı model (70B param)
- **⚡ llama-3.3-70b-specdec**: Hızlı speculative decoding (70B param)
- **🤖 llama-3.1-405b-reasoning**: Muhakeme odaklı dev model (405B param!)
- **👁️ llama-3.2-90b-vision-preview**: Gelişmiş görsel analiz (90B param, metin tabanlı)
- **🔮 Gemini Vision API**: Google Gemini ile gerçek görsel analiz ve OCR
- **🛠️ llama3-groq-70b-8192-tool-use-preview**: Tool kullanımı optimize
- **💾 8K Token Limit**: Uzun metinler ve detaylı analiz


## 🚀 New Features

- **🖼️ Görsel Analiz (Gemini Vision)**: Resim yükleyin, Gemini API ile içeriği ve metni analiz edin
- **🌍 Çoklu Dil Desteği**: Türkçe, İngilizce, Fransızca otomatik algılama
- **🧠 Gelişmiş AI**: Akıllı promptlar ve bağlam anlayışı
- **📱 Modern Arayüz**: Görsel yükleme ve geliştirilmiş tasarım
- **🛠️ Gelişmiş Tool'lar**: Zaman, analiz, özet, dil algılama, arama

## 📦 Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturun ve Groq API anahtarınızı ekleyin:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🏃‍♂️ Çalıştırma

```bash
python main.py
```

## � Kullanım Örnekleri

### Metin Sohbet
- "Merhaba!" → Türkçe yanıt
- "Hello!" → İngilizce yanıt  
- "Saat kaç?" → Güncel saat bilgisi
- "Bu metni analiz et: Lorem ipsum..."

### Image Analysis (Gemini Vision)
- Upload a photo + "What is this?" (analyzed by Gemini Vision)
- Screenshot + "What does this page say?" (OCR by Gemini Vision)
- Chart/graph + "Explain this graphic" (visual context by Gemini Vision)


## 🎯 AI Capabilities

- **Vision Models**: llama-3.2-90b-vision-preview (text-based), Gemini Vision API (real image understanding)
- **Text Model**: llama-3.1-70b-versatile (70B parametreli!)
- **Dil Algılama**: Otomatik çoklu dil desteği
- **Görsel Okuma**: Yüksek çözünürlük resim, grafik, metin analizi (Gemini Vision ile gerçek OCR ve görsel anlama)
- **Akıllı Yanıtlar**: 4K token ile detaylı bağlam bilincinde cevaplar
- **Hızlı İşleme**: Groq'un optimize edilmiş inference hızı

## �📁 Proje Yapısı

```
llm_agent_project/
├── main.py              # Ana uygulama
├── agents/              # LangChain agent'ları
│   ├── llm_agent.py     # Ana LLM agent (vision + text)
│   └── tools.py         # Gelişmiş agent tool'ları
├── ui/                  # Gradio arayüz
│   └── interface.py     # Görsel yükleme destekli arayüz
├── config/              # Konfigürasyon
│   └── settings.py      # Gelişmiş ayarlar ve promptlar
├── requirements.txt     # Python bağımlılıkları
└── .env                # API anahtarı
```

## 🔧 Gelişmiş Özellikler

### Sistem Promptları
- Otomatik dil algılama ve uygun yanıt
- Görsel analiz için özel talimatlar
- Profesyonel ama samimi ton

### Tool Sistemi
- **Zaman**: Türkçe tarih/saat bilgisi
- **Analiz**: Gelişmiş metin istatistikleri
- **Özet**: Akıllı cümle seçimi
- **Dil**: Çoklu dil algılama
- **Arama**: Temel bilgi sorguları

### Vision AI (Gemini + LLaMA)
- Image upload (file/webcam)
- Automatic size optimization
- Detailed image description (Gemini Vision API)
- Text reading (real OCR with Gemini Vision)
- Visual context and scene understanding

Bu sistem artık profesyonel seviyede bir AI asistan! 🤖✨
