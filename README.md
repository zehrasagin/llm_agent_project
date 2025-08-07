# CevapLlama 3.3 Vision - Ultra Advanced AI

Bu proje Gradio arayüzü kullanarak kullanıcıdan metin ve görsel alır, Groq AI üzerinden **LLaMA 3.3** serisi modelleri kullanır ve LangChain agent yapısı ile güçlendirilmiş bir sistem sunar.

## 🚀 LLaMA 3.3 Serisi Modelleri

- **🧠 llama-3.3-70b-versatile**: En güçlü genel amaçlı model (70B param)
- **⚡ llama-3.3-70b-specdec**: Hızlı speculative decoding (70B param)
- **🤖 llama-3.1-405b-reasoning**: Muhakeme odaklı dev model (405B param!)
- **👁️ llama-3.2-90b-vision-preview**: En gelişmiş görsel analiz (90B param)
- **🛠️ llama3-groq-70b-8192-tool-use-preview**: Tool kullanımı optimize
- **💾 8K Token Limit**: Uzun metinler ve detaylı analiz

## 🚀 Yeni Özellikler

- **🖼️ Görsel Analiz**: Resim yükleyip ne olduğunu öğrenin
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

### Görsel Analiz
- Fotoğraf yükleyin + "Bu nedir?"
- Ekran görüntüsü + "Bu sayfada ne yazıyor?"
- Grafik + "Bu grafiği açıkla"

## 🎯 AI Yetenekleri

- **Vision Model**: llama-3.2-90b-vision-preview (90B parametreli!)
- **Text Model**: llama-3.1-70b-versatile (70B parametreli!)
- **Dil Algılama**: Otomatik çoklu dil desteği
- **Görsel Okuma**: Yüksek çözünürlük resim, grafik, text analizi  
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

### Vision AI
- Resim yükleme (dosya/webcam)
- Otomatik boyut optimizasyonu
- Detaylı görsel açıklama
- Yazı okuma (OCR benzeri)

Bu sistem artık profesyonel seviyede bir AI asistan! 🤖✨
