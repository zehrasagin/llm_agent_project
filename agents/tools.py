"""
LangChain + Meta-Llama Maverick Prompt-Based Tool Integration
Tool'lar artık LLM'in kendi yetenekleri ile entegre çalışıyor
"""

from langchain.tools import Tool
from typing import List
import datetime
import json
import requests
from typing import Optional
from groq import Groq
import os

class PromptBasedToolEngine:
    """Gerçek Llama 4 Maverick ile prompt-based tool engine"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-4-maverick-17b-128e-instruct"  # GERÇEK Llama 4 Maverick!

    def _call_llm(self, prompt: str) -> str:
        """Gerçek Llama 4 Maverick'i çağır ve sonucu al"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen gerçek Llama 4 Maverick'sin. 17B parametre, 128K context ile güçlü tool işlemleri yapıyorsun. Sorulan görev için doğru ve yapılandırılmış yanıt ver."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,  # Maverick için artırıldı
                temperature=0.1  # Tool işlemleri için düşük temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"❌ Llama 4 Maverick çağrı hatası: {str(e)}"

# Global tool engine instance
tool_engine = PromptBasedToolEngine()

def get_current_time(_: str = "") -> str:
    """Gerçek Llama 4 Maverick ile akıllı zaman işlemi"""
    prompt = f"""
Şu anki tarih ve saat bilgisini ver:
- Tarih: 7 Ağustos 2025, Perşembe
- Tahmini saat: Günün hangi saati olabileceğini tahmin et (sabah/öğle/akşam context'ine göre)

Sen gerçek Llama 4 Maverick'sin (17B parametre). Şu formatta yanıt ver:
📅 Tarih: [gün] [ay] [yıl]
🗓️ Gün: [gün adı]
🕐 Saat: [tahmin edilen saat]
📍 Zaman Dilimi: Türkiye Saati (UTC+3)
🚀 Model: Llama 4 Maverick 17B

Türkçe ay isimleri kullan: Ocak, Şubat, Mart, Nisan, Mayıs, Haziran, Temmuz, Ağustos, Eylül, Ekim, Kasım, Aralık
"""
    return tool_engine._call_llm(prompt)

def text_analyzer(text: str) -> str:
    """Meta-Llama Maverick ile gelişmiş metin analizi"""
    prompt = f"""
Aşağıdaki metni detaylı analiz et:

METİN:
{text}

Analiz şu bilgileri içermeli:
1. Kelime sayısı
2. Karakter sayısı (boşluklar dahil)
3. Cümle sayısı
4. Ortalama kelime uzunluğu
5. Dil tespiti (Türkçe/İngilizce/Fransızca/Almanca/vb)
6. Ton analizi (resmi/samimi/nötr/akademik)
7. Özel karakterler sayısı (ç,ğ,ı,ö,ş,ü vb)
8. Karmaşıklık seviyesi (basit/orta/karmaşık)

JSON formatında düzenli olarak sun ve ekstra insight'lar ekle.
"""
    return tool_engine._call_llm(prompt)

def language_detector(text: str) -> str:
    """Meta-Llama Maverick ile gelişmiş dil analizi"""
    prompt = f"""
Aşağıdaki metnin dilini ve dil özelliklerini analiz et:

METİN:
{text}

Analiz şu bilgileri içermeli:
1. Ana dil tespiti
2. Güven skoru (yüzde olarak)
3. Tespit edilen dil özellik kelimeleri
4. Karıştırılan diller varsa onları da belirt
5. Metindeki kültürel referanslar
6. Dil seviyesi (günlük/resmi/teknik/akademik)
7. Bölgesel dialekt ipuçları varsa

JSON formatında detaylı rapor ver ve linguistic insight'lar ekle.
"""
    return tool_engine._call_llm(prompt)

def text_summarizer(text: str) -> str:
    """Meta-Llama Maverick ile akıllı özetleme"""
    prompt = f"""
Aşağıdaki metni Maverick seviyede akıllı özetleme ile özetle:

METİN:
{text}

Özetleme kuralları:
1. Ana fikirleri koru ve vurgula
2. Önemli detayları kaçırma
3. Orijinal tonunu muhafaza et
4. Key insight'ları korumaya odaklan
5. Özet oranını belirt (örn: 500→100 kelime)

Şu formatta sun:
📝 ÖZET: [özetlenmiş metin]

📊 İSTATİSTİK:
- Orijinal: [kelime sayısı] kelime
- Özet: [kelime sayısı] kelime  
- Sıkıştırma oranı: [yüzde]%

💡 ANA FİKİRLER:
- [ana fikir 1]
- [ana fikir 2]
- [ana fikir 3]
"""
    return tool_engine._call_llm(prompt)

def web_search(query: str) -> str:
    """Meta-Llama Maverick ile akıllı arama simülasyonu"""
    prompt = f"""
Aşağıdaki arama sorgusuna Maverick seviyede bilgi ver:

SORGU: {query}

Yapman gerekenler:
1. Sorguyu analiz et ve ne tür bilgi arandığını tespit et
2. Genel bilgi databasen'den relevan bilgileri çek
3. Güncel kontekst ile birleştir (2025 Ağustos bazlı)
4. Yapılandırılmış ve değerli yanıt oluştur

Özel konular:
- Hava durumu: Genel tahmin ver
- Teknoloji: Güncel trends dahil et
- Haberler: 2025 kontekstinde genel bilgi
- Programlama: Best practices ve örnekler

Şu formatta yanıt ver:
🔍 ARAMA SONUCU: [ana bilgi]

📊 DETAYLAR:
- [detay 1]
- [detay 2]  
- [detay 3]

� ÖNERİLER:
- [öneri/insight 1]
- [öneri/insight 2]

Eğer bilgi yoksa, en iyi tahminini ve alternatifleri sun.
"""
    return tool_engine._call_llm(prompt)

def create_tools() -> List[Tool]:
    """Meta-Llama Maverick ile prompt-based tool'ları oluştur"""
    
    tools = [
        Tool(
            name="get_current_time",
            description=(
                "Meta-Llama Maverick ile akıllı zaman/tarih işlemi. "
                "Kullanıcı 'Saat kaç?', 'Bugün ne günü?', 'Tarih nedir?' sorduğunda çağır. "
                "LLM ile tahminli saat ve detaylı zaman bilgisi verir."
            ),
            func=get_current_time
        ),
        Tool(
            name="text_analyzer",
            description=(
                "Meta-Llama Maverick ile gelişmiş metin analizi. "
                "Kullanıcı 'Bu metni analiz et', 'Kaç kelime var?', 'İstatistik ver' dediğinde çağır. "
                "LLM ile detaylı analiz, dil tespiti, ton analizi yapar."
            ),
            func=text_analyzer
        ),
        Tool(
            name="text_summarizer",
            description=(
                "Meta-Llama Maverick ile akıllı özetleme. "
                "Kullanıcı 'Özetle', 'Kısalt', 'Summary' istediğinde çağır. "
                "LLM ile key insight'ları koruyarak özetler."
            ),
            func=text_summarizer
        ),
        Tool(
            name="language_detector",
            description=(
                "Meta-Llama Maverick ile gelişmiş dil analizi. "
                "Kullanıcı 'Hangi dilde?', 'Dil analizi yap' dediğinde çağır. "
                "LLM ile kültürel context ve linguistic insight'lar verir."
            ),
            func=language_detector
        ),
        Tool(
            name="web_search",
            description=(
                "Meta-Llama Maverick ile akıllı bilgi arama. "
                "Kullanıcı güncel bilgi, hava durumu, teknoloji, genel sorular sorduğunda çağır. "
                "LLM ile 2025 kontekstinde yapılandırılmış bilgi verir."
            ),
            func=web_search
        )
    ]
    return tools
