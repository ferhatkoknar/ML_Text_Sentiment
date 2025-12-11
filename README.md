# HİBRİT DUYGU ANALİZİ (Hybrid Sentiment Analysis: ML + LLM)

Bu proje, Geleneksel Makine Öğrenmesi (**Naive Bayes**) ile Üretken Yapay Zeka (**Google Gemini LLM**) teknolojilerini birleştiren hibrit bir **Türkçe Duygu Analizi** sistemidir.

Proje, metinleri **Pozitif**, **Negatif** ve **Nötr** olarak sınıflandırır. İlk aşamada yerel bir model hızlı tahmin yapar, ardından kullanıcı isterse Gemini AI (LLM) devreye girerek bu tahmini doğrular, düzeltir ve detaylı açıklama sunar.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🚀 Özellikler

- **Hibrit Mimari:** Hız için `Naive Bayes`, anlamsal derinlik ve açıklama için `Gemini AI` (LLM) kullanılır.
- **Modern Arayüz (UI):** `CustomTkinter` ile geliştirilmiş, karanlık moda (Dark Mode) sahip, kullanıcı dostu modern arayüz.
- **Doğal Dil İşleme (NLP):** Türkçe metinler için özel temizleme, kök bulma (TurkishStemmer) ve stopword temizliği.
- **Güvenli API Yönetimi:** API anahtarları `.env` dosyası üzerinden güvenli bir şekilde yönetilir.
- **Veri Yönetimi:** SQLite veritabanı ile eğitim verisi saklama ve yeni veri ekleme imkanı.
- **Dinamik Model Seçimi:** Google'ın en güncel ve hızlı modellerini (Gemini Flash) kullanır.

## 🛠 Kullanılan Teknolojiler

* **Dil:** Python
* **Arayüz:** CustomTkinter
* **Makine Öğrenmesi:** Scikit-learn (Naive Bayes, CountVectorizer)
* **Yapay Zeka (LLM):** Google Generative AI (Gemini API)
* **NLP:** NLTK, TurkishStemmer
* **Veritabanı:** SQLite
* **Ortam Yönetimi:** Python-dotenv

## 📸 Ekran Görüntüleri

*(Projenin ekran görüntülerini buraya ekleyebilirsiniz)*

## ⚙️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

 1. Depoyu Klonlayın
```bash
git clone [https://github.com/ferhatkoknar/ML_Text_Sentiment.git](https://github.com/ferhatkoknar/ML_Text_Sentiment.git)
cd ML_Text_Sentiment
```
```bash
2. Gereksinimleri Yükleyin
Bash
```
pip install -r requirements.txt
```bash
3. API Anahtarını Ayarlayın (.env)
Bu proje Google Gemini API kullanır. Google AI Studio adresinden ücretsiz bir API anahtarı alın.
```
Proje ana dizininde .env adında bir dosya oluşturun (uzantısı olmadan sadece .env) ve içine anahtarınızı aşağıdaki formatta ekleyin:

Plaintext

GOOGLE_API_KEY=AIzaSyB.......(Sizin_Anahtariniz)
4. Uygulamayı Başlatın
Bash

python main.py
🧠 Nasıl Çalışır?
Yerel Analiz: Kullanıcı metni girer. Eğitilmiş Naive Bayes modeli metni temizler ve anında bir sınıflandırma yapar (Örn: "Pozitif").

LLM Doğrulaması: Kullanıcı sonucu gördükten sonra, sistem arka planda Google Gemini'ye bağlanır.

Prompt Mühendisliği: Sisteme şu komut gönderilir: "Yerel modelimiz buna 'Pozitif' dedi. Sen ne düşünüyorsun? Doğru mu yanlış mı açıkla."

Sonuç: LLM'in cevabı ayrı bir pencerede detaylı açıklama ile kullanıcıya sunulur.


📂 Proje Yapısı
├── data/
│   └── data.db              # Eğitim verilerinin tutulduğu veritabanı
├── models/
│   ├── sentiment_model.pkl  # Eğitilmiş Naive Bayes modeli
│   └── vectorizer.pkl       # Metin vektörleştirici
├── main.py                  # Ana uygulama dosyası
├── requirements.txt         # Gerekli kütüphaneler
├── .env                     # API Anahtarı (GitHub'a yüklenmez!)
└── README.md                # Proje dokümantasyonu


👥 Yazarlar
Ferhat Köknar -
Hamza Güneş - 
