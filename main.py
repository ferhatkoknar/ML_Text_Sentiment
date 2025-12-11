import sqlite3
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox
import joblib
import re
from TurkishStemmer import TurkishStemmer
from nltk.corpus import stopwords
import nltk
import google.generativeai as genai
import threading
import os
from pathlib import Path  # Dosya yolu bulmak için eklendi
from dotenv import load_dotenv

# --- AYARLAR VE API KEY YÜKLEME (GÜNCELLENDİ) ---

# 1. Dosyanın çalıştığı klasörü ve .env yolunu tam olarak bul
current_dir = Path(__file__).resolve().parent
env_path = current_dir / ".env"

# 2. .env dosyasını yükle
load_dotenv(dotenv_path=env_path)

# 3. Anahtarı al
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 4. Kontrol et ve Yapılandır
if not GOOGLE_API_KEY:
    print(f"HATA: .env dosyası bulunamadı veya içi boş! \nAranan yol: {env_path}")
    print("ÇÖZÜM: .env dosyanızın main.py ile aynı klasörde olduğundan ve içinde 'GOOGLE_API_KEY=...' yazdığından emin olun.")
    
    # Hızlı çözüm için API Key'i geçici olarak buraya (aşağıya) yapıştırabilirsiniz:
    # GOOGLE_API_KEY = "BURAYA_UZUN_API_ANAHTARINIZI_YAZIN"
else:
    print("BAŞARILI: API Key yüklendi.")
    genai.configure(api_key=GOOGLE_API_KEY)


# CustomTkinter Ayarları
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# NLTK ve Stemmer Hazırlığı
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

turkish_stopwords = set(stopwords.words('turkish'))
stemmer = TurkishStemmer()


# --- YARDIMCI FONKSİYONLAR ---

def create_db_connection():
    try:
        # DB dosyasının tam yolunu verelim ki hata olmasın
        db_path = current_dir / "data" / "data.db"
        conn = sqlite3.connect(str(db_path))
        return conn
    except sqlite3.Error as e:
        print(f"Veritabanı hatası: {e}")
        return None

def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zçığıöşü\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in turkish_stopwords]
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)
    except Exception:
        return ""

def classify_local(input_text, model, vectorizer):
    """Yerel Naive Bayes modeli ile tahmin yapar."""
    try:
        cleaned_input_text = clean_text(input_text)
        input_vectorized = vectorizer.transform([cleaned_input_text])
        prediction = model.predict(input_vectorized)
        
        # Olasılık değerlerini de alalım (Güven skoru için)
        proba = model.predict_proba(input_vectorized).max()
        
        label = "Negative" if prediction == 0 else "Positive" if prediction == 1 else "Notr"
        return label, proba
    except Exception as e:
        return "Hata", 0.0

from google.generativeai.types import HarmCategory, HarmBlockThreshold # Bu importları eklemeye gerek yok, string olarak vereceğiz.

def ask_llm(input_text, local_prediction):
    """Gemini API'ye sorar ve değerlendirme ister. (Güçlendirilmiş Versiyon)"""
    
    if not GOOGLE_API_KEY:
        return "HATA: API Key bulunamadı! Lütfen .env dosyasını kontrol edin."

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # MODEL SEÇİMİ: Listenizde olan ve en stabil modellerden biri
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # GÜVENLİK AYARLARI: Gereksiz engellemeleri kaldırmak için 'BLOCK_NONE' yapıyoruz
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

        prompt = f"""
        Sen uzman bir Türkçe Duygu Analizi asistanısın.
        
        Analiz edilecek metin: "{input_text}"
        
        Bizim yerel makine öğrenmesi modelimiz (Naive Bayes) bu metni şu şekilde etiketledi: "{local_prediction}"
        
        Görevin:
        1. Metnin duygu durumunu sen de analiz et (Positive, Negative veya Notr).
        2. Yerel modelin tahmini doğru mu yanlış mı değerlendir.
        3. Kısa bir açıklama yap.
        
        Cevabını şu formatta ver:
        LLM Tahmini: [Senin Kararın]
        Yerel Model: [Doğru/Yanlış]
        Açıklama: [Kısa açıklaman]
        """
        
        # İsteği gönderirken güvenlik ayarlarını ekliyoruz
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # HATA KONTROLÜ: Cevap boş mu dolu mu kontrol et
        if response.parts:
            return response.text
        else:
            # Eğer cevap boşsa ama hata yoksa, muhtemelen başka bir filtreye takıldı
            print(f"DEBUG: Model boş cevap döndü. Finish Reason: {response.candidates[0].finish_reason}")
            return "Yapay zeka boş bir cevap döndürdü. Lütfen tekrar deneyin."

    except Exception as e:
        # Hata mesajını yakala ve kullanıcıya göster
        err_msg = str(e)
        if "429" in err_msg:
            return "HATA: Google API kotası doldu (429). Lütfen 1 dakika bekleyip tekrar deneyin."
        return f"Bağlantı Hatası: {err_msg}"

# --- ARAYÜZ SINIFI (MODERN UI) ---

class ModernSentimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere Ayarları
        self.title("Hibrit Duygu Analizi (ML + LLM)")
        self.geometry("900x650")
        
        # Veri ve Model Yükleme
        self.conn = create_db_connection()
        self.model, self.vectorizer = self.load_model()

        # Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Title
        self.grid_rowconfigure(1, weight=1) # Content

        self.create_widgets()

    def load_model(self):
        try:
            # Model dosyalarının tam yolunu bul
            model_path = current_dir / 'models' / 'sentiment_model.pkl'
            vec_path = current_dir / 'models' / 'vectorizer.pkl'
            
            if not model_path.exists():
                print(f"Model dosyası bulunamadı: {model_path}")
                return None, None

            model = joblib.load(model_path)
            vectorizer = joblib.load(vec_path)
            return model, vectorizer
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None, None

    def create_widgets(self):
        # --- BAŞLIK ALANI ---
        self.header_frame = ctk.CTkFrame(self, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="Türkçe Metin Duygu Analizi", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)

        # --- ANA İÇERİK ALANI ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)
        
        # Metin Girişi
        self.input_label = ctk.CTkLabel(self.content_frame, text="Analiz edilecek metni girin:", font=ctk.CTkFont(size=14))
        self.input_label.pack(anchor="w", pady=(0, 5))
        
        self.text_input = ctk.CTkTextbox(self.content_frame, height=100, font=ctk.CTkFont(size=14))
        self.text_input.pack(fill="x", pady=(0, 20))

        # Analiz Butonu
        self.analyze_btn = ctk.CTkButton(self.content_frame, text="ANALİZ ET (Local + LLM)", 
                                         command=self.start_analysis, 
                                         height=50, 
                                         font=ctk.CTkFont(size=16, weight="bold"),
                                         fg_color="#1a73e8", hover_color="#155bb5")
        self.analyze_btn.pack(fill="x", pady=(0, 30))

        # --- SONUÇ KARTLARI ---
        self.results_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True)
        
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=1)

        # Sol Kart: Yerel Model
        self.local_card = ctk.CTkFrame(self.results_frame)
        self.local_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.lbl_local_title = ctk.CTkLabel(self.local_card, text="Yerel Model (Naive Bayes)", font=ctk.CTkFont(size=16, weight="bold"), text_color="#aaaaaa")
        self.lbl_local_title.pack(pady=15)
        
        self.lbl_local_result = ctk.CTkLabel(self.local_card, text="-", font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_local_result.pack(pady=10)
        
        self.lbl_local_conf = ctk.CTkLabel(self.local_card, text="Güven: -", font=ctk.CTkFont(size=12))
        self.lbl_local_conf.pack(pady=(0, 15))

        # Sağ Kart: Durum / LLM Butonu
        self.status_card = ctk.CTkFrame(self.results_frame)
        self.status_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        self.lbl_status_title = ctk.CTkLabel(self.status_card, text="Sistem Durumu", font=ctk.CTkFont(size=16, weight="bold"), text_color="#aaaaaa")
        self.lbl_status_title.pack(pady=15)
        
        self.lbl_status_msg = ctk.CTkLabel(self.status_card, text="Veri bekleniyor...", font=ctk.CTkFont(size=14))
        self.lbl_status_msg.pack(pady=20)
        
        # Model Eğitimi & Veri Ekleme Butonları (Alt kısım)
        self.footer_frame = ctk.CTkFrame(self, corner_radius=0, height=50)
        self.footer_frame.grid(row=2, column=0, sticky="ew")
        
        self.btn_train = ctk.CTkButton(self.footer_frame, text="Modeli Yeniden Eğit", command=self.dummy_train_popup, fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.btn_train.pack(side="left", padx=20, pady=10)
        
        self.btn_add_data = ctk.CTkButton(self.footer_frame, text="Veri Ekle", command=self.dummy_add_popup, fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.btn_add_data.pack(side="right", padx=20, pady=10)

    def start_analysis(self):
        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Uyarı", "Lütfen bir metin girin.")
            return

        if not self.model:
            messagebox.showerror("Hata", "Model dosyaları bulunamadı. Önce modeli eğitin.")
            return

        # 1. Aşama: Yerel Model Tahmini
        prediction, proba = classify_local(text, self.model, self.vectorizer)
        
        # Sonucu Ekrana Bas
        color = "#2CC985" if prediction == "Positive" else "#FF4C4C" if prediction == "Negative" else "#FFD700"
        self.lbl_local_result.configure(text=prediction, text_color=color)
        self.lbl_local_conf.configure(text=f"Güven Skoru: %{proba*100:.2f}")
        
        self.lbl_status_msg.configure(text="LLM (Yapay Zeka) Doğrulaması Başlatılıyor...")
        self.update() # Arayüzü yenile

        # 2. Aşama: LLM Penceresi Aç ve Sorgula (Thread kullanarak arayüzü dondurma)
        threading.Thread(target=self.open_llm_window, args=(text, prediction)).start()

    def open_llm_window(self, text, local_pred):
        # API Çağrısı
        llm_response = ask_llm(text, local_pred)
        
        # Pencereyi Ana Thread'de açmak için after kullanıyoruz
        self.after(0, lambda: self.show_llm_results(llm_response))

    def show_llm_results(self, response_text):
        self.lbl_status_msg.configure(text="LLM Doğrulaması Tamamlandı.")
        
        # Yeni Pencere Oluştur (Toplevel)
        llm_window = ctk.CTkToplevel(self)
        llm_window.title("Yapay Zeka (LLM) Değerlendirmesi")
        llm_window.geometry("500x400")
        llm_window.attributes("-topmost", True) # Her zaman üstte
        
        title = ctk.CTkLabel(llm_window, text="Gemini AI Değerlendirmesi", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=20)
        
        textbox = ctk.CTkTextbox(llm_window, font=ctk.CTkFont(size=14))
        textbox.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        textbox.insert("0.0", response_text)
        textbox.configure(state="disabled") # Sadece okunabilir

    def dummy_train_popup(self):
        messagebox.showinfo("Bilgi", "Eğitim fonksiyonu bu modern arayüzde arka planda çalıştırılabilir. Eski kodunuzdaki train_model fonksiyonunu buraya entegre edebilirsiniz.")

    def dummy_add_popup(self):
        # Buraya veri ekleme popup'ı gelecek (Eski kodunuzdaki mantıkla aynı)
        dialog = ctk.CTkInputDialog(text="Etiket (Positive/Negative/Notr):", title="Veri Ekle")
        # Basitlik için sadece placeholder
        pass

if __name__ == "__main__":
    app = ModernSentimentApp()
    app.mainloop()