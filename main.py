"""
Authors:Ferhat Koknar,Hamza Gunes
Created 19/12/2024
"""
import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
from TurkishStemmer import TurkishStemmer
from nltk.corpus import stopwords
import nltk

# NLTK ve Stemmer için gerekli yüklemeler
# Necessary downloads for NLTK and Stemmer
nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))  # Türkçe stop words (stop words in Turkish)
stemmer = TurkishStemmer()

# Veritabanı bağlantısını kurma
# Establish database connection
def create_db_connection():
    try:
        conn = sqlite3.connect("data/data.db")
        return conn
    except sqlite3.Error as e:
        messagebox.showerror("Veritabanı Hatası", f"Veritabanı bağlantısı hatası: {e}")
        return None

# Metin temizleme işlevi
# Text cleaning function
def clean_text(text):
    try:
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^a-zçığıöşü\s]', '', text)  # Remove non-alphabetic characters
        words = text.split()  # Split the text into words
        words = [word for word in words if word not in turkish_stopwords]  # Remove stop words
        words = [stemmer.stem(word) for word in words]  # Stem the words
        return ' '.join(words)
    except Exception as e:
        messagebox.showerror("Hata", f"Metin temizleme hatası: {e}")
        return ""

# Veritabanından veri yükleme
# Load data from the database
def load_from_database(conn):
    try:
        query = "SELECT * FROM text_data"
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        messagebox.showerror("Veritabanı Hatası", f"Veritabanından veri yüklenemedi: {e}")
        return pd.DataFrame()

# Model eğitme fonksiyonu
# Function to train the model
def train_model(df):
    try:
        df['label'] = df['label'].map({'Positive': 1, 'Negative': 0, 'Notr': 2})  # Convert labels to numeric
        df['cleaned_text'] = df['text'].apply(clean_text)  # Clean the text
        X_train = df['cleaned_text']
        y_train = df['label']

        # Vektörleştirici ve model eğitimi
        # Vectorizer and model training
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        # Model ve vektörleştirici kaydetme
        # Save the model and vectorizer
        joblib.dump(model, 'models/sentiment_model.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        print("Model başarıyla eğitildi ve kaydedildi.")  # Model successfully trained and saved
        return model, vectorizer
    except Exception as e:
        messagebox.showerror("Model Eğitimi Hatası", f"Model eğitimi sırasında bir hata oluştu: {e}")
        return None, None

# Metin sınıflandırma işlevi
# Function to classify text
def classify_text(input_text, model, vectorizer):
    try:
        cleaned_input_text = clean_text(input_text)  # Clean the input text
        input_vectorized = vectorizer.transform([cleaned_input_text])  # Vectorize the cleaned text
        prediction = model.predict(input_vectorized)  # Predict the sentiment
        return "Negative" if prediction == 0 else "Positive" if prediction == 1 else "Notr"
    except Exception as e:
        messagebox.showerror("Sınıflandırma Hatası", f"Metin sınıflandırma hatası: {e}")
        return ""

# Tkinter GUI sınıfı
# Tkinter GUI class
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis")  # Set the window title
        self.conn = create_db_connection()  # Establish database connection
        self.model, self.vectorizer = self.load_model()  # Load the model and vectorizer

        # Kapanma işlemi için on_closing fonksiyonu atanıyor
        # Assign the on_closing function for closing the application
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # GUI bileşenlerini oluşturma
        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Metin Analizine Hoşgeldiniz.", font=("Arial", 18))  # Create label
        self.label.pack(pady=10)
        
        self.label = tk.Label(self.root, text="Lütfen bir işlem seçin:", font=("Arial", 14))  # Create label
        self.label.pack(pady=10)

        self.classify_button = tk.Button(self.root, text="Metni Sınıflandır", command=self.classify_text_ui, width=20)  # Button for classification
        self.classify_button.pack(pady=5)

        self.add_data_button = tk.Button(self.root, text="Yeni Veri Ekle", command=self.add_data_ui, width=20)  # Button to add data
        self.add_data_button.pack(pady=5)

        self.train_button = tk.Button(self.root, text="Modeli Yeniden Eğit", command=self.retrain_model_ui, width=20)  # Button to retrain the model
        self.train_button.pack(pady=5)

        self.exit_button = tk.Button(self.root, text="Çıkış", command=self.on_closing, width=20)  # Button to exit
        self.exit_button.pack(pady=20)

    def load_model(self):
        try:
            model = joblib.load('models/sentiment_model.pkl')  # Load the model
            vectorizer = joblib.load('models/vectorizer.pkl')  # Load the vectorizer
            print("Model yüklendi.")  # Model loaded
            return model, vectorizer
        except (FileNotFoundError, Exception) as e:
            messagebox.showerror("Model Yükleme Hatası", f"Model yüklenemedi: {e}")
            return None, None

    def classify_text_ui(self):
        if not self.model or not self.vectorizer:
            messagebox.showerror("Hata", "Model yüklenemedi. Önce modeli eğitin.")  # Error message if model is not loaded
            return

        input_text = simpledialog.askstring("Metni Sınıflandır", "Lütfen sınıflandırmak istediğiniz metni girin:")  # Ask for text input
        if input_text and input_text.isdigit():  # Check if input is digits only
            messagebox.showwarning("Uyarı", "Lütfen sadece sayılardan oluşmayan bir metin girin.")  # Warning for numeric input
            return

        if input_text:
            prediction = classify_text(input_text, self.model, self.vectorizer)  # Classify the input text
            messagebox.showinfo("Sonuç", f"Tahmini sınıf: {prediction}")  # Show the prediction result
        else:
            messagebox.showwarning("Uyarı", "Metin girişi boş olamaz.")  # Warning for empty input

    def add_data_ui(self):
        text = simpledialog.askstring("Yeni Veri", "Lütfen metni girin:")  # Ask for text input
        label = simpledialog.askstring("Yeni Veri", "Lütfen etiketi girin (Positive, Negative, Notr):")  # Ask for label

        if text and label:
            if label not in ["Positive", "Negative", "Notr"]:  # Validate label
                messagebox.showerror("Hata", "Geçersiz etiket. Lütfen 'Positive', 'Negative' veya 'Notr' girin.")  # Error for invalid label
                return
            new_data = pd.DataFrame([[text, label]], columns=["text", "label"])  # Create new data entry
            self.save_to_database(new_data)  # Save the new data to the database
            messagebox.showinfo("Başarılı", "Yeni veri başarıyla eklendi.")  # Success message
        else:
            messagebox.showwarning("Uyarı", "Metin ve etiket boş bırakılamaz.")  # Warning for empty fields

    def save_to_database(self, df):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    label TEXT
                )
            """)
            self.conn.commit()
            df.to_sql("text_data", self.conn, if_exists="append", index=False)  # Save the data to the database
        except sqlite3.Error as e:
            messagebox.showerror("Veritabanı Hatası", f"Veritabanına veri eklenirken hata oluştu: {e}")  # Error during database insertion

    def retrain_model_ui(self):
        df = load_from_database(self.conn)  # Load data from the database
        if df.empty:
            messagebox.showwarning("Uyarı", "Veritabanında eğitim için yeterli veri bulunamadı.")  # Warning if there's no data
            return

        self.model, self.vectorizer = train_model(df)  # Retrain the model
        if self.model and self.vectorizer:
            messagebox.showinfo("Başarılı", "Model başarıyla yeniden eğitildi.")  # Success message
        else:
            messagebox.showerror("Model Eğitimi Hatası", "Model eğitilemedi. Lütfen tekrar deneyin.")  # Error during retraining

    def on_closing(self):
        try:
            if messagebox.askokcancel("Çıkış", "Programdan çıkmak istediğinize emin misiniz?"):  # Confirm exit
                self.conn.close()  # Close the database connection
                self.root.destroy()  # Close the Tkinter window
        except Exception as e:
            print(f"Bir hata oluştu: {e}")  # Error during closing
            self.root.destroy()

# Tkinter ana döngüsü
# Tkinter main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
