import streamlit as st
import pickle
import re
import string
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Download stopword nltk (jaga-jaga jika belum ada di server streamlit)
nltk.download('stopwords')
list_stopwords = set(stopwords.words('indonesian'))

# Load model yang sudah disimpan sebelumnya
@st.cache_resource
def load_models():
    with open('model_klasifikasi_detik.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['w2v_model'], data['nb_model'], data['vector_size']

w2v_model, nb_model, vector_size = load_models()

# Fungsi Preprocessing (Harus sama persis dengan saat training V1)
def preprocessing_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in list_stopwords]
    return tokens

# Fungsi untuk mengambil teks berita dari URL Detik via Web Scraping
def get_detik_news_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Detik biasanya meletakkan isi berita utama di dalam tag div dengan class 'detail__body-text' atau 'itp_bodycontent'
        article_body = soup.find('div', class_='detail__body-text')
        if not article_body:
            # Alternatif struktur div detik yang lain
            article_body = soup.find('div', class_='itp_bodycontent')
            
        if article_body:
            # Ambil semua paragraf di dalam badan berita
            paragraphs = article_body.find_all('p')
            text_content = " ".join([p.get_text() for p in paragraphs])
            return text_content
        else:
            return None
    except Exception as e:
        return None

# Fungsi untuk mengubah token menjadi vektor dokumen
def get_document_vector(tokens, model, v_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(v_size)
    return np.mean(vectors, axis=0)

# --- TAMPILAN ANTARMUKA STREAMLIT ---
st.title("📰 Klasifikasi Kategori Berita Detik.com")
st.write("Masukkan link berita Detik (Kategori **Sport** atau **Finance**) untuk diuji prediksinya menggunakan Word2Vec Skip-gram & Naive Bayes.")

# Input Form URL dari user
url_input = st.text_input("Masukkan URL Berita Detik:", placeholder="https://sport.detik.com/... atau https://finance.detik.com/...")

if st.button("Prediksi Kategori"):
    if not url_input:
        st.warning("⚠️ Mohon masukkan URL terlebih dahulu!")
    elif "detik.com" not in url_input:
        st.error("❌ Link tidak valid! Pastikan itu adalah URL dari domain detik.com.")
    else:
        with st.spinner("Sedang mengambil dan menganalisis berita..."):
            # 1. Scraping teks dari URL
            raw_text = get_detik_news_text(url_input)
            
            if not raw_text or len(raw_text.strip() < 50):
                st.error("❌ Gagal mengambil isi berita atau halaman tidak ditemukan. Coba URL yang lain.")
            else:
                # 2. Preprocessing
                tokens = preprocessing_text(raw_text)
                
                # 3. Ubah ke Vektor Word2Vec
                doc_vector = get_document_vector(tokens, w2v_model, vector_size).reshape(1, -1)
                
                # 4. Prediksi dengan Naive Bayes
                prediction = nb_model.predict(doc_vector)[0]
                probabilities = nb_model.predict_proba(doc_vector)[0]
                confidence = np.max(probabilities) * 100
                
                # Validasi tambahan: Jika keyakinan model rendah atau di luar konteks olahraga/keuangan
                # (Model kita dilatih khusus untuk sport dan finance)
                st.success("Analisis Selesai!")
                
                st.markdown("### Hasil Prediksi:")
                st.info(f"Kategori Terdeteksi: **{prediction.upper()}** (Tingkat Keyakinan: {confidence:.2f}%)")
                
                # Menampilkan cuplikan teks hasil scraping
                with st.expander("Lihat Cuplikan Isi Berita"):
                    st.write(raw_text[:600] + "...")