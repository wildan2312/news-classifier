import streamlit as st
import pickle
import re
import string
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Berita Detik",
    page_icon="📰",
    layout="centered"
)


# ==========================================
# DOWNLOAD STOPWORD NLTK
# ==========================================

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("indonesian"))


list_stopwords = load_stopwords()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_models():

    with open("model_klasifikasi_detik.pkl", "rb") as f:
        data = pickle.load(f)

    return (
        data["w2v_model"],
        data["nb_model"],
        data["vector_size"]
    )


w2v_model, nb_model, vector_size = load_models()


# ==========================================
# PREPROCESSING TEXT
# ==========================================

def preprocessing_text(text):

    if not isinstance(text, str):
        return []

    # lowercase
    text = text.lower()

    # hapus URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # hapus punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # tokenisasi sederhana
    tokens = text.split()

    # hapus stopword
    tokens = [
        word
        for word in tokens
        if word not in list_stopwords
    ]

    return tokens


# ==========================================
# SCRAPING BERITA DETIK
# ==========================================

def get_detik_news_text(url):

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Struktur Detik
        article_body = soup.find(
            "div",
            class_="detail__body-text"
        )

        # Alternatif struktur
        if not article_body:

            article_body = soup.find(
                "div",
                class_="itp_bodycontent"
            )

        if article_body:

            paragraphs = article_body.find_all("p")

            text_content = " ".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
            )

            return text_content

        return None

    except Exception as e:

        st.error(f"Terjadi error saat mengambil berita: {e}")

        return None


# ==========================================
# DOCUMENT VECTOR
# ==========================================

def get_document_vector(
    tokens,
    model,
    v_size
):

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if len(vectors) == 0:

        return np.zeros(v_size)

    return np.mean(
        vectors,
        axis=0
    )


# ==========================================
# USER INTERFACE
# ==========================================

st.title("📰 Klasifikasi Kategori Berita Detik.com")

st.write(
    "Masukkan link berita Detik dari kategori "
    "**Sport** atau **Finance** untuk melakukan "
    "prediksi menggunakan Word2Vec Skip-gram "
    "dan Naive Bayes."
)


# ==========================================
# INPUT URL
# ==========================================

url_input = st.text_input(
    "Masukkan URL Berita Detik:",
    placeholder="https://sport.detik.com/..."
)


# ==========================================
# BUTTON PREDIKSI
# ==========================================

if st.button(
    "🔍 Prediksi Kategori",
    use_container_width=True
):

    if not url_input:

        st.warning(
            "⚠️ Mohon masukkan URL terlebih dahulu!"
        )

    elif "detik.com" not in url_input.lower():

        st.error(
            "❌ Link tidak valid! "
            "Pastikan URL berasal dari domain detik.com."
        )

    else:

        with st.spinner(
            "Sedang mengambil dan menganalisis berita..."
        ):

            # ==================================
            # 1. SCRAPING
            # ==================================

            raw_text = get_detik_news_text(
                url_input
            )

            if (
                not raw_text
                or len(raw_text.strip()) < 50
            ):

                st.error(
                    "❌ Gagal mengambil isi berita "
                    "atau halaman tidak ditemukan. "
                    "Coba URL yang lain."
                )

            else:

                # ==============================
                # 2. PREPROCESSING
                # ==============================

                tokens = preprocessing_text(
                    raw_text
                )

                # ==============================
                # 3. WORD2VEC
                # ==============================

                doc_vector = get_document_vector(
                    tokens,
                    w2v_model,
                    vector_size
                ).reshape(1, -1)

                # ==============================
                # 4. NAIVE BAYES
                # ==============================

                prediction = nb_model.predict(
                    doc_vector
                )[0]

                probabilities = nb_model.predict_proba(
                    doc_vector
                )[0]

                confidence = (
                    np.max(probabilities) * 100
                )

                # ==============================
                # HASIL
                # ==============================

                st.success(
                    "Analisis Selesai!"
                )

                st.markdown(
                    "### Hasil Prediksi:"
                )

                st.info(
                    f"Kategori Terdeteksi: "
                    f"**{str(prediction).upper()}**\n\n"
                    f"Tingkat Keyakinan: "
                    f"**{confidence:.2f}%**"
                )

                # ==============================
                # CUPLIKAN BERITA
                # ==============================

                with st.expander(
                    "📄 Lihat Cuplikan Isi Berita"
                ):

                    st.write(
                        raw_text[:600] + "..."
                    )
