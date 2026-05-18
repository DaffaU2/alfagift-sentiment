# ============================================================
#  Dashboard Analitik Sentimen Ulasan Pengguna Aplikasi Alfagift
#  Framework : Streamlit
#  Metode    : SVM + TF-IDF
#  Penulis   : (Muhammad Daffa Umam)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi halaman ────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Sentimen Alfagift",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Warna tema Alfagift ─────────────────────────────────────
MERAH   = "#E8192C"
HIJAU   = "#639922"
MERAH_M = "#E24B4A"
ABU     = "#6B7280"

# ── CSS Custom ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

/* ══ FORCE LIGHT MODE ══ */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container,
[class*="css"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E7EB !important;
    color: #111827 !important;
}
[data-testid="stHeader"] { background-color: #FFFFFF !important; }
[data-testid="stToolbar"] { background-color: #FFFFFF !important; }
[data-testid="metric-container"] {
    background-color: #F9FAFB !important;
    border-radius: 8px; padding: 8px;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 700 !important;
    color: #111827 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] { color: #374151 !important; }
p, span, label, div, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p { color: #111827 !important; }
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background-color: #FFFFFF !important; color: #111827 !important;
}
[data-testid="stDataFrame"], .stDataFrame iframe { background-color: #FFFFFF !important; }
hr { border-color: #E5E7EB !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F3F4F6; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }

.alfagift-title {
    font-size: 1.8rem; font-weight: 800;
    color: #E8192C !important; letter-spacing: -0.5px;
}
.subtitle { color: #6B7280 !important; font-size: 0.9rem; margin-top: -6px; }
.section-header {
    font-size: 1rem; font-weight: 700; color: #111827 !important;
    border-left: 4px solid #E8192C;
    padding-left: 10px; margin-bottom: 8px;
}
.footer { text-align:center; color:#9CA3AF !important; font-size:0.75rem; padding-top:20px; }
.result-positif {
    background: #EAF3DE; border-left: 6px solid #639922;
    padding: 14px 18px; border-radius: 0 10px 10px 0;
    font-size: 1.1rem; font-weight: 700; color: #2d6a00 !important;
}
.result-negatif {
    background: #FCEBEB; border-left: 6px solid #E8192C;
    padding: 14px 18px; border-radius: 0 10px 10px 0;
    font-size: 1.1rem; font-weight: 700; color: #9b1c1c !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data_dashboard2.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Bulan'] = df['Date'].dt.to_period('M').astype(str)
    return df

@st.cache_resource
def load_model():
    model      = joblib.load("model_svm.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    df = load_data()
    svm_model, tfidf_vec = load_model()
    data_loaded = True
except FileNotFoundError as e:
    st.error(
        f"⚠️ File tidak ditemukan: {e}\n\n"
        "Pastikan file berikut ada di folder yang sama dengan `app.py`:\n"
        "- `data_dashboard2.csv`\n- `model_svm.pkl`\n"
        "- `tfidf_vectorizer.pkl`\n- `kamuskatabaku.xlsx`"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════
# FUNGSI OTOMATIS: Buat deskripsi topik dari data
# ═══════════════════════════════════════════════════════════
# Kata yang diabaikan karena terlalu umum
STOPWORDS_CUSTOM = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
    'untuk', 'tidak', 'ada', 'sudah', 'bisa', 'akan', 'juga',
    'lebih', 'kalau', 'jadi', 'banget', 'sangat', 'sekali',
    'saja', 'masih', 'belum', 'harus', 'agar', 'karena', 'tapi',
    'kami', 'saya', 'kita', 'mereka', 'anda', 'nya', 'pun',
    'mau', 'buat', 'atas', 'bawah', 'sama', 'lagi', 'dong',
    'tolong', 'mohon', 'alfa', 'alfagift', 'alfamart', 'aplikasi',
    'barang', 'pesan', 'belanja', 'beli', 'layan', 'toko',
}

def buat_deskripsi_otomatis(df_kat, nama_kategori, n_kata=5):
    """
    Membuat deskripsi otomatis berdasarkan:
    - Kata paling sering muncul di kategori tersebut
    - Sentimen dominan (positif/negatif)
    """
    if df_kat.empty:
        return "Tidak ada data untuk kategori ini."

    # Hitung sentimen dominan
    sentimen_count = df_kat['Prediksi_Sentimen'].value_counts()
    sentimen_dominan = sentimen_count.index[0] if len(sentimen_count) > 0 else 'N/A'
    pct_dominan = sentimen_count.iloc[0] / len(df_kat) * 100

    # Ambil kata paling sering dari stemming_data
    semua_teks = ' '.join(df_kat['stemming_data'].fillna('').tolist())
    kata_list = [
        w for w in semua_teks.split()
        if len(w) > 3 and w.lower() not in STOPWORDS_CUSTOM
    ]
    top_kata = [w for w, _ in Counter(kata_list).most_common(n_kata)]

    if not top_kata:
        return f"Sentimen {sentimen_dominan.lower()} mendominasi ({pct_dominan:.0f}%)."

    kata_str = ', '.join(top_kata)

    # Buat deskripsi berdasarkan sentimen dominan
    if sentimen_dominan == 'Positif':
        deskripsi = f"Mayoritas ulasan positif ({pct_dominan:.0f}%). Topik utama: {kata_str}."
    else:
        deskripsi = f"Mayoritas ulasan negatif ({pct_dominan:.0f}%). Isu utama: {kata_str}."

    return deskripsi


def warna_kategori(df_kat):
    """Tentukan warna badge berdasarkan sentimen dominan"""
    if df_kat.empty:
        return ('⚪', '#F3F4F6', '#374151')
    sentimen_dominan = df_kat['Prediksi_Sentimen'].value_counts().index[0]
    if sentimen_dominan == 'Positif':
        return ('🟢', '#EAF3DE', '#2d6a00')
    else:
        return ('🔴', '#FCEBEB', '#9b1c1c')


# ═══════════════════════════════════════════════════════════
# SIDEBAR — NAVIGASI & FILTER
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding-bottom:16px;
                border-bottom:1px solid #E5E7EB;margin-bottom:16px;'>
        <div style='width:42px;height:42px;background:#E8192C;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 2px 8px rgba(232,25,44,0.3);'>
            <span style='color:white;font-weight:800;font-size:14px;'>AG</span>
        </div>
        <div>
            <div style='font-weight:800;font-size:1rem;color:#111827;'>Alfagift</div>
            <div style='font-size:0.72rem;color:#6B7280;'>Sentiment Monitor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔧 Filter Data")

    semua_kategori = ['Semua Kategori'] + sorted(df['Kategori'].unique().tolist())
    filter_kategori = st.selectbox("📂 Kategori Ulasan", semua_kategori)

    st.markdown("⭐ **Filter Rating**")
    filter_rating = st.multiselect(
        "Pilih rating:",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{'⭐'*x} ({x})"
    )

    bulan_list = sorted(df['Bulan'].dropna().unique().tolist())
    if len(bulan_list) >= 2:
        filter_bulan = st.select_slider(
            "📅 Periode",
            options=bulan_list,
            value=(bulan_list[0], bulan_list[-1])
        )
    else:
        filter_bulan = (bulan_list[0], bulan_list[-1]) if bulan_list else (None, None)

    filter_sentimen = st.radio(
        "💬 Tampilkan Sentimen",
        ["Semua", "Positif", "Negatif"],
        horizontal=True
    )

    st.divider()
    st.markdown("""
    <div class='footer'>
        Metode: SVM + TF-IDF<br>
        Dataset: Google Play Store
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TERAPKAN FILTER
# ═══════════════════════════════════════════════════════════
df_filtered = df.copy()

if filter_kategori != 'Semua Kategori':
    df_filtered = df_filtered[df_filtered['Kategori'] == filter_kategori]
if filter_rating:
    df_filtered = df_filtered[df_filtered['Rating'].isin(filter_rating)]
if filter_sentimen != "Semua":
    df_filtered = df_filtered[df_filtered['Prediksi_Sentimen'] == filter_sentimen]
if filter_bulan[0] and filter_bulan[1]:
    df_filtered = df_filtered[
        (df_filtered['Bulan'] >= filter_bulan[0]) &
        (df_filtered['Bulan'] <= filter_bulan[1])
    ]


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("<div class='alfagift-title'>🛒 Dashboard Analisis Sentimen Alfagift</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analisis Sentimen Ulasan Pengguna Play Store · SVM (Linear, C=1) · TF-IDF</div>", unsafe_allow_html=True)
with col_badge:
    st.metric("Total Data Asli", f"{len(df):,} ulasan")

st.divider()

total_filtered = len(df_filtered)
if total_filtered == 0:
    st.warning("⚠️ Tidak ada data yang sesuai filter. Coba ubah pilihan filter.")
    st.stop()

jml_pos = (df_filtered['Prediksi_Sentimen'] == 'Positif').sum()
jml_neg = (df_filtered['Prediksi_Sentimen'] == 'Negatif').sum()
pct_pos = jml_pos / total_filtered * 100
pct_neg = jml_neg / total_filtered * 100

AKURASI  = 89.9
PRESISI  = 88.21
RECALL   = 80.65
F1_SCORE = 83.61

# ── Ringkasan Sentimen ──
st.markdown("<div class='section-header'>📊 Ringkasan Sentimen</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("😊 Positif",     f"{pct_pos:.1f}%", f"{jml_pos:,} ulasan")
c2.metric("😠 Negatif",     f"{pct_neg:.1f}%", f"{jml_neg:,} ulasan")
c3.metric("🎯 Akurasi SVM", f"{AKURASI}%")
c4.metric("📐 Presisi",     f"{PRESISI}%")
c5.metric("🔁 F1-Score",    f"{F1_SCORE}%")

st.divider()

# ── Grafik Sentimen + Topik ──
st.markdown("<div class='section-header'>📈 Grafik Sentimen</div>", unsafe_allow_html=True)
col_bar, col_pie, col_topik = st.columns([1.3, 1, 1.2])

with col_bar:
    distribusi = df_filtered['Prediksi_Sentimen'].value_counts()
    fig_bar, ax_bar = plt.subplots(figsize=(4.5, 3.2))
    bar_colors = [HIJAU if l == 'Positif' else MERAH_M for l in distribusi.index]
    bars = ax_bar.bar(distribusi.index, distribusi.values,
                      color=bar_colors, width=0.45, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, distribusi.values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_bar.set_title('Distribusi Sentimen', fontsize=11, fontweight='bold', pad=8)
    ax_bar.set_ylabel('Jumlah Ulasan', fontsize=9)
    ax_bar.tick_params(axis='both', labelsize=9)
    ax_bar.spines[['top','right']].set_visible(False)
    ax_bar.set_ylim(0, distribusi.max() * 1.18)
    ax_bar.set_facecolor('#FAFAFA')
    fig_bar.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

with col_pie:
    fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.2))
    vals   = [jml_pos, jml_neg]
    labels = [f'Positif\n{pct_pos:.1f}%', f'Negatif\n{pct_neg:.1f}%']
    colors = [HIJAU, MERAH_M]
    wedges, texts = ax_pie.pie(
        vals, labels=None, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
    )
    ax_pie.set_title('Proporsi Sentimen', fontsize=11, fontweight='bold', pad=8)
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax_pie.legend(handles=legend_patches, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    fig_pie.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    st.pyplot(fig_pie, use_container_width=True)
    plt.close()

with col_topik:
    # ── DESKRIPSI TOPIK OTOMATIS ──────────────────────────
    st.markdown("**🗂️ Topik Utama per Kategori**")
    kategori_top4 = df_filtered['Kategori'].value_counts().head(4).index.tolist()

    for kat in kategori_top4:
        df_kat = df_filtered[df_filtered['Kategori'] == kat]
        ikon, bg, fg = warna_kategori(df_kat)
        deskripsi    = buat_deskripsi_otomatis(df_kat, kat, n_kata=5)
        jumlah       = len(df_kat)

        st.markdown(f"""
        <div style='background:{bg};border-radius:8px;padding:8px 12px;margin-bottom:6px;'>
            <span style='font-weight:700;color:{fg};font-size:0.82rem;'>{ikon} {kat} ({jumlah} ulasan)</span><br>
            <span style='font-size:0.75rem;color:{fg};opacity:0.9;'>{deskripsi}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── WordCloud + Frekuensi Kata ──
st.markdown("<div class='section-header'>☁️ Word Cloud & Kata Sering Muncul</div>", unsafe_allow_html=True)
col_wc1, col_wc2, col_freq = st.columns([1, 1, 1])

def buat_wordcloud(teks_series, colormap, bg):
    teks = ' '.join(teks_series.fillna('').tolist())
    if not teks.strip():
        return None
    wc = WordCloud(
        width=500, height=300,
        background_color=bg,
        colormap=colormap,
        max_words=80,
        min_font_size=8
    ).generate(teks)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor(bg)
    plt.tight_layout(pad=0)
    return fig

with col_wc1:
    st.markdown("**😊 Kata Positif**")
    teks_pos = df_filtered[df_filtered['Prediksi_Sentimen'] == 'Positif']['stemming_data']
    fig_wc = buat_wordcloud(teks_pos, 'Greens', '#0a1f0a')
    if fig_wc:
        st.pyplot(fig_wc, use_container_width=True)
        plt.close()
    else:
        st.info("Tidak ada data positif.")

with col_wc2:
    st.markdown("**😠 Kata Negatif**")
    teks_neg = df_filtered[df_filtered['Prediksi_Sentimen'] == 'Negatif']['stemming_data']
    fig_wc2 = buat_wordcloud(teks_neg, 'Reds', '#1f0a0a')
    if fig_wc2:
        st.pyplot(fig_wc2, use_container_width=True)
        plt.close()
    else:
        st.info("Tidak ada data negatif.")

with col_freq:
    st.markdown("**📊 Top 10 Kata Terbanyak**")
    semua_teks = ' '.join(df_filtered['stemming_data'].fillna('').tolist())
    kata_list  = semua_teks.split()
    kata_count = Counter(kata_list).most_common(10)
    if kata_count:
        words, counts = zip(*kata_count)
        fig_freq, ax_freq = plt.subplots(figsize=(4.5, 3.5))
        y_pos = range(len(words))
        bar_h = ax_freq.barh(y_pos, counts, color=MERAH, alpha=0.85, edgecolor='white')
        ax_freq.set_yticks(y_pos)
        ax_freq.set_yticklabels(words, fontsize=9)
        ax_freq.invert_yaxis()
        ax_freq.set_xlabel('Frekuensi', fontsize=8)
        ax_freq.tick_params(axis='x', labelsize=8)
        ax_freq.spines[['top','right']].set_visible(False)
        ax_freq.set_facecolor('#FAFAFA')
        for i, (bar, cnt) in enumerate(zip(bar_h, counts)):
            ax_freq.text(cnt + 0.5, i, str(cnt), va='center', fontsize=8)
        fig_freq.patch.set_facecolor('#FFFFFF')
        plt.tight_layout()
        st.pyplot(fig_freq, use_container_width=True)
        plt.close()

st.divider()

# ── Tren Sentimen per Bulan ──
st.markdown("<div class='section-header'>📅 Tren Sentimen per Bulan</div>", unsafe_allow_html=True)
df_tren = df_filtered.groupby(['Bulan','Prediksi_Sentimen']).size().unstack(fill_value=0)

if not df_tren.empty and len(df_tren) > 1:
    fig_tren, ax_tren = plt.subplots(figsize=(10, 3.5))
    if 'Positif' in df_tren.columns:
        ax_tren.plot(df_tren.index, df_tren['Positif'],
                     color=HIJAU, marker='o', linewidth=2, label='Positif', markersize=6)
    if 'Negatif' in df_tren.columns:
        ax_tren.plot(df_tren.index, df_tren['Negatif'],
                     color=MERAH_M, marker='s', linewidth=2, label='Negatif',
                     linestyle='--', markersize=6)
    ax_tren.set_title('Tren Jumlah Ulasan per Bulan', fontsize=11, fontweight='bold', pad=8)
    ax_tren.set_xlabel('Periode', fontsize=9)
    ax_tren.set_ylabel('Jumlah Ulasan', fontsize=9)
    ax_tren.legend(fontsize=9, frameon=False)
    ax_tren.tick_params(axis='both', labelsize=8)
    ax_tren.spines[['top','right']].set_visible(False)
    ax_tren.set_facecolor('#FAFAFA')
    fig_tren.patch.set_facecolor('#FFFFFF')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig_tren, use_container_width=True)
    plt.close()
else:
    st.info("ℹ️ Tren per bulan memerlukan data dari minimal 2 bulan berbeda.")

st.divider()

# ── Tabel Ulasan + Insight ──
col_tbl, col_insight = st.columns([1.6, 1])

with col_tbl:
    st.markdown("<div class='section-header'>📋 Ulasan Terbaru</div>", unsafe_allow_html=True)
    tampil_n = st.slider("Jumlah ulasan ditampilkan", 5, 50, 10, step=5)
    df_tampil = (df_filtered
                 .sort_values('Date', ascending=False)
                 .head(tampil_n)[['Username','Rating','Review Text','Prediksi_Sentimen','Kategori']]
                 .copy())

    def style_sentimen(val):
        if val == 'Positif':
            return 'background-color:#EAF3DE; color:#2d6a00; font-weight:600;'
        elif val == 'Negatif':
            return 'background-color:#FCEBEB; color:#9b1c1c; font-weight:600;'
        return ''

    styled_df = df_tampil.style.map(style_sentimen, subset=['Prediksi_Sentimen'])
    st.dataframe(styled_df, use_container_width=True, height=320)

    csv_dl = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Data Terfilter (CSV)",
        data=csv_dl,
        file_name="data_sentimen_alfagift_filtered.csv",
        mime="text/csv"
    )

with col_insight:
    st.markdown("<div class='section-header'>💡 Insight Kunci</div>", unsafe_allow_html=True)
    kat_negatif = (df_filtered[df_filtered['Prediksi_Sentimen']=='Negatif']['Kategori']
                   .value_counts().idxmax() if jml_neg > 0 else 'N/A')
    kat_positif = (df_filtered[df_filtered['Prediksi_Sentimen']=='Positif']['Kategori']
                   .value_counts().idxmax() if jml_pos > 0 else 'N/A')

    insights = [
        (HIJAU,    f"<b>{pct_pos:.1f}%</b> ulasan bersentimen positif ({jml_pos:,} dari {total_filtered:,} ulasan terfilter)."),
        (MERAH_M,  f"Keluhan terbanyak pada kategori <b>{kat_negatif}</b> — perlu perhatian prioritas."),
        (HIJAU,    f"Apresiasi terbanyak pada kategori <b>{kat_positif}</b> — pertahankan kualitas ini."),
        ("#185FA5", f"Model SVM (Linear, C=1) mencapai akurasi <b>{AKURASI}%</b> dengan F1-Score <b>{F1_SCORE}%</b>."),
        (ABU,      "Distribusi rating cenderung terpolarisasi (banyak bintang 1 dan bintang 5)."),
    ]

    for warna, teks in insights:
        st.markdown(f"""
        <div style='border-left:4px solid {warna};padding:8px 12px;
                    background:#FAFAFA;border-radius:0 6px 6px 0;
                    margin-bottom:8px;font-size:0.83rem;color:#374151;'>
            {teks}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**📂 Distribusi Kategori**")
    kat_count = df_filtered['Kategori'].value_counts().head(6)
    fig_kat, ax_kat = plt.subplots(figsize=(4, 2.8))
    ax_kat.barh(kat_count.index[::-1], kat_count.values[::-1],
                color=MERAH, alpha=0.8, edgecolor='white')
    ax_kat.tick_params(axis='both', labelsize=7)
    ax_kat.spines[['top','right']].set_visible(False)
    ax_kat.set_facecolor('#FAFAFA')
    fig_kat.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    st.pyplot(fig_kat, use_container_width=True)
    plt.close()

st.divider()

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class='footer'>
Dashboard Analitik Sentimen Ulasan Pengguna Aplikasi Alfagift · SVM + TF-IDF · Data: Google Play Store<br>
Dibuat dengan Streamlit · © Muhammad Daffa Umam
</div>
""", unsafe_allow_html=True)