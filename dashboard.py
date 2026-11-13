import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================================================================
# KONFIGURASI HALAMAN & CUSTOM UI THEME
# ==============================================================================
st.set_page_config(
    page_title="Bike Sharing Analytics Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Latar belakang halaman utama */
    .stApp {
        background-color: #F1F5F9;
    }

    /* Ruang atas agar lebih proporsional */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Styling Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 55%, #0EA5E9 100%);
        padding: 26px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #E0F2FE;
        margin: 0;
    }
    .filter-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin-top: 12px;
        margin-right: 8px;
        backdrop-filter: blur(4px);
    }

    /* Styling Kartu KPI */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0284C7;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13.5px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
    }

    /* Styling Container Card untuk Grafik */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        padding: 8px 12px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }

    .section-title {
        font-size: 17px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }
    </style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# HELPER FUNCTIONS (AGREGASI DATA)
# ==============================================================================
def create_daily_rentals_df(df):
    day_df = df.drop_duplicates(subset=["dateday"]).copy()
    daily_rentals_df = day_df.resample(rule="D", on="dateday").agg({
        "casual_users_day": "sum",
        "registered_users_day": "sum",
        "total_count_day": "sum"
    }).reset_index()
    return daily_rentals_df

def create_weather_impact_df(df):
    day_df = df.drop_duplicates(subset=["dateday"])
    return (
        day_df.groupby("weather_cond_day", observed=False)["total_count_day"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

def create_season_impact_df(df):
    day_df = df.drop_duplicates(subset=["dateday"])
    return (
        day_df.groupby("season", observed=False)["total_count_day"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

def create_hourly_df(df, day_filter):
    if day_filter == "Hari Libur / Akhir Pekan":
        target_df = df[df["workingday"] == "No"]
    else:
        target_df = df[df["workingday"] == "Yes"]
    return (
        target_df.groupby("hour", observed=False)[["casual_users_hour", "registered_users_hour"]]
        .mean()
        .reset_index()
    )

def create_monthly_rentals_df(df):
    day_df = df.drop_duplicates(subset=["dateday"])
    monthly_df = (
        day_df.groupby(["year", "month"], observed=False)["total_count_day"]
        .sum()
        .unstack(level=0)
    )
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return monthly_df.reindex(month_order)

def create_workingday_prop_df(df):
    day_df = df.drop_duplicates(subset=["dateday"])
    return day_df.groupby("workingday", observed=False)[["casual_users_day", "registered_users_day"]].sum()

def create_time_cluster_df(df):
    return (
        df.groupby("time_category", observed=False)["total_count_hour"]
        .agg(total_rentals="sum", avg_rentals="mean")
        .sort_values(by="total_rentals", ascending=False)
        .reset_index()
    )

# ==============================================================================
# LOAD DATA
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "main_data.csv"))
    df["dateday"] = pd.to_datetime(df["dateday"])
    df["year"] = df["year"].astype(str)
    df.sort_values(by="dateday", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

all_df = load_data()

# ==============================================================================
# SIDEBAR FILTER LENGKAP
# ==============================================================================
min_date = all_df["dateday"].min().date()
max_date = all_df["dateday"].max().date()
year_options = sorted(all_df["year"].dropna().unique().tolist())
season_options = all_df["season"].dropna().unique().tolist()
weather_options = all_df["weather_cond_day"].dropna().unique().tolist()

with st.sidebar:
    st.markdown("### 🚲 Bike Sharing Analytics")
    st.caption("Panel Kontrol & Filter Data")
    st.markdown("---")

    date_range = st.date_input(
        label="📅 Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    selected_years = st.multiselect(
        label="🗓️ Tahun",
        options=year_options,
        default=year_options
    )

    selected_seasons = st.multiselect(
        label="🍂 Musim",
        options=season_options,
        default=season_options
    )

    selected_weather = st.multiselect(
        label="🌤️ Kondisi Cuaca",
        options=weather_options,
        default=weather_options
    )

    selected_day_type = st.radio(
        label="🏢 Kategori Hari",
        options=["Semua Hari", "Hari Kerja", "Hari Libur / Akhir Pekan"],
        index=0
    )

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

main_df = all_df[
    (all_df["dateday"].dt.date >= start_date) &
    (all_df["dateday"].dt.date <= end_date) &
    (all_df["year"].isin(selected_years)) &
    (all_df["season"].isin(selected_seasons)) &
    (all_df["weather_cond_day"].isin(selected_weather))
]

if selected_day_type == "Hari Kerja":
    main_df = main_df[main_df["workingday"] == "Yes"]
elif selected_day_type == "Hari Libur / Akhir Pekan":
    main_df = main_df[main_df["workingday"] == "No"]

if main_df.empty:
    st.warning("Tidak ada data yang cocok dengan kombinasi filter yang dipilih. Silakan sesuaikan kembali filter di sidebar.")
    st.stop()

# Menyiapkan dataframe agregasi
daily_rentals_df = create_daily_rentals_df(main_df)
weather_df = create_weather_impact_df(main_df)
season_df = create_season_impact_df(main_df)
hourly_df = create_hourly_df(main_df, selected_day_type)
monthly_rentals_df = create_monthly_rentals_df(main_df)
workingday_prop_df = create_workingday_prop_df(main_df)
clustered_df = create_time_cluster_df(main_df)

# Palet Warna Seragam
PRIMARY_BLUE = "#0284C7"
ACCENT_ORANGE = "#F59E0B"
SLATE_LINE = "#94A3B8"

# ==============================================================================
# HERO HEADER BANNER
# ==============================================================================
total_rentals = int(daily_rentals_df["total_count_day"].sum())
total_registered = int(daily_rentals_df["registered_users_day"].sum())
total_casual = int(daily_rentals_df["casual_users_day"].sum())
active_days = int((daily_rentals_df["total_count_day"] > 0).sum())
avg_daily = int(total_rentals / active_days) if active_days > 0 else 0

reg_share = (total_registered / total_rentals * 100) if total_rentals > 0 else 0
cas_share = (total_casual / total_rentals * 100) if total_rentals > 0 else 0

st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🚲 Bike Sharing Analytics Dashboard</div>
        <p class="hero-subtitle">Monitoring performa penyewaan sepeda, pola perilaku pengguna, dan klaster waktu operasional</p>
        <div>
            <span class="filter-badge">📅 {start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}</span>
            <span class="filter-badge">🏢 {selected_day_type}</span>
            <span class="filter-badge">📊 {active_days:,} Hari Operasional</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# BARIS 1: KARTU KPI UTAMA
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Penyewaan", f"{total_rentals:,}")
with col2:
    st.metric("Registered Users", f"{total_registered:,}", f"{reg_share:.1f}% dari total")
with col3:
    st.metric("Casual Users", f"{total_casual:,}", f"{cas_share:.1f}% dari total")
with col4:
    st.metric("Rata-rata / Hari", f"{avg_daily:,}")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# BARIS 2: TREN PENYEWAAN HARIAN (FULL CARD)
# ==============================================================================
with st.container(border=True):
    st.markdown('<div class="section-title">📈 Tren Penyewaan Harian</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.plot(
        daily_rentals_df["dateday"],
        daily_rentals_df["total_count_day"],
        linewidth=1.8,
        color=PRIMARY_BLUE
    )
    ax.fill_between(
        daily_rentals_df["dateday"],
        daily_rentals_df["total_count_day"],
        color=PRIMARY_BLUE,
        alpha=0.08
    )
    ax.set_ylabel("Jumlah Penyewaan", fontsize=11, color="#334155")
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# BARIS 3: BAGIAN 1 - PENGARUH KONDISI CUACA & MUSIM (FULL CARD)
# ==============================================================================
with st.container(border=True):
    st.markdown('<div class="section-title">1. Pengaruh Kondisi Cuaca & Musim terhadap Penyewaan Sepeda</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5.5))

    sns.barplot(
        x="weather_cond_day",
        y="total_count_day",
        data=weather_df,
        order=weather_df["weather_cond_day"],
        color=PRIMARY_BLUE,
        ax=ax[0]
    )
    ax[0].set_title("Rata-rata Penyewaan Berdasarkan Cuaca", fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    ax[0].set_ylabel("Rata-rata Penyewaan Harian", fontsize=11.5, color="#334155")
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis="both", labelsize=11)
    ax[0].grid(axis="y", linestyle="--", alpha=0.35)
    if ax[0].containers:
        ax[0].bar_label(ax[0].containers[0], fmt="%.0f", fontsize=11.5, fontweight="bold", padding=4)

    sns.barplot(
        x="season",
        y="total_count_day",
        data=season_df,
        order=season_df["season"],
        color=PRIMARY_BLUE,
        ax=ax[1]
    )
    ax[1].set_title("Rata-rata Penyewaan Berdasarkan Musim", fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis="both", labelsize=11)
    ax[1].grid(axis="y", linestyle="--", alpha=0.35)
    if ax[1].containers:
        ax[1].bar_label(ax[1].containers[0], fmt="%.0f", fontsize=11.5, fontweight="bold", padding=4)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# BARIS 4: BAGIAN 2 & BAGIAN 3 BERDAMPINGAN (2-COLUMN CARD GRID)
# ==============================================================================
grid_col1, grid_col2 = st.columns(2)

with grid_col1:
    with st.container(border=True):
        hourly_label = "Hari Libur / Akhir Pekan" if selected_day_type == "Hari Libur / Akhir Pekan" else "Hari Kerja (Workingday)"
        st.markdown(f'<div class="section-title">2. Pola Penyewaan Per Jam ({hourly_label})</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(
            hourly_df["hour"],
            hourly_df["registered_users_hour"],
            label="Registered Users",
            marker="o",
            markersize=5,
            linewidth=2.3,
            color=PRIMARY_BLUE
        )
        ax.plot(
            hourly_df["hour"],
            hourly_df["casual_users_hour"],
            label="Casual Users",
            marker="o",
            markersize=5,
            linewidth=2.3,
            color=ACCENT_ORANGE
        )
        ax.set_xlabel("Jam Operasional (00:00 - 23:00)", fontsize=11, color="#334155")
        ax.set_ylabel("Rata-rata Penyewaan", fontsize=11, color="#334155")
        ax.set_xticks(range(0, 24, 2))
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=10.5, frameon=True, facecolor="white", edgecolor="#E2E8F0")
        ax.grid(True, linestyle="--", alpha=0.35)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with grid_col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">3. Tren Pertumbuhan Bulanan (2011 vs 2012)</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        if "2011" in monthly_rentals_df.columns:
            ax.plot(
                monthly_rentals_df.index,
                monthly_rentals_df["2011"],
                label="Tahun 2011",
                marker="o",
                markersize=5,
                linewidth=2.2,
                color=SLATE_LINE
            )
        if "2012" in monthly_rentals_df.columns:
            ax.plot(
                monthly_rentals_df.index,
                monthly_rentals_df["2012"],
                label="Tahun 2012",
                marker="o",
                markersize=5,
                linewidth=2.6,
                color=PRIMARY_BLUE
            )

        ax.set_xlabel("Bulan", fontsize=11, color="#334155")
        ax.set_ylabel("Total Penyewaan Sepeda", fontsize=11, color="#334155")
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(title="Tahun", fontsize=10.5, frameon=True, facecolor="white", edgecolor="#E2E8F0")
        ax.grid(True, linestyle="--", alpha=0.35)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# BARIS 5: BAGIAN 4 - PROPORSI PENGGUNA (FULL CARD)
# ==============================================================================
with st.container(border=True):
    st.markdown('<div class="section-title">4. Proporsi Pengguna Casual vs Registered (Hari Libur vs Hari Kerja)</div>', unsafe_allow_html=True)

    labels = ["Casual Users", "Registered Users"]
    pie_colors = [ACCENT_ORANGE, PRIMARY_BLUE]
    col_left, col_right = st.columns(2)

    with col_left:
        if "No" in workingday_prop_df.index and workingday_prop_df.loc["No"].sum() > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.pie(
                workingday_prop_df.loc["No"],
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=pie_colors,
                explode=(0.05, 0),
                textprops={"fontsize": 11.5}
            )
            ax.set_title("Hari Libur / Akhir Pekan (workingday = No)", fontsize=13.5, fontweight="bold", pad=10, color="#0F172A")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Data hari libur tidak tersedia pada kombinasi filter saat ini.")

    with col_right:
        if "Yes" in workingday_prop_df.index and workingday_prop_df.loc["Yes"].sum() > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.pie(
                workingday_prop_df.loc["Yes"],
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=pie_colors,
                explode=(0.05, 0),
                textprops={"fontsize": 11.5}
            )
            ax.set_title("Hari Kerja (workingday = Yes)", fontsize=13.5, fontweight="bold", pad=10, color="#0F172A")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Data hari kerja tidak tersedia pada kombinasi filter saat ini.")

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# BARIS 6: BAGIAN 5 - CLUSTERING KATEGORI WAKTU (FULL CARD)
# ==============================================================================
with st.container(border=True):
    st.markdown('<div class="section-title">5. Distribusi Penyewaan Berdasarkan Klaster Waktu Operasional</div>', unsafe_allow_html=True)

    col_chart, col_summary = st.columns([2.2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        sns.barplot(
            x="time_category",
            y="total_rentals",
            data=clustered_df,
            order=clustered_df["time_category"],
            color=PRIMARY_BLUE,
            ax=ax
        )
        ax.set_xlabel("Kategori Waktu", fontsize=11.5, color="#334155")
        ax.set_ylabel("Total Penyewaan", fontsize=11.5, color="#334155")
        ax.ticklabel_format(style="plain", axis="y")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if ax.containers:
            ax.bar_label(ax.containers[0], fmt="%.0f", fontsize=11.5, fontweight="bold", padding=4)

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_summary:
        st.markdown("**Ringkasan Klaster Waktu**")
        table_df = clustered_df.copy()
        table_df.columns = ["Klaster Waktu", "Total Sewa", "Rata-rata/Jam"]
        table_df["Total Sewa"] = table_df["Total Sewa"].map("{:,.0f}".format)
        table_df["Rata-rata/Jam"] = table_df["Rata-rata/Jam"].map("{:,.0f}".format)
        st.dataframe(table_df, hide_index=True, use_container_width=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.caption("Copyright © Andini Febrina 2026")