# Digital Twin IoT Power Prediction

Proyek data mining untuk prediksi konsumsi daya listrik berbasis Digital Twin menggunakan Machine Learning.

## Deskripsi

Proyek ini mengimplementasikan Digital Twin untuk prediksi konsumsi daya listrik berdasarkan data sensor IoT menggunakan metode regresi.

## Dataset

- **Source:** IoT Gateway (Raspberry Pi)
- **Records:** 93,121 readings
- **Duration:** ~4.5 days (20-24 Mei 2026)
- **Features:** Suhu, Kelembaban, Tegangan, Arus
- **Target:** Prediksi Daya (Watt)

## Live Demo

Aplikasi Streamlit tersedia secara publik di:

**[Streamlit Cloud App](https://your-app-name.streamlit.app)**

## Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Streamlit Dashboard

```bash
streamlit run streamlit_web.py
```

Buka browser di `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. Push kode ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Klik **New app** dan pilih repository ini
5. Set main file path ke `streamlit_web.py`
6. Klik **Deploy!**

### Docker (Optional)

```bash
# Build image
docker build -t digital-twin .

# Run container
docker run -p 8501:8501 digital-twin
```

## Struktur File

```
digital-twin-iot-power-prediction/
├── streamlit_web.py         # Web dashboard (Streamlit)
├── main_analysis.py         # Pipeline analisis data mining
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── Dockerfile               # Docker configuration
└── README.md                # This file
```

## Model yang Digunakan

| Model | RMSE (W) | R² Score |
|-------|----------|----------|
| **Linear Regression** | 0.61 | 93.41% |
| Random Forest | 0.73 | 90.48% |
| XGBoost | 1.43 | 63.37% |

> **Best Model:** Linear Regression - karena korelasi Tegangan-Daya sangat tinggi (r = 0.934), sesuai dengan Hukum Ohm P = V × I