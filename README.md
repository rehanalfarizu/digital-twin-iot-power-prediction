# Digital Twin IoT Power Prediction

Proyek data mining untuk prediksi konsumsi daya listrik berbasis Digital Twin menggunakan Machine Learning.

## Deskripsi

Proyek ini mengimplementasikan Digital Twin untuk prediksi konsumsi daya listrik berdasarkan data sensor IoT menggunakan metode XGBoost dan LSTM.

## Dataset

- **Source:** IoT Gateway (Raspberry Pi)
- **Records:** 93,121 readings
- **Duration:** ~4.5 days
- **Features:** Suhu, Kelembaban, Tegangan, Arus, Daya
- **Target:** Prediksi Daya (Watt)

## Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Analysis Pipeline

```bash
python main_analysis.py
```

### 3. Jalankan Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Buka browser di `http://localhost:8501`

## Docker Deployment (Optional)

```bash
# Build image
docker build -t digital-twin .

# Run container
docker run -p 8501:8501 digital-twin
```

## Struktur File

```
digital-twin-iot-power-prediction/
├── main_analysis.py         # Pipeline analisis data mining
├── streamlit_app.py         # Web dashboard deployment
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── CLAUDE.md              # Project documentation
└── README.md              # This file
```

## Model yang Digunakan

| Model | Kelebihan |
|-------|-----------|
| **XGBoost** | High accuracy, fast |
| **Random Forest** | Robust, interpretable |
| **LSTM** | Time-series aware |
| **Linear Regression** | Simple baseline |