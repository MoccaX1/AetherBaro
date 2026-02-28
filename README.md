# AetherBaro: Advanced Atmospheric Layering & Spectral Analyzer

🌍 **Live Demo:** [https://aetherbaro.streamlit.app/](https://aetherbaro.streamlit.app/)

**AetherBaro** là một hệ thống phân tích áp suất khí quyển độ phân giải siêu cao (raw data lên tới 32Hz). Ứng dụng là một "kính hiển vi" khảo sát các sóng trọng lực khí quyển (Atmospheric Gravity Waves) và nhiễu động nhiệt, giúp bóc tách các hiện tượng vật lý thông qua 5 lớp phân tích độc lập. Hệ thống tự động tối ưu hóa bù trừ nhiễu màng nhĩ phần cứng (Empirical Noise: Turbulence, Electronic Noise, VLF Drift) dựa trên cảm biến thiết bị đo (vd: Bosch BMP380, InvenSense).

## 🚀 Tính năng nổi bật (Major Features)

### Lớp 1 (Synoptic & Fixed Bands Filter)
* **Bộ lọc tuyến tính (Linear Bandpass Filters):** Cô lập năng lượng vào các "rọ" lý thuyết: **S3, S4, Boss, Mother, Child, Micro**.
* **Trừ Dư số Synoptic (Residual / Wave-Only Fluctuation):** Bóc tách xu hướng áp suất chậm ra khỏi dao động sóng và nhiễu (áp suất dư 0 hPa).
* **Phantom Waves Overlay (Tính năng X-Ray):** Hiển thị những con "Sóng Bóng ma" (Sóng vật lý phát hiện bởi Trọng tài Layer 2) lồng ghép đè lên dữ liệu đo thực tế, bóc trần cấu trúc thực sự của rọ lọc băng thông.
* **Thời tiết Không gian:** Tích hợp phương trình thiên văn tính toán pha Mặt trăng, góc Cao độ Mặt trời và Thủy triều Khí quyển (Solar+Lunar Tides).

### Lớp 2 (Multi-Method Wave Spectrum Analysis)
Đây là cốt lõi của AetherBaro, khảo sát phổ tần số bằng 5 phương pháp Xử lý tín hiệu song song nhằm tránh thiên kiến toán học:
1. **FFT (Fast Fourier Transform):** Zero-padded FFT, độ phân giải cao nhất, nhạy bén tuyệt đối với sóng ngắn.
2. **PSD (Welch's Periodogram):** Loại bỏ nhiễu ngẫu nhiên bằng Gaussian Smoothing, bắt các đỉnh sóng bền vững.
3. **STFT (Spectrogram Dual-Window):** Khảo sát thời gian - tần số. Đánh giá độ dai dẳng của sóng qua bản đồ nhiệt.
4. **CWT (Continuous Wavelet Morlet):** Biến đổi Wavelet liên tục (Scalogram), khảo sát cực nhạy phân bố năng lượng theo thang đo Logarit của các sóng siêu dài (S3, Boss).
5. **HHT/EMD (Hilbert-Huang):** Phân tích phi tuyến tính, đi tìm đường bao cực trị để bóc tách các Tần số Nội tại gốc (Intrinsic Mode Functions).

### Lớp 3 (Trọng tài Consensus Thông minh)
* **Smart Evidence-Based Scoring System (0-100/100):** Vượt qua hạn chế của việc đếm "số Vote" thông thường. Hệ thống chấm điểm dựa trên:
  * **Chuyên môn thuật toán:** (VD: HHT/CWT uy tín đặc biệt cho sóng dài, FFT uy tín cho sóng ngắn).
  * **Tỷ lệ Tín hiệu/Nhiễu (SNR) Thích ứng Phần cứng:** Sử dụng nền tảng nhiễu động học. Ngưỡng nhiễu Turbulence `(< 1m) là 0.0072 hPa`, nhiễu trôi nhiệt tĩnh VLF `(> 160m) là 0.1656 hPa`.
  * **Sàng lọc tự động:** Chấm điểm `Confirmed 🟢`, `Likely 🟡`, `Weak 🟠`, `Uncertain ⚪`. Những sóng bị điểm liệt (ảo ảnh toán học, rò rỉ phổ) sẽ tự động bị loại.

### Lớp 4 & 5 (Atmosphere State & Micro-events)
* **Permutation Entropy:** Đo lường độ hỗn loạn khí quyển, Kolmogorov Global Spectral Slope. Nhận diện sự bất ổn định trước dông lốc.

## 🛠 Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.10+
* **Hệ sinh thái:** Streamlit (Dark Mode Optimized), Plotly (Interactive Heatmaps)
* **Toán học & Tín hiệu:** SciPy, NumPy, PyWavelets (CWT), EMD-signal (HHT)

## 📁 Cấu trúc dữ liệu yêu cầu

Dữ liệu đầu vào cần được đặt trong thư mục `data/` với cấu trúc:
```text
data/
└── Pressure_YYYYMMDD_HHMM/
    ├── Pressure.csv (Dữ liệu 32Hz ngõ vào)
    └── meta/
        ├── device.csv (Thông số phần cứng, Hardware noise tolerance)
        ├── time.csv 
        └── location.csv (Tọa độ GPS)
```

## 💻 Hướng dẫn cài đặt

```bash
# Cài đặt thư viện:
pip install -r requirements.txt

# Khởi chạy:
streamlit run app.py
```

---
