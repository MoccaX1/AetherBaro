# AetherBaro: High-Resolution Atmospheric Layering & Spectral Analyzer

**AetherBaro** là một hệ thống phân tích áp suất khí quyển độ phân giải cao, được thiết kế để bóc tách các hiện tượng vật lý từ dữ liệu thô thông qua cấu trúc phân tích 5 lớp độc lập. Ứng dụng tự động tối ưu hóa dựa trên thông số phần cứng của thiết bị đo (như LG V60, Sony Xperia...) để mang lại độ chính xác cao nhất.

## 🚀 Tính năng chính (5 Lớp Vật lý)

1.  **Lớp 1 (Synoptic & Tides):** Phân tích xu hướng quy mô lớn và thủy triều khí quyển (Mặt Trăng & Mặt Trời).
2.  **Lớp 2 (Wave Spectrum):** Nhận diện động các dải sóng **Boss**, **Mother**, **Child** và **Micro** thông qua Zero-padded FFT.
3.  **Lớp 3 (Atmosphere State):** Đo lường độ hỗn loạn khí quyển bằng **Permutation Entropy** và Rolling Variance.
4.  **Lớp 4 (Micro-events):** Phát hiện các xung động áp suất cực ngắn (Gust Proxy) từ dữ liệu gốc 32Hz.
5.  **Lớp 5 (Planetary Link):** Đối chiếu dữ liệu thực tế với các mỏ neo bên ngoài (External Anchors) và so sánh Baseline.

## 🛠 Công nghệ sử dụng

*   **Ngôn ngữ:** Python 3.10+
*   **Giao diện:** Streamlit (Dark Mode Optimized)
*   **Đồ họa:** Plotly (Interactive & Dynamic Decimation)
*   **Xử lý tín hiệu:** NumPy, SciPy (Butterworth SOS Filters, Gaussian Order-1 Derivatives)
*   **Thiên văn:** Astral (Solar Elevation & Moon Phase calculations)

## 📁 Cấu trúc dữ liệu yêu cầu

Dữ liệu đầu vào cần được đặt trong thư mục `data/` với cấu trúc:
```text
data/
└── Pressure_YYYYMMDD_HHMM/
    ├── Pressure.csv (Dữ liệu 32Hz thô)
    └── meta/
        ├── device.csv (Thông số cảm biến từ NSX)
        ├── time.csv (Thời gian bắt đầu/kết thúc)
        └── location.csv (Tọa độ GPS để tính thủy triều)
```

## 💻 Hướng dẫn cài đặt

1.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chạy ứng dụng:**
    ```bash
    run.bat
    # Hoặc chạy lệnh trực tiếp:
    streamlit run app.py
    ```

## 🔋 Khả năng tương thích thiết bị

Hệ thống tự động đọc file `device.csv` để:
*   **Điều chỉnh sai số (Tolerance):** Tự động nhận diện độ phân giải cảm biến (ví dụ: 0.01 hPa cho LG V60).
*   **Giới hạn tần số (Nyquist):** Tự động giới hạn tần số phân tích tối đa dựa trên `MinDelay` của phần cứng.

---
*Phát triển bởi Antigravity AI Code Team.*
