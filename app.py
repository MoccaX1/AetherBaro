import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
from analysis import (
    load_and_preprocess_data, 
    analyze_layer_1, 
    analyze_layer_2,
    detect_waves_fft,
    detect_waves_psd,
    detect_waves_stft,
    detect_waves_cwt,
    detect_waves_hht,
    compute_wave_consensus,
    analyze_layer_3,
    analyze_layer_4,
    analyze_layer_5,
    analyze_device_performance,
    export_features
)

st.set_page_config(page_title="High-Res Pressure Analyzer", layout="wide")

st.title("Phân Tích Dữ Liệu Áp Suất Thời Gian Thực (5 Lớp Vật Lý)")

# Ensure DATA_DIR works on both Windows Local and Linux Cloud (Streamlit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(DATA_DIR):
    st.error(f"Kh\u00f4ng t\u00ecm th\u1ea5y th\u01b0 m\u1ee5c d\u1eef li\u1ec7u t\u1ea1i: {DATA_DIR}. Vui l\u00f2ng ki\u1ec3m tra l\u1ea1i th\u01b0 m\u1ee5c data tr\u00ean Github.")

@st.cache_data
def get_processed_data(folder_path, target_fs=1.0):
    st.write(f"Đang xử lý dữ liệu từ: {folder_path} (Tần số: {target_fs}Hz)...")
    df_32hz, df_base = load_and_preprocess_data(folder_path, target_fs=target_fs)
    return df_32hz, df_base

def load_device_info(folder_path):
    device_path = os.path.join(folder_path, "meta", "device.csv")
    info = {'Resolution': 0.01, 'Model': 'Unknown', 'Sensor': 'Unknown'}
    if os.path.exists(device_path):
        try:
            df = pd.read_csv(device_path)
            res_rows = df[df['property'] == 'pressure Resolution']
            if not res_rows.empty:
                info['Resolution'] = float(res_rows['value'].values[0])
            
            model_rows = df[df['property'] == 'deviceModel']
            if not model_rows.empty:
                info['Model'] = model_rows['value'].values[0]
                
            delay_rows = df[df['property'] == 'pressure MinDelay']
            if not delay_rows.empty:
                min_delay_us = float(delay_rows['value'].values[0])
                if min_delay_us > 0:
                    info['MaxFS'] = round(1000000.0 / min_delay_us)
                
            vendor_rows = df[df['property'] == 'pressure Vendor']
            name_rows = df[df['property'] == 'pressure Name']
            vendor = vendor_rows['value'].values[0] if not vendor_rows.empty else ""
            name = name_rows['value'].values[0] if not name_rows.empty else ""
            if vendor or name:
                info['Sensor'] = f"{vendor} {name}".strip()
        except Exception:
            pass
    return info

def load_location_info(folder_path):
    loc_path = os.path.join(folder_path, "meta", "location.csv")
    info = {
        'City': 'Ho Chi Minh City',
        'Region': 'Vietnam',
        'Country': 'Vietnam',
        'Timezone': 'Asia/Ho_Chi_Minh',
        'Latitude': 10.7626,
        'Longitude': 106.6601
    }
    if os.path.exists(loc_path):
        try:
            df = pd.read_csv(loc_path)
            for prop, val in zip(df['property'], df['value']):
                p = prop.lower().strip()
                if 'lat' in p: info['Latitude'] = float(val)
                elif 'lon' in p: info['Longitude'] = float(val)
                elif 'timezone' in p: info['Timezone'] = str(val)
                elif 'city' in p: info['City'] = str(val)
                elif 'region' in p: info['Region'] = str(val)
                elif 'country' in p: info['Country'] = str(val)
        except Exception:
            pass
    return info

def main():
    st.sidebar.header("Chọn Dữ Liệu")
    
    # Lấy danh sách thư mục (chỉ chứa data)
    try:
        folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f.startswith("Pressure_")]
    except FileNotFoundError:
        st.error(f"Không tìm thấy thư mục: {DATA_DIR}")
        return
        
    selected_folder = st.sidebar.selectbox("Thư mục dữ liệu:", folders)
    
    if selected_folder:
        folder_path = os.path.join(DATA_DIR, selected_folder)
        
        # Load Device & Location Info
        device_info = load_device_info(folder_path)
        location_info = load_location_info(folder_path)
        
        tolerance = device_info['Resolution']
        max_fs = device_info.get('MaxFS', 32)
        
        # --- Sampling Rate Option ---
        st.sidebar.markdown("---")
        
        # Dynamically build options
        base_options = ["1Hz (Mặc định - Nhanh)", "5Hz (Chi tiết)"]
        if max_fs > 5:
            max_option = f"{max_fs}Hz (Bản gốc - Nặng)"
            options = base_options + [max_option]
        else:
            options = base_options
            
        resample_option = st.sidebar.selectbox(
            "Tần số phân tích (Performance/Detail):", 
            options
        )
        
        if "1Hz" in resample_option:
            fs = 1.0
        elif "5Hz" in resample_option:
            fs = 5.0
        else:
            fs = float(max_fs)
            
        try:
            df_32hz, df_base = get_processed_data(folder_path, fs)
        except Exception as e:
            st.error(f"Lỗi khi load dữ liệu: {e}")
            return
            
        # Optional: Baseline Comparison
        baseline_options = ["None"] + [f for f in folders if f != selected_folder]
        st.sidebar.markdown("---")
        baseline_folder = st.sidebar.selectbox("Baseline Folder (Layer 5)", baseline_options)
        
        # External Truth (MSLP)
        external_mslp = st.sidebar.number_input("External MSLP (Trạm VVTS) - hPa", value=1010.0, step=0.1)
        
        st.sidebar.success(f"✅ Dữ liệu đã được tiền xử lý ({int(fs)}Hz)")
        
        with st.spinner("Đang chẩn đoán độ phân giải phổ năng lượng (Welch PSD)..."):
            metrics_device = analyze_device_performance(df_32hz, device_info)
            tol = device_info.get('Resolution', 0.01)
            emp_white_noise = metrics_device['Empirical White Noise Std (hPa)']
            emp_turb = metrics_device.get('Empirical Turbulence RMS (hPa)', 0)
            emp_waves = metrics_device.get('Empirical Waves RMS (hPa)', 0)
            emp_pink_noise = metrics_device['Empirical Pink Noise RMS (hPa)'] # This is now pure VLF drift
            emp_res = metrics_device['Empirical Resolution (hPa)']
            
            # The hard noise floor for detecting waves is now the structural VLF drift + white noise
            noise_limit = max(tol, emp_white_noise + emp_pink_noise, emp_res)
            
        st.write(f"### Tổng quan dữ liệu Gốc (Đã Resample {int(fs)}Hz cho hiệu năng)")
        
        # Calculate start/end and dual date for overview header
        t_start = df_base['Datetime'].iloc[0]
        t_end = df_base['Datetime'].iloc[-1]
        try:
            from lunardate import LunarDate
            lunar = LunarDate.fromSolarDate(t_start.year, t_start.month, t_start.day)
            overview_date_str = f"Ngày Dương: {t_start.strftime('%d/%m/%Y')} | Ngày Âm: {lunar.day:02d}/{lunar.month:02d}"
        except Exception:
            overview_date_str = f"Ngày: {t_start.strftime('%d/%m/%Y')}"
            
        duration = t_end - t_start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        st.caption(f"**Thời gian đo:** {overview_date_str} (Từ {t_start.strftime('%H:%M:%S')} đến {t_end.strftime('%H:%M:%S')}). **Tổng thời gian:** {duration_str}")
        st.caption(rf"**Thiết bị đo:** {device_info['Model']} | **Cảm biến Áp suất:** {device_info['Sensor']} | **Sai số phần cứng:** $\pm{tol}$ hPa")
        st.caption(rf"**Nhiễu điện tử (>1Hz):** $\pm{emp_white_noise:.4f}$ hPa | **Turbulence gió (1s-1m):** $\pm{emp_turb:.4f}$ hPa | **Nhiễu Trôi VLF (>160m):** $\pm{emp_pink_noise:.4f}$ hPa")
        st.caption(f"**Vị trí đo:** {location_info['City']}, {location_info['Region']}, {location_info['Country']} ({location_info['Latitude']}, {location_info['Longitude']}) | **Múi giờ:** {location_info['Timezone']}")
        
        # Plot downsampled if it's 32Hz to avoid massive browser lag
        plot_df = df_base.iloc[::int(max(1, fs))] if fs == 32.0 else df_base
        
        fig = px.line(plot_df, x='Datetime', y='Pressure (hPa)', title=f"Áp suất - {selected_folder} ({int(fs)}Hz Data)",
                     template="plotly_dark", render_mode="svg")
        fig.update_xaxes(title=None)
                     
        # Extract Min/Max with Dynamic Sensor Error Margin
        p_max_val = plot_df['Pressure (hPa)'].max()
        p_min_val = plot_df['Pressure (hPa)'].min()
        
        y_max_ov = plot_df['Pressure (hPa)'].where((p_max_val - plot_df['Pressure (hPa)']) <= tolerance, np.nan)
        y_min_ov = plot_df['Pressure (hPa)'].where((plot_df['Pressure (hPa)'] - p_min_val) <= tolerance, np.nan)
        
        t_max_series = plot_df.loc[~y_max_ov.isna(), 'Datetime']
        t_min_series = plot_df.loc[~y_min_ov.isna(), 'Datetime']
        
        # Plot as thick lines with tiny markers so opacity doesn't stack and ruin visibility in SVG
        fig.add_scatter(x=plot_df['Datetime'], y=y_max_ov, mode='lines+markers', line=dict(color='#ff4b4b', width=12), marker=dict(size=2), opacity=0.4, showlegend=False, name="Pmax Area")
        fig.add_scatter(x=plot_df['Datetime'], y=y_min_ov, mode='lines+markers', line=dict(color='#00d4ff', width=12), marker=dict(size=2), opacity=0.4, showlegend=False, name="Pmin Area")
        
        # Add range annotations for distinct peaks to highlight the tolerance zone
        def annotate_ranges_overview(t_series, p_val, color, prefix, y_pos):
            if t_series.empty: return
            blocks = []
            current_block = [t_series.iloc[0]]
            for t in t_series.iloc[1:]:
                # If the gap between two consecutive points in the tolerance zone is > 5 minutes, 
                # it means the pressure curve left the zone and came back. Break the block here.
                if (t - current_block[-1]).total_seconds() > 300: 
                    blocks.append((current_block[0], current_block[-1]))
                    current_block = [t]
                else:
                    current_block.append(t)
            blocks.append((current_block[0], current_block[-1]))
            
            for t_start, t_end in blocks:
                t_mid = t_start + (t_end - t_start) / 2
                if (t_end - t_start).total_seconds() < 300: # Very short
                    fig.add_vline(x=t_mid, line_width=1, line_dash="dot", line_color=color)
                    fig.add_annotation(x=t_mid, y=p_val, text=f"{prefix}: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                else:
                    fig.add_vrect(x0=t_start, x1=t_end, fillcolor=color, opacity=0.15, layer="below", line_width=0)
                    fig.add_vline(x=t_start, line_width=1, line_dash="dash", line_color=color)
                    fig.add_vline(x=t_end, line_width=1, line_dash="dash", line_color=color)
                    fig.add_annotation(x=t_mid, y=p_val, text=f"{prefix} Zone: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                    fig.add_annotation(x=t_start, y=0.0, yref="paper", yanchor="bottom", text=t_start.strftime('%H:%M'), showarrow=False, font=dict(color=color), xanchor="right")
                    fig.add_annotation(x=t_end, y=0.0, yref="paper", yanchor="bottom", text=t_end.strftime('%H:%M'), showarrow=False, font=dict(color=color), xanchor="left")

        annotate_ranges_overview(t_max_series, p_max_val, '#ff4b4b', 'Pmax', 'top')
        annotate_ranges_overview(t_min_series, p_min_val, '#00d4ff', 'Pmin', 'top')
        
        st.plotly_chart(fig, width="stretch")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Layer 1 (Synoptic)", 
            "Layer 2 (Waves)", 
            "Layer 3 (Atmosphere State)", 
            "Layer 4 (Micro)", 
            "Layer 5 (Planetary)",
            "Đánh giá Thiết bị"
        ])
        
        with tab1:
            st.header("1. Động lực học Quy mô Lớn", help="Nghiên cứu các biên độ áp suất khổng lồ, thay đổi chậm theo giờ/ngày do Bức xạ Mặt Trời (Thermal Tides), Trọng lực (Gravitational Tides), và các đợt Front lạnh/Áp thấp rộng hàng trăm km (Synoptic Scale).")
            df_l1, metrics_l1 = analyze_layer_1(df_base, fs=fs, location_data=location_info)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Synoptic Trend", metrics_l1['Synoptic Trend'], help="Xu hướng tổng thể của quy mô lớn. Rising = Áp suất đang nhích dần lên (thường báo hiệu trời quang đãng, lạnh). Falling = Áp suất sụt giảm (chuẩn bị có mưa, bão hoặc không khí nóng nóng chảy lên).")
            c2.metric("Max dP/dt", f"{metrics_l1['Max dP/dt']:.4f} hPa/hr", help="Tốc độ Tăng áp suất nhanh nhất (hPa/giờ). Thường xảy ra khi Front không khí lạnh đè ập xuống hoặc đang leo lên sườn đỉnh Thủy triều nhiệt.")
            c3.metric("Min dP/dt", f"{metrics_l1['Min dP/dt']:.4f} hPa/hr", help="Tốc độ Giảm áp suất nhanh nhất (âm). Dấu hiệu đặc trưng khi rãnh áp thấp, bão đang tiến lại gần, vắt kiệt và hut không khí lên cao.")
            
            # Dual Calendar
            try:
                from lunardate import LunarDate
                s_date = df_base['Datetime'].iloc[0]
                lunar = LunarDate.fromSolarDate(s_date.year, s_date.month, s_date.day)
                date_str = f"{s_date.strftime('%d/%m')} | {lunar.day:02d}/{lunar.month:02d}"
            except Exception as e:
                date_str = df_base['Datetime'].iloc[0].strftime('%d/%m')
                
            c4.metric("Âm Dương Lịch", f"{date_str}", help="Ngày bắt đầu file đo đạc được quy chiếu ra Âm Lịch Việt Nam để dùng chung với pha Mặt trăng.")
            
            phase_val = metrics_l1.get('Avg Moon Phase', 0)
            phase_name = metrics_l1.get('Lunar Phase Name', 'Không rõ')
            
            # Map phase to emoji and illumination %
            # 0=New, 7.38=First Quarter, 14.76=Full, 22.14=Last Quarter, 29.53=New
            illumination = 50.0 * (1.0 - np.cos(2 * np.pi * phase_val / 29.53))
            
            emojis = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘", "🌑"]
            idx = int(round((phase_val / 29.53) * 8)) % 8
            moon_emoji = emojis[idx]
            
            # Shorten name if it contains parens to avoid UI clipping
            short_name = phase_name.split(' (')[0] if '(' in phase_name else phase_name
            
            c5.metric(f"Mặt Trăng {moon_emoji}", f"{short_name} ({illumination:.0f}%)", help="Thông số Trăng tính theo phương trình góc nhìn thiên văn. % là tỷ lệ bề mặt nhận được ánh sáng từ góc nhìn ngắm trên Trái Đất.")
            # --- Performance Boost for Plotly Rendering ---
            # Max 1Hz for visualization to prevent browser freezing on dense 32Hz data
            plot_step = int(max(1, fs))
            df_l1_plot = df_l1.iloc[::plot_step] if fs > 1.0 else df_l1
            
            fig1 = px.line(df_l1_plot, x='Datetime', y=['Pressure (hPa)', 'Smoothed (1h)', 'Theoretical Tide (Solar+Lunar)', 'Residual Pressure (Synoptic Only)'], 
                           title="Synoptic Trend & Atmospheric Tides", template="plotly_dark", render_mode="svg")
            
            # Make theoretical tide dashed for clarity
            fig1.update_traces(line=dict(dash='dash'), selector=dict(name='Theoretical Tide (Solar+Lunar)'))
            fig1.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig1.update_xaxes(title=None)
            
            # Add annotations and projections for multiple extremum points with Dynamic Tolerance
            # 1. Base Pressure Max/Min
            p_max_l1 = df_l1['Pressure (hPa)'].max()
            p_min_l1 = df_l1['Pressure (hPa)'].min()
            
            y_max_l1 = df_l1_plot['Pressure (hPa)'].where((p_max_l1 - df_l1_plot['Pressure (hPa)']) <= tolerance, np.nan)
            y_min_l1 = df_l1_plot['Pressure (hPa)'].where((df_l1_plot['Pressure (hPa)'] - p_min_l1) <= tolerance, np.nan)
            
            t_max_l1_series = df_l1_plot.loc[~y_max_l1.isna(), 'Datetime']
            t_min_l1_series = df_l1_plot.loc[~y_min_l1.isna(), 'Datetime']
            
            # 2. Theoretical Tide Max/Min
            p_max_tide = df_l1['Theoretical Tide (Solar+Lunar)'].max()
            p_min_tide = df_l1['Theoretical Tide (Solar+Lunar)'].min()
            
            y_max_tide = df_l1_plot['Theoretical Tide (Solar+Lunar)'].where((p_max_tide - df_l1_plot['Theoretical Tide (Solar+Lunar)']) <= tolerance, np.nan)
            y_min_tide = df_l1_plot['Theoretical Tide (Solar+Lunar)'].where((df_l1_plot['Theoretical Tide (Solar+Lunar)'] - p_min_tide) <= tolerance, np.nan)
            
            t_max_tide_series = df_l1_plot.loc[~y_max_tide.isna(), 'Datetime']
            t_min_tide_series = df_l1_plot.loc[~y_min_tide.isna(), 'Datetime']
            
            # Plot as thick lines with tiny markers so opacity doesn't stack in SVG
            fig1.add_scatter(x=df_l1_plot['Datetime'], y=y_max_l1, mode='lines+markers', line=dict(color='#ff4b4b', width=12), marker=dict(size=2), opacity=0.4, showlegend=False)
            fig1.add_scatter(x=df_l1_plot['Datetime'], y=y_min_l1, mode='lines+markers', line=dict(color='#00d4ff', width=12), marker=dict(size=2), opacity=0.4, showlegend=False)
            fig1.add_scatter(x=df_l1_plot['Datetime'], y=y_max_tide, mode='lines+markers', line=dict(color='#ffaa00', width=12), marker=dict(size=2), opacity=0.4, showlegend=False)
            fig1.add_scatter(x=df_l1_plot['Datetime'], y=y_min_tide, mode='lines+markers', line=dict(color='#ffaa00', width=12), marker=dict(size=2), opacity=0.4, showlegend=False)
            
            # Range annotations for smooth data (Theoretical Tides)
            def get_tide_blocks_l1(t_series):
                if t_series.empty: return []
                blocks = []
                current_block = [t_series.iloc[0]]
                for t in t_series.iloc[1:]:
                    if (t - current_block[-1]).total_seconds() > 900: 
                        blocks.append((current_block[0], current_block[-1]))
                        current_block = [t]
                    else:
                        current_block.append(t)
                blocks.append((current_block[0], current_block[-1]))
                return blocks
                
            def draw_tide_blocks(fig, blocks, p_val, color, prefix, y_pos):
                for t_start, t_end in blocks:
                    t_mid = t_start + (t_end - t_start) / 2
                    if (t_end - t_start).total_seconds() < 300: # Very short
                        fig.add_vline(x=t_mid, line_width=1, line_dash="dot", line_color=color)
                        if p_val is not None:
                            fig.add_annotation(x=t_mid, y=p_val, text=f"{prefix}: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                    else:
                        fig.add_vrect(x0=t_start, x1=t_end, fillcolor=color, opacity=0.15, layer="below", line_width=0)
                        fig.add_vline(x=t_start, line_width=1, line_dash="dash", line_color=color)
                        fig.add_vline(x=t_end, line_width=1, line_dash="dash", line_color=color)
                        if p_val is not None:
                            fig.add_annotation(x=t_mid, y=p_val, text=f"{prefix} Zone: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                        fig.add_annotation(x=t_start, y=0.0, yref="paper", yanchor="bottom", text=t_start.strftime('%H:%M'), showarrow=False, font=dict(color=color), xanchor="right")
                        fig.add_annotation(x=t_end, y=0.0, yref="paper", yanchor="bottom", text=t_end.strftime('%H:%M'), showarrow=False, font=dict(color=color), xanchor="left")

            l1_max_blocks = get_tide_blocks_l1(t_max_l1_series)
            l1_min_blocks = get_tide_blocks_l1(t_min_l1_series)
            tide_max_blocks = get_tide_blocks_l1(t_max_tide_series)
            tide_min_blocks = get_tide_blocks_l1(t_min_tide_series)
            
            # Draw on Layer 1
            draw_tide_blocks(fig1, l1_max_blocks, p_max_l1, '#ff4b4b', 'Pmax', 'top')
            draw_tide_blocks(fig1, l1_min_blocks, p_min_l1, '#00d4ff', 'Pmin', 'top')
            draw_tide_blocks(fig1, tide_max_blocks, p_max_tide, '#ffaa00', 'Tide Max', 'top')
            draw_tide_blocks(fig1, tide_min_blocks, p_min_tide, '#ffaa00', 'Tide Min', 'top')
                
            fig1.update_xaxes(title=None)
            st.plotly_chart(fig1, width="stretch")
            
            # --- Residual Fluctuation centered at 0 ---
            df_l1_plot['Residual Fluctuation (+/- hPa)'] = df_l1_plot['Residual Pressure (Synoptic Only)'] - df_l1_plot['Residual Pressure (Synoptic Only)'].mean()
            
            y_res = df_l1_plot['Residual Fluctuation (+/- hPa)']
            import plotly.graph_objects as go
            fig1_res = go.Figure()
            
            # Create masked arrays so filling doesn't cross the y=0 boundary incorrectly
            pos_y = np.where(y_res >= 0, y_res, 0)
            neg_y = np.where(y_res < 0, y_res, 0)
            
            fig1_res.add_trace(go.Scatter(x=df_l1_plot['Datetime'], y=pos_y, mode='lines', 
                                          line=dict(color='#00ff00', width=1), fill='tozeroy', name='Bù (+)'))
            fig1_res.add_trace(go.Scatter(x=df_l1_plot['Datetime'], y=neg_y, mode='lines', 
                                          line=dict(color='#ff4b4b', width=1), fill='tozeroy', name='Trừ (-)'))
                                          
            fig1_res.update_layout(title="Độ bù trừ Áp suất Dư số (Residual Fluctuation ± hPa)", 
                                   template="plotly_dark", showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            
            fig1_res.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Baseline (0 hPa)")
            fig1_res.update_xaxes(title=None)
            st.plotly_chart(fig1_res, width="stretch")
            
            fig2 = px.line(df_l1_plot, x='Datetime', y='dP/dt (hPa/hr)', title="Tốc độ biến thiên (dP/dt) (1 Phút Smoothed)", template="plotly_dark", render_mode="svg")
            fig2.update_traces(line_color='#00d4ff')
            fig2.update_xaxes(title=None)
            st.plotly_chart(fig2, width="stretch")
            
            fig2_raw = px.line(df_l1_plot, x='Datetime', y='Raw dP/dt (hPa/hr)', title="Tốc độ biến thiên (Raw dP/dt)", template="plotly_dark", render_mode="svg")
            fig2_raw.update_traces(line_color='#ff4b4b', opacity=0.7)
            fig2_raw.update_xaxes(title=None)
            st.plotly_chart(fig2_raw, width="stretch")
            
            # --- Astronomical Features Chart ---
            fig_astro = px.line(df_l1_plot, x='Datetime', y=['Solar Elevation (deg)', 'Moon Elevation (deg)'],
                               title="Thông số Thiên văn Cốt lõi", template="plotly_dark", render_mode="svg")
            
            fig_astro.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            
            # Khôi phục màu vàng cho mặt trời và xanh cho mặt trăng
            fig_astro.update_traces(selector=dict(name='Solar Elevation (deg)'), line_color='#ffaa00')
            fig_astro.update_traces(selector=dict(name='Moon Elevation (deg)'), line_color='#00d4ff')
            
            fig_astro.update_yaxes(title_text="<b>Elevation</b> (Degrees)")
            fig_astro.update_xaxes(title=None)
            
            st.plotly_chart(fig_astro, width="stretch")
            
        with tab2:
            st.header("2. Pho Song Da Phuong Phap (Multi-Method Wave Analysis)", help="5 phuong phap doc lap phat hien song khi quyen. Tab Tong hop bieu quyet da so.")
            filtered_signals, freqs, power, periods_min, power_valid, exact_peak_period, dynamic_bands = analyze_layer_2(df_base, fs=fs)
            df_waves = df_base[['Datetime']].copy()
            for name, sig in filtered_signals.items():
                df_waves[name] = sig
            df_waves_plot = df_waves.iloc[::plot_step] if fs > 1.0 else df_waves
            total_duration_mins = duration.total_seconds() / 60
            dynamic_noise_floor = emp_turb + emp_white_noise
            max_band_period = max((info['period_range'][1] for info in dynamic_bands.values()), default=300)
            fft_max_period = max(300, max_band_period * 1.15)
            color_map = {'S3': 'cyan', 'S4': 'magenta', 'Boss': 'red', 'Mother': 'green', 'Child': 'blue', 'Micro': 'orange', 'Wildcard': 'purple'}
            
            with st.spinner("Dang chay 5 phuong phap phan tich pho..."):
                result_fft = detect_waves_fft(df_base, fs=fs)
                result_psd = detect_waves_psd(df_base, fs=fs)
                result_stft = detect_waves_stft(df_base, fs=fs)
                result_cwt = detect_waves_cwt(df_base, fs=fs)
                result_hht = detect_waves_hht(df_base, fs=fs)
                all_results = [result_fft, result_psd, result_stft, result_cwt, result_hht]
                consensus = compute_wave_consensus(all_results)
            
            stab_con, stab_fft, stab_psd, stab_stft, stab_cwt, stab_hht = st.tabs(["Tong hop", "FFT", "PSD Welch", "STFT", "CWT", "HHT/EMD"])
            
            def draw_spectrum(result, ct):
                if 'periods' not in result: return
                df_s = pd.DataFrame({'Period (min)': result['periods'], 'Power': result['power']})
                df_s = df_s[(df_s['Period (min)'] >= 10) & (df_s['Period (min)'] <= fft_max_period)]
                fig = px.line(df_s, x='Period (min)', y='Power', log_y=True, title=f"Pho nang luong: {result['method']}", template="plotly_dark", render_mode="svg")
                for info in dynamic_bands.values():
                    lp, hp = info['period_range']; bn = info['base_name']
                    fig.add_vrect(x0=lp, x1=hp, fillcolor=color_map.get(bn,'gray'), opacity=0.10, line_width=0)
                for i, w in enumerate(result['waves'][:6]):
                    p = w['period_min']
                    if 10 <= p <= fft_max_period:
                        fig.add_vline(x=p, line_dash="dot", line_color="white", opacity=0.7)
                        fig.add_annotation(x=p, y=1.0-i*0.08, yref="paper", text=f"{p:.1f}m", showarrow=True, arrowhead=2, font=dict(color="white", size=10))
                fig.update_xaxes(title=None); fig.update_yaxes(title=result.get('ylabel','Power'))
                ct.plotly_chart(fig, width="stretch")
                
                # Bảng thống kê tần số phát hiện
                if result['waves']:
                    ct.markdown(f"**Danh sach Dinh song ({result['method']}):**")
                    ct.dataframe(pd.DataFrame([{"Xep hang": f"#{i+1}", "Chu ky (min)": f"{w['period_min']:.1f}", "Bien do (hPa)": f"{w.get('amplitude',0):.5f}", "Nang luong": f"{w.get('power',0):.5f}"} for i, w in enumerate(result['waves'][:6])]), width="stretch")
            
            def draw_heatmap(result, ct, title):
                import plotly.graph_objects as go
                sd = result.get('spectrogram') or result.get('scalogram')
                if sd is None: return
                pm = sd['periods_min']; pmask = (pm >= 10) & (pm <= fft_max_period)
                if not np.any(pmask): return
                pm_filtered = pm[pmask]
                z_data = sd['power_db'][pmask, :]
                fig = go.Figure(data=go.Heatmap(z=z_data, x=sd['time_hours'], y=pm_filtered, colorscale='Magma', colorbar=dict(title='dB')))
                for w in result['waves'][:5]:
                    p = w['period_min']
                    if 10 <= p <= fft_max_period:
                        fig.add_hline(y=p, line_dash='dash', line_color='white', opacity=0.5, annotation_text=f"{p:.0f}m", annotation_font_color='white')
                
                # Adaptive Y-axis: bound by actual data range (with small padding in log space)
                # Use where the data actually has energy (above 10% percentile of dB)
                energy_thresh = np.percentile(z_data, 15)
                hot_rows = np.any(z_data >= energy_thresh, axis=1)
                if np.any(hot_rows):
                    y_lo = max(pm_filtered[hot_rows].min() * 0.7, 10)
                    y_hi = min(pm_filtered[hot_rows].max() * 1.4, fft_max_period)
                else:
                    y_lo = pm_filtered.min()
                    y_hi = pm_filtered.max()
                
                import math
                fig.update_layout(
                    title=title, xaxis_title="Thoi gian (gio)", yaxis_title="Chu ky (phut)",
                    yaxis=dict(type='log', range=[math.log10(max(y_lo, 1)), math.log10(max(y_hi, y_lo+1))]),
                    template='plotly_dark', height=480
                )
                ct.plotly_chart(fig, width="stretch")
                ct.caption(f"Truc Y (log): {y_lo:.0f} - {y_hi:.0f} min | Vung sang = song co energy manh.")
            
            with stab_con:
                st.subheader("Bieu quyet Da so (5 Phuong phap)")
                if consensus:
                    st.dataframe(pd.DataFrame([{"": c['icon'], "Chu ky (min)": f"{c['period_min']:.1f}", "Bien do TB": f"{c['amplitude']:.5f}", "Xac nhan": f"{c['n_methods']}/5", "Tin cay": c['confidence'], "Cac PP": ", ".join(c['methods'])} for c in consensus]), width="stretch")
                n_conf = sum(1 for c in consensus if c['n_methods'] >= 3)
                st.metric("Song xac nhan (>=3/5 PP)", f"{n_conf}/{len(consensus)}")
                st.markdown(f"**Do tin cay (NSX: {tol:.4f} hPa | San nhieu: {dynamic_noise_floor:.5f} hPa)**")
                wr = []
                for name, sig in filtered_signals.items():
                    amp = (sig.max()-sig.min())/2; snr_t = amp/tol if tol>0 else 999; snr_e = amp/dynamic_noise_floor if dynamic_noise_floor>0 else 999
                    def gs(s): return "✅" if s>=3 else ("🟡" if s>=1.5 else "🔴")
                    ps = name.split('(')[1].replace('m)','') if '(' in name else '0'
                    try: pv=float(ps)
                    except: pv=0
                    wr.append({"Lop": ("🌫️ " if pv>total_duration_mins else "")+name.split(' ')[0], "T(m)": ps, "Amp": f"{amp:.4f}", "SNR_NSX": f"{snr_t:.1f}x", "SNR_Emp": f"{snr_e:.1f}x", "": gs(snr_e)})
                st.dataframe(pd.DataFrame(wr), width="stretch")
                fig_c = px.line(df_waves_plot, x='Datetime', y=list(filtered_signals.keys()), title="Tat ca Dai Song Ket Hop", template="plotly_dark", render_mode="svg")
                fig_c.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
                draw_tide_blocks(fig_c, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top'); draw_tide_blocks(fig_c, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
                fig_c.update_xaxes(title=None); st.plotly_chart(fig_c, width="stretch")
            
            with stab_fft:
                st.subheader("Phuong phap 1: FFT (Fast Fourier Transform)")
                st.caption("Zero-padded FFT. Do nhay cao nhat, nhung cung nhieu 'gai' nhieu nhat.")
                draw_spectrum(result_fft, st)
                # Usability chart
                fig_u = px.line(df_waves_plot, x='Datetime', y=list(filtered_signals.keys()), title="Bieu do Kha dung (Da loc San Nhieu)", template="plotly_dark", render_mode="svg")
                for tr in fig_u.data:
                    s = filtered_signals[tr.name]; a = (s.max()-s.min())/2
                    ps = tr.name.split('(')[1].replace('m)','') if '(' in tr.name else '0'
                    try: pv=float(ps)
                    except: pv=0
                    if pv>total_duration_mins or a<dynamic_noise_floor: tr.line.dash='dot'; tr.opacity=0.3
                    else: tr.line.width=2; tr.opacity=1.0
                fig_u.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
                fig_u.add_hline(y=dynamic_noise_floor, line_dash="dash", line_color="rgba(255,255,255,0.5)", annotation_text="+ San Nhieu")
                fig_u.add_hline(y=-dynamic_noise_floor, line_dash="dash", line_color="rgba(255,255,255,0.5)", annotation_text="- San Nhieu")
                fig_u.update_xaxes(title=None); st.plotly_chart(fig_u, width="stretch")
            
            with stab_psd:
                st.subheader("Phuong phap 2: PSD Welch")
                st.caption("Lam min nhieu bang trung binh Welch. Dinh nhon = song ben vung.")
                draw_spectrum(result_psd, st)
            
            with stab_stft:
                st.subheader("Phuong phap 3: STFT (Spectrogram)")
                st.caption("Ban do Thoi gian-Tan so. Vung sang ngang = song ben vung.")
                draw_spectrum(result_stft, st)
                draw_heatmap(result_stft, st, "Spectrogram")
            
            with stab_cwt:
                st.subheader("Phuong phap 4: CWT (Continuous Wavelet Transform)")
                st.caption("Wavelet Morlet: co gian cua so theo tan so.")
                draw_spectrum(result_cwt, st)
                draw_heatmap(result_cwt, st, "Scalogram")
            
            with stab_hht:
                st.subheader("Phuong phap 5: HHT/EMD (Hilbert-Huang Transform)")
                st.caption("Phan tach song thanh IMF roi tinh tan so tuc thoi.")
                if result_hht.get('imfs'):
                    for imf_d in result_hht['imfs'][:6]:
                        idx = imf_d['imf_index']; per = imf_d['median_period_min']; amp = imf_d['mean_amplitude']
                        fig_i = px.line(x=imf_d['time_hours'], y=imf_d['signal'], title=f"IMF {idx+1}: ~{per:.1f}min | ~{amp:.5f} hPa", template="plotly_dark", render_mode="svg")
                        fig_i.update_layout(height=200, margin=dict(t=30, b=10)); fig_i.update_xaxes(title="Gio"); fig_i.update_yaxes(title="hPa")
                        st.plotly_chart(fig_i, width="stretch")
                    if result_hht['waves']:
                        st.dataframe(pd.DataFrame([{"IMF": f"#{w.get('imf_index',0)+1}", "Chu ky (min)": f"{w['period_min']:.1f}", "Bien do": f"{w.get('amplitude',0):.5f}"} for w in result_hht['waves'][:6]]), width="stretch")
                else:
                    st.warning("EMD khong the phan tach tin hieu nay.")
            
            
        with tab3:
            st.header("3. Trạng thái Khí quyển (Atmosphere State)", help="Khảo sát độ hỗn loạn (Turbulence) và độ tĩnh lặng của dòng chảy không khí. Càng hỗn loạn (Entropy cao) hệ thống khí quyển cành bất ổn định (có thể giông lốc).")
            with st.spinner("Đang tính Permutation Entropy..."):
                df_l3, metrics_l3 = analyze_layer_3(df_base, fs=fs)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Spectral Slope", f"{metrics_l3['Global Spectral Slope']:.4f}", help="Hệ số góc phổ Kolmogorov. Ở quy mô Synoptic và Mesoscale gió, hệ số này thường quanh mốc -5/3 (-1.67) cho dòng chảy rối 3D (3D-Turbulence). Lớn hơn mức này (trần truồng, ví dụ -3) hệ thống tĩnh lại thành phân tầng 2D chuyên dẹt dọc theo bề mặt đất.")
            c2.metric("Max Permutation Entropy", f"{metrics_l3['Max Entropy']:.4f}", help="Hệ số phân hóa thứ tự cao nhất (Chạy từ 0 đến 1). Giá trị đạt trên 0.95 thường báo hiệu sự thay đổi dữ dội phá vỡ mô hình dự đoán (cực kỳ rối).")
            c3.metric("Min Permutation Entropy", f"{metrics_l3['Min Entropy']:.4f}", help="Trạng thái yên bình (Laminar Flow) nhất của khí quyển được ghi lại trong suốt chiều dài dữ liệu.")
            
            fig3 = px.line(df_l3, x='Datetime', y='Permutation Entropy', title="Permutation Entropy (Rolling 10m)", template="plotly_dark", render_mode="svg")
            
            draw_tide_blocks(fig3, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top')
            draw_tide_blocks(fig3, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
            draw_tide_blocks(fig3, tide_max_blocks, None, '#ffaa00', 'Tide Max', 'top')
            draw_tide_blocks(fig3, tide_min_blocks, None, '#ffaa00', 'Tide Min', 'top')
            
            # Highlight NaN regions (Data Initialization / Corruption)
            nan_mask = df_l3['Permutation Entropy'].isna()
            if nan_mask.any():
                start_nan = df_l3.loc[nan_mask, 'Datetime'].iloc[0]
                end_nan = df_l3.loc[nan_mask, 'Datetime'].iloc[-1]
                fig3.add_vrect(x0=start_nan, x1=end_nan, fillcolor="red", opacity=0.3, layer="below", line_width=0, 
                               annotation_text="Dữ liệu Khởi tạo (NaN)", annotation_position="top left", annotation_font_color="red")
            
            fig3.update_xaxes(title=None)
            st.plotly_chart(fig3, width="stretch")
            
            fig3b = px.line(df_l3, x='Datetime', y='Rolling Variance (10m)', title="Rolling Variance (Proxy for Turbulence)", template="plotly_dark", render_mode="svg")
            
            draw_tide_blocks(fig3b, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top')
            draw_tide_blocks(fig3b, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
            draw_tide_blocks(fig3b, tide_max_blocks, None, '#ffaa00', 'Tide Max', 'top')
            draw_tide_blocks(fig3b, tide_min_blocks, None, '#ffaa00', 'Tide Min', 'top')
            
            if nan_mask.any():
                fig3b.add_vrect(x0=start_nan, x1=end_nan, fillcolor="red", opacity=0.3, layer="below", line_width=0, 
                                annotation_text="Dữ liệu Khởi tạo (NaN)", annotation_position="top left", annotation_font_color="red")
                                
            fig3b.update_xaxes(title=None)
            st.plotly_chart(fig3b, width="stretch")
            
        with tab4:
            st.header("4. Nhiễu động cục bộ & Micro-events (32Hz)", help="Khai thác dữ liệu đo với tần số quét siêu cao để tóm gọn các xung Microbaroms kéo dài chưa tới vài giây (Gió thốc giật, cánh quạt, cửa sập hoặc siêu tiếng ồn nhiệt động).")
            df_l4, metrics_l4 = analyze_layer_4(df_32hz)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Gust Proxy (Std)", f"{metrics_l4['Max Gust Proxy']:.4f}", help="Biên độ dao động áp suất cực vi mô bị làm rung lắc bởi Gió giật mạnh (Gust) va đập vào điểm đo. Giá trị cao nghĩa là gió rất hung bạo.")
            c2.metric("Avg Gust Proxy", f"{metrics_l4['Avg Gust Proxy']:.4f}", help="Thể hiện sức gió nền liên tục (Ambient Wind turbulence) rít qua bề mặt thiết bị suốt buổi đo.")
            c3.metric("Pressure Skewness", f"{metrics_l4['Pressure Skewness']:.4f}", help="Độ lệnh chuẩn phân bố. Nếu âm sâu (< -0.5), không khí thốc mạnh trồi lên cao (Updrafts) do bốc hơi hoặc bão. Nếu dương gắt (> 0.5), khối khí lạnh năng trên mây đang nén dập xuống đất (Downdrafts / Microburst).")
            
            # Subsample for rendering performance in browser (use 1Hz max gust to preserve peaks and connect points)
            df_l4_plot = df_l4.set_index('Datetime').resample('1s').max().reset_index().dropna(subset=['Gust Proxy (Rolling Std)'])
            
            fig4 = px.line(df_l4_plot, x='Datetime', y='Gust Proxy (Rolling Std)', title="Max Gust Proxy (1s Downsampled for plotting)", template="plotly_dark", render_mode="svg")
            fig4.update_xaxes(title=None)
            st.plotly_chart(fig4, width="stretch")
            
        with tab5:
            st.header("5. Kết nối Hành tinh & External Anchor", help="Tìm kiếm sự đồng bộ của Sóng Khí quyển (Teleconnection) giữa các trạm đo cách xa nhau dọc theo hành tinh và căn chỉnh áp suất hệ quy chiếu chuẩn.")
            
            df_l2_baseline_waves = None
            if baseline_folder != "None":
                base_path = os.path.join(DATA_DIR, baseline_folder)
                _, df_base_compare = get_processed_data(base_path, target_fs=fs)
                df_l2_baseline_waves, _, _, _, _, _, _ = analyze_layer_2(df_base_compare, fs=fs)
                # Convert dict to df for convenience
                df_l2_baseline_waves = pd.DataFrame(df_l2_baseline_waves)
                
            metrics_l5 = analyze_layer_5(pd.DataFrame(filtered_signals), df_l2_baseline_waves, external_mslp)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Boss Amplitude", f"{metrics_l5.get('Boss Wave Amplitude (Current)', 0):.4f}", help="Biên độ thực tế của dòng sóng Boss (chu kỳ khổng lồ nhất) đang ngầm quét qua trạm đo của bạn.")
            if 'Boss Amplitude Ratio' in metrics_l5:
                c2.metric("Boss vs Baseline Ratio", f"{metrics_l5['Boss Amplitude Ratio']:.2f}x", help="Tần suất sức mạnh của sóng Boss hiện hành so sánh với hồ sơ gốc (Baseline). Lớn hơn 1x nghĩa là bầu trời đang bị khuấy động mãnh liệt hơn quá khứ.")
            c3.metric("MSLP Anchor", f"{metrics_l5.get('Current MSLP Ref', 0)}", help="Áp suất tham chiếu quy mặt nước biển chuẩn (Mean Sea Level Pressure) lấy từ nguồn METAR quốc tế để làm mỏ neo gỡ sai số.")
            
            if 'Boss Amplitude Ratio' in metrics_l5:
                st.info("💡 Tỷ lệ này cho phép dự đoán độ mạnh của dải áp cao/dòng xiết khu vực so với dữ liệu quá khứ.")
                
        with tab6:
            st.header("6. Đánh giá Thiết bị & Độ tin cậy (Device Evaluation)", help="Phân tích cơ học lượng tử của dòng dữ liệu nhằm mổ xẻ chất lượng điện tử nội tại của bản thân con chip Cảm biến trước khi tin tưởng các chỉ số vật lý nó cung cấp.")
            st.write("Đánh giá chất lượng dữ liệu thu thập được từ thiết bị đo để xác định độ tin cậy của các phân tích vật lý.")
            
            with st.spinner("Đang định dạng báo cáo thiết bị..."):
                pass # Already computed at the top level
                
            c1, c2, c3, c4 = st.columns(4)
            c5, c6, c7, c8 = st.columns(4)
            
            # Formulate reliability color
            score = metrics_device['Reliability Score']
            if score >= 90:
                score_str = f"🟢 {score:.1f}% (Tuyệt vời)"
            elif score >= 70:
                score_str = f"🟡 {score:.1f}% (Khá)"
            elif score >= 50:
                score_str = f"🟠 {score:.1f}% (Trung bình)"
            else:
                score_str = f"🔴 {score:.1f}% (Kém)"
                
            c1.metric("Độ Tin Cậy NSX", score_str, help="Điểm quy đổi từ Tỷ lệ mất gói tin, mức độ nhiễu điện tử và độ phân giải.")
            c2.metric("Mất Dữ Liệu", f"{metrics_device['Data Missing Ratio (%)']:.4f}%", help="Tỷ lệ gói tin bị rớt mạng.")
            c3.metric("Độ Phân Giải Thực", f"{metrics_device['Empirical Resolution (hPa)']:.6f} hPa", help="Bước nhảy nhạy bén thực sự ghi nhận được ngoài môi trường.")
            c4.metric("Dư Số Tổng (Total)", f"{metrics_device.get('Total Residual RMS (hPa)', 0):.5f} hPa", help="Tổng độ lệnh chuẩn sau khi loại bỏ áp suất thuỷ triều. Bao gồm sóng + gió + nhiễu.")
            
            # Spectral Decomposition Row
            c5.metric("Nhiễu Điện Tử (>1Hz)", f"{metrics_device['Empirical White Noise Std (hPa)']:.6f} hPa", help="Nhiễu trắng nội tại chip silicon (White Noise Floor). Sạch nhất.")
            c6.metric("Turbulence & Gió (1s-1m)", f"{metrics_device.get('Empirical Turbulence RMS (hPa)', 0):.6f} hPa", help="Nhiễu động cơ học cục bộ (Gió quạt, người đi lại, tiếng ồn).")
            c7.metric("Khí Quyển (1m-160m)", f"{metrics_device.get('Empirical Waves RMS (hPa)', 0):.6f} hPa", help="Năng lượng của Sóng Trọng trường Mesoscale. ĐÂY LÀ ĐỐI TƯỢNG NGHIÊN CỨU!")
            c8.metric("Nhiễu Trôi VLF (>160m)", f"{metrics_device['Empirical Pink Noise RMS (hPa)']:.6f} hPa", help="Khấu hao tụt áp tụ điện (VLF Drift) do thay đổi nhiệt độ máy. Sàn nhiễu cuối cùng.")
            
            st.markdown("### Khuyến nghị Phân tích (Dựa trên thông số phần cứng)")
            rec_html = "<ul>"
            
            if emp_pink_noise < tol:
                rec_html += f"<li>✅ Nhiễu môi trường ({emp_pink_noise:.5f}) thấp hơn sai số lý thuyết của cảm biến ({tol}). Dữ liệu khá sạch.</li>"
            else:
                rec_html += f"<li>⚠️ <b>Cảm biến lang thang (Wandering)</b>: Nhiễu môi trường (Drift: {emp_pink_noise:.5f}) cao hơn sai số lý thuyết ({tol}). Các sóng siêu dài dễ bị lẫn vào hiện tượng Drift.</li>"
                
            if metrics_device['Data Missing Ratio (%)'] > 1.0:
                rec_html += "<li>⚠️ Cảnh báo: Tỉ lệ mất gói tin khá cao, có thể ảnh hưởng đến kết quả biến đổi Fourier (Layer 2) và Entropy (Layer 3).</li>"
            else:
                rec_html += "<li>✅ Tính liên tục của chuỗi thời gian rất tốt, đảm bảo độ chính xác cho phân tích tần số (FFT).</li>"
                
            rec_html += "</ul>"
            st.markdown(rec_html, unsafe_allow_html=True)
            
            st.markdown("### Đánh giá Độ chính xác theo Dải Sóng (Quét động theo Layer 2)")
            wave_rec_html = "<ul>"
            
            # File duration in minutes
            total_duration_mins = duration.total_seconds() / 60
            
            for name, sig in filtered_signals.items():
                amplitude = (sig.max() - sig.min()) / 2
                snr_tol = amplitude / tol if tol > 0 else 999
                dynamic_noise_floor = emp_turb + emp_white_noise
                snr_emp = amplitude / dynamic_noise_floor if dynamic_noise_floor > 0 else 999
                
                # Extract period to check if it's hypothetical
                period_str = name.split('(')[1].replace('m)', '') if '(' in name else '0'
                try: period_val = float(period_str)
                except ValueError: period_val = 0
                
                is_hypothetical = period_val > total_duration_mins
                prefix = "🌫️ <b>[Giả định]</b>" if is_hypothetical else ""
                
                if snr_emp >= 3.0:
                    wave_rec_html += f"<li>✅ {prefix} <b>{name}:</b> Rất Tốt (SNR Thực tế: {snr_emp:.1f}x | SNR NSX: {snr_tol:.1f}x). Biên độ dao động vật lý vượt xa ngưỡng nhiễu động của máy. {('Tuy nhiên, sóng này dài hơn thời gian đo nên chỉ mang tính tham khảo.' if is_hypothetical else 'Hoàn toàn tin cậy.')}</li>"
                elif snr_emp >= 1.5:
                    wave_rec_html += f"<li>🟡 {prefix} <b>{name}:</b> Cảnh Báo (SNR Thực tế: {snr_emp:.1f}x | SNR NSX: {snr_tol:.1f}x). Sóng bị mờ nhạt hoặc tiệm cận với biên độ của sàn nhiễu Gió/Điện từ.</li>"
                else:
                    wave_rec_html += f"<li>🔴 {prefix} <b>{name}:</b> Suy thoái (SNR Thực tế: {snr_emp:.1f}x | SNR NSX: {snr_tol:.1f}x). Sàn nhiễu động ({dynamic_noise_floor:.3f} hPa) dập tắt hoàn toàn bước sóng. Rất dễ bị diễn giải sai!</li>"
                    
            wave_rec_html += "</ul>"
            st.markdown(wave_rec_html, unsafe_allow_html=True)
            
            # Plot High Frequency Noise
            # To avoid huge UI lag, plot downsampled noise
            df_noise = pd.DataFrame({'Datetime': df_32hz['Datetime'], 'Noise': metrics_device['White Noise Signal']})
            df_noise_plot = df_noise.iloc[::32] # downsample to 1Hz
            
            fig_noise = px.line(df_noise_plot, x='Datetime', y='Noise', title="Nhiễu phần cứng/môi trường > 16Hz (Đã Downsample 1Hz để hiển thị)", template="plotly_dark", render_mode="svg")
            fig_noise.add_hline(y=tol, line_dash="dash", line_color="red", annotation_text="+ Tolearance")
            fig_noise.add_hline(y=-tol, line_dash="dash", line_color="red", annotation_text="- Tolearance")
            fig_noise.update_xaxes(title=None)
            st.plotly_chart(fig_noise, width="stretch")
            
            # Hiển thị thông số phần cứng
            st.markdown("### Thống số Phần cứng Gốc (Từ Hệ điều hành)")
            st.json(device_info)

        # --- Export Features ---
        st.sidebar.markdown("---")
        if st.sidebar.button("Export Analysis Summary"):
            with st.spinner("Đang xuất báo cáo..."):
                out_path = export_features(folder_path, metrics_l1, {'Bands': 'Exported in full dataframe'}, metrics_l3, metrics_l4, metrics_l5)
                st.sidebar.success(f"Đã lưu tại: {out_path}")

if __name__ == "__main__":
    main()
