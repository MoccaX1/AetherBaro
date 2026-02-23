import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
from analysis import (
    load_and_preprocess_data, 
    analyze_layer_1, 
    analyze_layer_2,
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
        st.caption(rf"**Thiết bị đo:** {device_info['Model']} | **Cảm biến Áp suất:** {device_info['Sensor']} | **Sai số phần cứng (Tolerance):** $\pm{tolerance}$ hPa")
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
            st.header("2. Hệ thống Sóng (Boss/Mother/Child)", help="Dùng phân tích phổ biến đổi Fourier (FFT) và lưới lọc kỹ thuật số (Bandpass) để tách sóng áp suất khổng lồ thành nhiều dải Sóng Trọng trường vi mô (Gravity Waves) xếp chồng lên nhau do địa hình hoặc mây giông tạo ra.")
            filtered_signals, freqs, power, periods_min, power_valid, exact_peak_period, dynamic_bands = analyze_layer_2(df_base, fs=fs)
            
            df_waves = df_base[['Datetime']].copy()
            for name, sig in filtered_signals.items():
                df_waves[name] = sig
                
            df_waves_plot = df_waves.iloc[::plot_step] if fs > 1.0 else df_waves
            
            macro_cols = [c for c in filtered_signals.keys() if 'Micro' not in c]
            micro_cols = [c for c in filtered_signals.keys() if 'Micro' in c]
            
            # 1. Combined Plot (All Waves)
            fig_waves_combined = px.line(df_waves_plot, x='Datetime', y=list(filtered_signals.keys()), 
                                         title="Tất cả Dải Sóng Kết Hợp (Macro + Micro)", template="plotly_dark", render_mode="svg")
            fig_waves_combined.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            draw_tide_blocks(fig_waves_combined, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top')
            draw_tide_blocks(fig_waves_combined, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
            draw_tide_blocks(fig_waves_combined, tide_max_blocks, None, '#ffaa00', 'Tide Max', 'top')
            draw_tide_blocks(fig_waves_combined, tide_min_blocks, None, '#ffaa00', 'Tide Min', 'top')
            fig_waves_combined.update_xaxes(title=None)
            st.plotly_chart(fig_waves_combined, width="stretch")
                
            # 2. Separated Macro Waves
            fig_waves = px.line(df_waves_plot, x='Datetime', y=macro_cols, 
                                title="Các Dải Sóng Dài (Boss/Mother/Child - Vĩ mô)", template="plotly_dark", render_mode="svg")
            fig_waves.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            draw_tide_blocks(fig_waves, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top')
            draw_tide_blocks(fig_waves, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
            draw_tide_blocks(fig_waves, tide_max_blocks, None, '#ffaa00', 'Tide Max', 'top')
            draw_tide_blocks(fig_waves, tide_min_blocks, None, '#ffaa00', 'Tide Min', 'top')
            fig_waves.update_xaxes(title=None)
            st.plotly_chart(fig_waves, width="stretch")
            
            if micro_cols:
                fig_micro = px.line(df_waves_plot, x='Datetime', y=micro_cols, 
                                    title="Dải Sóng Ngắn (Micro - Nhiễu động nhiệt)", template="plotly_dark", render_mode="svg", color_discrete_sequence=['#ffaa00'])
                fig_micro.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
                draw_tide_blocks(fig_micro, l1_max_blocks, None, '#ff4b4b', 'Pmax', 'top')
                draw_tide_blocks(fig_micro, l1_min_blocks, None, '#00d4ff', 'Pmin', 'top')
                draw_tide_blocks(fig_micro, tide_max_blocks, None, '#ffaa00', 'Tide Max', 'top')
                draw_tide_blocks(fig_micro, tide_min_blocks, None, '#ffaa00', 'Tide Min', 'top')
                fig_micro.update_xaxes(title=None)
                st.plotly_chart(fig_micro, width="stretch")
            
            df_fft = pd.DataFrame({'Period (minutes)': periods_min, 'Power': power_valid})
            df_fft = df_fft[(df_fft['Period (minutes)'] >= 10) & (df_fft['Period (minutes)'] <= 300)]
            
            fig_fft = px.line(df_fft, x='Period (minutes)', y='Power', log_y=True, 
                              title="Phổ năng lượng (Zero-padded FFT)", template="plotly_dark", render_mode="svg")
            
            # Draw dynamic bands
            color_map = {'Boss': 'red', 'Mother': 'green', 'Child': 'blue', 'Micro': 'orange', 'Wildcard': 'purple'}
            for info in dynamic_bands.values():
                low_p, high_p = info['period_range']
                base_name = info['base_name']
                color = color_map.get(base_name, 'gray')
                fig_fft.add_vrect(x0=low_p, x1=high_p, fillcolor=color, opacity=0.15, line_width=0, annotation_text=base_name)
            
            if exact_peak_period is not None:
                fig_fft.add_vline(x=exact_peak_period, line_width=2, line_dash="dash", line_color="white")
                fig_fft.add_annotation(x=exact_peak_period, y=0.95, yref="paper", text=f"Peak: {exact_peak_period:.2f}m", showarrow=True, arrowhead=2, font=dict(color="white"))
                
            fig_fft.update_xaxes(title=None)
            st.plotly_chart(fig_fft, width="stretch")
            
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
            
            with st.spinner("Đang phân tích độ tin cậy thiết bị..."):
                metrics_device = analyze_device_performance(df_32hz, device_info)
                
            c1, c2, c3, c4 = st.columns(4)
            
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
                
            c1.metric("Độ Tin Cậy Dữ Liệu", score_str, help="Điểm tổng thể quy đổi tử Tỷ lệ gián đoạn thông tin, mức độ dơ bẩn của dòng tín hiệu nhiễu cực đại và mật độ bước nhảy số.")
            c2.metric("Tỉ lệ Mất Dữ Liệu", f"{metrics_device['Data Missing Ratio (%)']:.4f}%", help="Tỷ lệ những gói tin (Packets) bị bay màu trên đường truyền hoặc vi xử lý bị kẹt không lấy mẫu kịp khung giờ quy định.")
            c3.metric("Nhiễu Cao Tần (Std)", f"{metrics_device['Empirical Noise Std (hPa)']:.6f} hPa", help="Độ lệch chuẩn của Sàn nhiễu trắng (White Noise Floor). Tín hiệu giả sinh ra do giao thoa điện từ trường và rung động nhiệt kế của điện dung nội tại cảm biến.")
            c4.metric("Độ Phân Giải Thực Tế", f"{metrics_device['Empirical Resolution (hPa)']:.6f} hPa", help="Bước nhảy nhạy bén nhỏ nhất thực sự đo đếm được (Grid Resolution) ở ngoài môi trường thay vì con số lý tưởng trong phòng thí nghiệm của Apple/Bosch.")
            
            st.markdown("### Khuyến nghị Phân tích (Dựa trên thông số phần cứng)")
            rec_html = "<ul>"
            tol = device_info.get('Resolution', 0.01)
            emp_noise = metrics_device['Empirical Noise Std (hPa)']
            
            if emp_noise < tol:
                rec_html += f"<li>✅ Nhiễu môi trường ({emp_noise:.5f}) thấp hơn sai số lý thuyết của cảm biến ({tol}). Dữ liệu rất sạch.</li>"
            else:
                rec_html += f"<li>⚠️ Nhiễu môi trường ({emp_noise:.5f}) cao hơn sai số lý thuyết ({tol}). Các hiện tượng vi mô ở Layer 4 có thể bị lẫn nhiễu vật lý.</li>"
                
            if metrics_device['Data Missing Ratio (%)'] > 1.0:
                rec_html += "<li>⚠️ Cảnh báo: Tỉ lệ mất gói tin khá cao, có thể ảnh hưởng đến kết quả biến đổi Fourier (Layer 2) và Entropy (Layer 3).</li>"
            else:
                rec_html += "<li>✅ Tính liên tục của chuỗi thời gian rất tốt, đảm bảo độ chính xác cho phân tích tần số (FFT).</li>"
                
            rec_html += "</ul>"
            st.markdown(rec_html, unsafe_allow_html=True)
            
            st.markdown("### Đánh giá Độ chính xác theo Dải Sóng (Layer 2 & 4)")
            wave_rec_html = "<ul>"
            emp_res = metrics_device['Empirical Resolution (hPa)']
            limit = max(tol, emp_noise, emp_res)
            
            if limit <= 0.02:
                wave_rec_html += "<li>✅ <b>Dải Vi mô (Micro - <10m):</b> Rất Tốt. Dữ liệu đủ sạch để quan sát nhiễu động nhiệt và gió giật (<0.02 hPa).</li>"
            else:
                wave_rec_html += f"<li>⚠️ <b>Dải Vi mô (Micro - <10m):</b> Kém chính xác. Nhiễu phần cứng ({limit:.3f} hPa) lớn hơn biên độ sóng vi mô thông thường.</li>"
                
            if limit <= 0.05:
                wave_rec_html += "<li>✅ <b>Dải Child (35-45m):</b> Độ chính xác cao. Dễ dàng nhận diện các dao động áp suất cục bộ trung bình.</li>"
            else:
                wave_rec_html += f"<li>⚠️ <b>Dải Child (35-45m):</b> Có thể lẫn nhiễu. Giới hạn cảm biến ({limit:.3f} hPa) tiệm cận với biên độ sóng Child.</li>"
                
            if limit <= 0.2:
                wave_rec_html += "<li>✅ <b>Dải Mother (75-85m):</b> Rất Tốt. Sóng ổn định định kỳ của bầu khí quyển hoàn toàn tin cậy.</li>"
            else:
                wave_rec_html += f"<li>⚠️ <b>Dải Mother (75-85m):</b> Cảnh báo độ chính xác bị suy giảm.</li>"
                
            wave_rec_html += "<li>✅ <b>Dải Boss (150-180m):</b> Hoàn toàn chính xác. Biên độ sóng Synoptic lớn (>0.5 hPa) dễ dàng vượt qua mọi giới hạn nhiễu phần cứng.</li>"
            wave_rec_html += "</ul>"
            st.markdown(wave_rec_html, unsafe_allow_html=True)
            
            # Plot High Frequency Noise
            # To avoid huge UI lag, plot downsampled noise
            df_noise = pd.DataFrame({'Datetime': df_32hz['Datetime'], 'Noise': metrics_device['Noise Signal']})
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
