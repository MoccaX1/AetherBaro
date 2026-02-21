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
    export_features
)

st.set_page_config(page_title="High-Res Pressure Analyzer", layout="wide")

st.title("Phân Tích Dữ Liệu Áp Suất Thời Gian Thực (5 Lớp Vật Lý)")

DATA_DIR = r"D:\Thanh\Pressure\Project\data"

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
                
            vendor_rows = df[df['property'] == 'pressure Vendor']
            name_rows = df[df['property'] == 'pressure Name']
            vendor = vendor_rows['value'].values[0] if not vendor_rows.empty else ""
            name = name_rows['value'].values[0] if not name_rows.empty else ""
            if vendor or name:
                info['Sensor'] = f"{vendor} {name}".strip()
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
    
    # --- Sampling Rate Option ---
    st.sidebar.markdown("---")
    resample_option = st.sidebar.selectbox(
        "Tần số phân tích (Performance/Detail):", 
        ["1Hz (Mặc định - Nhanh)", "5Hz (Chi tiết)", "32Hz (Bản gốc - Nặng)"]
    )
    
    if "1Hz" in resample_option:
        fs = 1.0
    elif "5Hz" in resample_option:
        fs = 5.0
    else:
        fs = 32.0
    
    if selected_folder:
        folder_path = os.path.join(DATA_DIR, selected_folder)
        
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
        
        # Load Device Info
        device_info = load_device_info(folder_path)
        tolerance = device_info['Resolution']
        
        st.write(f"### Tổng quan dữ liệu Gốc (Đã Resample {int(fs)}Hz cho hiệu năng)")
        st.caption(f"**Thiết bị đo:** {device_info['Model']} | **Cảm biến Áp suất:** {device_info['Sensor']} | **Sai số phần cứng (Tolerance):** $\pm{tolerance}$ hPa")
        
        # Plot downsampled if it's 32Hz to avoid massive browser lag
        plot_df = df_base.iloc[::int(max(1, fs))] if fs == 32.0 else df_base
        
        fig = px.line(plot_df, x='Datetime', y='Pressure (hPa)', title=f"Áp suất - {selected_folder} ({int(fs)}Hz Data)",
                     template="plotly_dark")
        fig.update_xaxes(title=None)
                     
        # Extract Min/Max with Dynamic Sensor Error Margin
        p_max_val = plot_df['Pressure (hPa)'].max()
        p_min_val = plot_df['Pressure (hPa)'].min()
        
        t_max_series = plot_df[p_max_val - plot_df['Pressure (hPa)'] <= tolerance]['Datetime']
        t_min_series = plot_df[plot_df['Pressure (hPa)'] - p_min_val <= tolerance]['Datetime']
        
        # Plot all points rapidly as a single scatter trace
        fig.add_scatter(x=t_max_series, y=plot_df.loc[t_max_series.index, 'Pressure (hPa)'], mode='markers', marker=dict(color='#ff4b4b', size=8), showlegend=False, name="Pmax Area")
        fig.add_scatter(x=t_min_series, y=plot_df.loc[t_min_series.index, 'Pressure (hPa)'], mode='markers', marker=dict(color='#00d4ff', size=8), showlegend=False, name="Pmin Area")
        
        # Add labels only for distinct peaks (separated by 30 mins) to prevent annotation lag
        def annotate_clusters_overview(t_series, p_val, color, prefix, y_pos):
            if t_series.empty: return
            clusters = [t_series.iloc[0]]
            for t in t_series.iloc[1:]:
                if (t - clusters[-1]).total_seconds() > 1800:
                    clusters.append(t)
            for t_rep in clusters:
                fig.add_annotation(x=t_rep, y=p_val, text=f"{prefix}: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                fig.add_vline(x=t_rep, line_width=1, line_dash="dot", line_color=color)
                fig.add_annotation(x=t_rep, y=0.0, yref="paper", yanchor="bottom", text=t_rep.strftime('%H:%M:%S'), showarrow=False, font=dict(color=color), xanchor="left")

        annotate_clusters_overview(t_max_series, p_max_val, '#ff4b4b', 'Pmax', 'top')
        annotate_clusters_overview(t_min_series, p_min_val, '#00d4ff', 'Pmin', 'top')
        
        st.plotly_chart(fig, width="stretch")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Layer 1 (Synoptic)", 
            "Layer 2 (Waves)", 
            "Layer 3 (Atmosphere State)", 
            "Layer 4 (Micro)", 
            "Layer 5 (Planetary)"
        ])
        
        with tab1:
            st.header("1. Động lực học Quy mô Lớn")
            df_l1, metrics_l1 = analyze_layer_1(df_base, fs=fs)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Synoptic Trend", metrics_l1['Synoptic Trend'])
            c2.metric("Max dP/dt", f"{metrics_l1['Max dP/dt']:.4f} hPa/hr")
            c3.metric("Min dP/dt", f"{metrics_l1['Min dP/dt']:.4f} hPa/hr")
            c4.metric("Moon Phase", f"{metrics_l1.get('Avg Moon Phase', 0):.1f} days")
            
            # --- Performance Boost for Plotly Rendering ---
            # Max 1Hz for visualization to prevent browser freezing on dense 32Hz data
            plot_step = int(max(1, fs))
            df_l1_plot = df_l1.iloc[::plot_step] if fs > 1.0 else df_l1
            
            fig1 = px.line(df_l1_plot, x='Datetime', y=['Pressure (hPa)', 'Smoothed (1h)', 'Theoretical Tide (Solar+Lunar)', 'Residual Pressure (Synoptic Only)'], 
                           title="Synoptic Trend & Atmospheric Tides", template="plotly_dark")
            
            # Make theoretical tide dashed for clarity
            fig1.update_traces(line=dict(dash='dash'), selector=dict(name='Theoretical Tide (Solar+Lunar)'))
            fig1.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig1.update_xaxes(title=None)
            
            # Add annotations and projections for multiple extremum points with Dynamic Tolerance
            # 1. Base Pressure Max/Min
            p_max_l1 = df_l1['Pressure (hPa)'].max()
            p_min_l1 = df_l1['Pressure (hPa)'].min()
            
            t_max_l1_series = df_l1[p_max_l1 - df_l1['Pressure (hPa)'] <= tolerance]['Datetime']
            t_min_l1_series = df_l1[df_l1['Pressure (hPa)'] - p_min_l1 <= tolerance]['Datetime']
            
            # 2. Theoretical Tide Max/Min (We can use a slightly larger tolerance for tides if needed, but hardware tolerance is fine)
            p_max_tide = df_l1['Theoretical Tide (Solar+Lunar)'].max()
            p_min_tide = df_l1['Theoretical Tide (Solar+Lunar)'].min()
            
            t_max_tide_series = df_l1[p_max_tide - df_l1['Theoretical Tide (Solar+Lunar)'] <= tolerance]['Datetime']
            t_min_tide_series = df_l1[df_l1['Theoretical Tide (Solar+Lunar)'] - p_min_tide <= tolerance]['Datetime']
            
            # Fast bulk scatter plotting
            fig1.add_scatter(x=t_max_l1_series, y=df_l1.loc[t_max_l1_series.index, 'Pressure (hPa)'], mode='markers', marker=dict(color='#ff4b4b', size=8), showlegend=False)
            fig1.add_scatter(x=t_min_l1_series, y=df_l1.loc[t_min_l1_series.index, 'Pressure (hPa)'], mode='markers', marker=dict(color='#00d4ff', size=8), showlegend=False)
            fig1.add_scatter(x=t_max_tide_series, y=df_l1.loc[t_max_tide_series.index, 'Theoretical Tide (Solar+Lunar)'], mode='markers', marker=dict(color='#ffaa00', size=8), showlegend=False)
            fig1.add_scatter(x=t_min_tide_series, y=df_l1.loc[t_min_tide_series.index, 'Theoretical Tide (Solar+Lunar)'], mode='markers', marker=dict(color='#ffaa00', size=8), showlegend=False)
            
            # Cluster text annotations (30 min gaps) to avoid browser freeze
            def annotate_clusters_l1(t_series, p_val, color, prefix, y_pos):
                if t_series.empty: return
                clusters = [t_series.iloc[0]]
                for t in t_series.iloc[1:]:
                    if (t - clusters[-1]).total_seconds() > 1800:
                        clusters.append(t)
                for t_rep in clusters:
                    fig1.add_annotation(x=t_rep, y=p_val, text=f"{prefix}: {p_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30 if y_pos=='top' else 30, font=dict(color=color))
                    fig1.add_vline(x=t_rep, line_width=1, line_dash="dot", line_color=color)
                    fig1.add_annotation(x=t_rep, y=0.0, yref="paper", yanchor="bottom", text=t_rep.strftime('%H:%M:%S'), showarrow=False, font=dict(color=color), xanchor="left")

            annotate_clusters_l1(t_max_l1_series, p_max_l1, '#ff4b4b', 'Pmax', 'top')
            annotate_clusters_l1(t_min_l1_series, p_min_l1, '#00d4ff', 'Pmin', 'top')
            annotate_clusters_l1(t_max_tide_series, p_max_tide, '#ffaa00', 'Tide Max', 'top')
            annotate_clusters_l1(t_min_tide_series, p_min_tide, '#ffaa00', 'Tide Min', 'top')
                
            fig1.update_xaxes(title=None)
            st.plotly_chart(fig1, width="stretch")
            
            fig2 = px.line(df_l1_plot, x='Datetime', y=['Raw dP/dt (hPa/hr)', 'dP/dt (hPa/hr)'], title="Tốc độ biến thiên (dP/dt)", template="plotly_dark")
            fig2.update_traces(opacity=0.4, selector=dict(name='Raw dP/dt (hPa/hr)'))
            fig2.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig2.update_xaxes(title=None)
            st.plotly_chart(fig2, width="stretch")
            
            # --- Astronomical Features Chart ---
            fig_astro = px.line(df_l1_plot, x='Datetime', y=['Solar Elevation (deg)', 'Moon Phase (days)'],
                               title="Thông số Thiên văn (Solar Elevation & Moon Phase)", template="plotly_dark")
            fig_astro.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig_astro.update_xaxes(title=None)
            # Put them on secondary y-axis or just rely on plotly autoscaling
            st.plotly_chart(fig_astro, width="stretch")
            
        with tab2:
            st.header("2. Hệ thống Sóng (Boss/Mother/Child)")
            filtered_signals, freqs, power, periods_min, power_valid, exact_peak_period, dynamic_bands = analyze_layer_2(df_base, fs=fs)
            
            df_waves = df_base[['Datetime']].copy()
            for name, sig in filtered_signals.items():
                df_waves[name] = sig
                
            df_waves_plot = df_waves.iloc[::plot_step] if fs > 1.0 else df_waves
            
            macro_cols = [c for c in filtered_signals.keys() if 'Micro' not in c]
            micro_cols = [c for c in filtered_signals.keys() if 'Micro' in c]
            
            # 1. Combined Plot (All Waves)
            fig_waves_combined = px.line(df_waves_plot, x='Datetime', y=list(filtered_signals.keys()), 
                                         title="Tất cả Dải Sóng Kết Hợp (Macro + Micro)", template="plotly_dark")
            fig_waves_combined.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig_waves_combined.update_xaxes(title=None)
            st.plotly_chart(fig_waves_combined, width="stretch")
                
            # 2. Separated Macro Waves
            fig_waves = px.line(df_waves_plot, x='Datetime', y=macro_cols, 
                                title="Các Dải Sóng Dài (Boss/Mother/Child - Vĩ mô)", template="plotly_dark")
            fig_waves.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig_waves.update_xaxes(title=None)
            st.plotly_chart(fig_waves, width="stretch")
            
            if micro_cols:
                fig_micro = px.line(df_waves_plot, x='Datetime', y=micro_cols, 
                                    title="Dải Sóng Ngắn (Micro - Nhiễu động nhiệt)", template="plotly_dark", color_discrete_sequence=['#ffaa00'])
                fig_micro.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
                fig_micro.update_xaxes(title=None)
                st.plotly_chart(fig_micro, width="stretch")
            
            df_fft = pd.DataFrame({'Period (minutes)': periods_min, 'Power': power_valid})
            df_fft = df_fft[(df_fft['Period (minutes)'] >= 10) & (df_fft['Period (minutes)'] <= 300)]
            
            fig_fft = px.line(df_fft, x='Period (minutes)', y='Power', log_y=True, 
                              title="Phổ năng lượng (Zero-padded FFT)", template="plotly_dark")
            
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
            st.header("3. Trạng thái Khí quyển (Atmosphere State)")
            with st.spinner("Đang tính Permutation Entropy..."):
                df_l3, metrics_l3 = analyze_layer_3(df_base, fs=fs)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Spectral Slope", f"{metrics_l3['Global Spectral Slope']:.4f}")
            c2.metric("Max Permutation Entropy", f"{metrics_l3['Max Entropy']:.4f}")
            c3.metric("Min Permutation Entropy", f"{metrics_l3['Min Entropy']:.4f}")
            
            fig3 = px.line(df_l3, x='Datetime', y='Permutation Entropy', title="Permutation Entropy (Rolling 10m)", template="plotly_dark")
            fig3.update_xaxes(title=None)
            st.plotly_chart(fig3, width="stretch")
            
            fig3b = px.line(df_l3, x='Datetime', y='Rolling Variance (10m)', title="Rolling Variance (Proxy for Turbulence)", template="plotly_dark")
            fig3b.update_xaxes(title=None)
            st.plotly_chart(fig3b, width="stretch")
            
        with tab4:
            st.header("4. Nhiễu động cục bộ & Micro-events (32Hz)")
            df_l4, metrics_l4 = analyze_layer_4(df_32hz)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Gust Proxy (Std)", f"{metrics_l4['Max Gust Proxy']:.4f}")
            c2.metric("Avg Gust Proxy", f"{metrics_l4['Avg Gust Proxy']:.4f}")
            c3.metric("Pressure Skewness", f"{metrics_l4['Pressure Skewness']:.4f}")
            
            # Subsample for rendering performance in browser (use 1Hz max gust to preserve peaks and connect points)
            df_l4_plot = df_l4.set_index('Datetime').resample('1s').max().reset_index().dropna(subset=['Gust Proxy (Rolling Std)'])
            
            fig4 = px.line(df_l4_plot, x='Datetime', y='Gust Proxy (Rolling Std)', title="Max Gust Proxy (1s Downsampled for plotting)", template="plotly_dark")
            fig4.update_xaxes(title=None)
            st.plotly_chart(fig4, width="stretch")
            
        with tab5:
            st.header("5. Kết nối Hành tinh & External Anchor")
            
            df_l2_baseline_waves = None
            if baseline_folder != "None":
                base_path = os.path.join(DATA_DIR, baseline_folder)
                _, df_base_compare = get_processed_data(base_path, target_fs=fs)
                df_l2_baseline_waves, _, _, _, _, _, _ = analyze_layer_2(df_base_compare, fs=fs)
                # Convert dict to df for convenience
                df_l2_baseline_waves = pd.DataFrame(df_l2_baseline_waves)
                
            metrics_l5 = analyze_layer_5(pd.DataFrame(filtered_signals), df_l2_baseline_waves, external_mslp)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Boss Amplitude", f"{metrics_l5.get('Boss Wave Amplitude (Current)', 0):.4f}")
            if 'Boss Amplitude Ratio' in metrics_l5:
                c2.metric("Boss vs Baseline Ratio", f"{metrics_l5['Boss Amplitude Ratio']:.2f}x")
            c3.metric("MSLP Anchor", f"{metrics_l5.get('Current MSLP Ref', 0)}")
            
            if 'Boss Amplitude Ratio' in metrics_l5:
                st.info("💡 Tỷ lệ này cho phép dự đoán độ mạnh của dải áp cao/dòng xiết khu vực so với dữ liệu quá khứ.")
                
        # --- Export Features ---
        st.sidebar.markdown("---")
        if st.sidebar.button("Export Analysis Summary"):
            with st.spinner("Đang xuất báo cáo..."):
                out_path = export_features(folder_path, metrics_l1, {'Bands': 'Exported in full dataframe'}, metrics_l3, metrics_l4, metrics_l5)
                st.sidebar.success(f"Đã lưu tại: {out_path}")

if __name__ == "__main__":
    main()
