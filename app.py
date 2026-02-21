import streamlit as st
import os
import pandas as pd
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
        
        st.write(f"### Tổng quan dữ liệu Gốc (Đã Resample {int(fs)}Hz cho hiệu năng)")
        # Plot downsampled if it's 32Hz to avoid massive browser lag
        plot_df = df_base.iloc[::int(max(1, fs))] if fs == 32.0 else df_base
        
        fig = px.line(plot_df, x='Datetime', y='Pressure (hPa)', title=f"Áp suất - {selected_folder} ({int(fs)}Hz Data)",
                     template="plotly_dark")
        fig.update_xaxes(title=None)
                     
        # Extract Min/Max
        idx_max = plot_df['Pressure (hPa)'].idxmax()
        idx_min = plot_df['Pressure (hPa)'].idxmin()
        p_max_val = plot_df.loc[idx_max, 'Pressure (hPa)']
        t_max_val = plot_df.loc[idx_max, 'Datetime']
        p_min_val = plot_df.loc[idx_min, 'Pressure (hPa)']
        t_min_val = plot_df.loc[idx_min, 'Datetime']
        
        fig.add_scatter(x=[t_max_val], y=[p_max_val], mode='markers+text', text=[f"Pmax: {p_max_val:.2f}"], textposition="top center", marker=dict(color='#ff4b4b', size=8), showlegend=False)
        fig.add_scatter(x=[t_min_val], y=[p_min_val], mode='markers+text', text=[f"Pmin: {p_min_val:.2f}"], textposition="bottom center", marker=dict(color='#00d4ff', size=8), showlegend=False)
        
        fig.add_vline(x=t_max_val, line_width=1, line_dash="dot", line_color="#ff4b4b")
        fig.add_annotation(x=t_max_val, y=0.0, yref="paper", yanchor="bottom", text=t_max_val.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#ff4b4b"), xanchor="left")
        
        fig.add_vline(x=t_min_val, line_width=1, line_dash="dot", line_color="#00d4ff")
        fig.add_annotation(x=t_min_val, y=0.0, yref="paper", yanchor="bottom", text=t_min_val.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#00d4ff"), xanchor="left")
        
        st.plotly_chart(fig, use_container_width=True)
        
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
            
            fig1 = px.line(df_l1, x='Datetime', y=['Pressure (hPa)', 'Smoothed (1h)', 'Theoretical Tide (Solar+Lunar)', 'Residual Pressure (Synoptic Only)'], 
                           title="Synoptic Trend & Atmospheric Tides", template="plotly_dark")
            
            # Make theoretical tide dashed for clarity
            fig1.update_traces(line=dict(dash='dash'), selector=dict(name='Theoretical Tide (Solar+Lunar)'))
            fig1.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig1.update_xaxes(title=None)
            
            # Add annotations and projections
            idx_max_l1 = df_l1['Pressure (hPa)'].idxmax()
            idx_min_l1 = df_l1['Pressure (hPa)'].idxmin()
            idx_max_tide = df_l1['Theoretical Tide (Solar+Lunar)'].idxmax()
            idx_min_tide = df_l1['Theoretical Tide (Solar+Lunar)'].idxmin()
            
            t_max_l1 = df_l1.loc[idx_max_l1, 'Datetime']
            p_max_l1 = df_l1.loc[idx_max_l1, 'Pressure (hPa)']
            t_min_l1 = df_l1.loc[idx_min_l1, 'Datetime']
            p_min_l1 = df_l1.loc[idx_min_l1, 'Pressure (hPa)']
            
            t_max_tide = df_l1.loc[idx_max_tide, 'Datetime']
            p_max_tide = df_l1.loc[idx_max_tide, 'Theoretical Tide (Solar+Lunar)']
            t_min_tide = df_l1.loc[idx_min_tide, 'Datetime']
            p_min_tide = df_l1.loc[idx_min_tide, 'Theoretical Tide (Solar+Lunar)']
            
            fig1.add_scatter(x=[t_max_l1], y=[p_max_l1], mode='markers+text', text=[f"Pmax: {p_max_l1:.2f}"], textposition="top center", marker=dict(color='#ff4b4b', size=8), showlegend=False)
            fig1.add_scatter(x=[t_min_l1], y=[p_min_l1], mode='markers+text', text=[f"Pmin: {p_min_l1:.2f}"], textposition="bottom center", marker=dict(color='#00d4ff', size=8), showlegend=False)
            
            fig1.add_scatter(x=[t_max_tide], y=[p_max_tide], mode='markers+text', text=[f"Tide Max: {p_max_tide:.2f}"], textposition="top center", marker=dict(color='#ffaa00', size=8), showlegend=False)
            fig1.add_scatter(x=[t_min_tide], y=[p_min_tide], mode='markers+text', text=[f"Tide Min: {p_min_tide:.2f}"], textposition="bottom center", marker=dict(color='#ffaa00', size=8), showlegend=False)
            
            fig1.add_vline(x=t_max_l1, line_width=1, line_dash="dot", line_color="#ff4b4b")
            fig1.add_annotation(x=t_max_l1, y=0.0, yref="paper", yanchor="bottom", text=t_max_l1.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#ff4b4b"), xanchor="left")
            
            fig1.add_vline(x=t_min_l1, line_width=1, line_dash="dot", line_color="#00d4ff")
            fig1.add_annotation(x=t_min_l1, y=0.0, yref="paper", yanchor="bottom", text=t_min_l1.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#00d4ff"), xanchor="left")
            
            fig1.add_vline(x=t_max_tide, line_width=1, line_dash="dot", line_color="#ffaa00")
            fig1.add_annotation(x=t_max_tide, y=0.0, yref="paper", yanchor="bottom", text=t_max_tide.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#ffaa00"), xanchor="right")
            
            fig1.add_vline(x=t_min_tide, line_width=1, line_dash="dot", line_color="#ffaa00")
            fig1.add_annotation(x=t_min_tide, y=0.0, yref="paper", yanchor="bottom", text=t_min_tide.strftime('%H:%M:%S'), showarrow=False, font=dict(color="#ffaa00"), xanchor="right")
            
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.line(df_l1, x='Datetime', y=['Raw dP/dt (hPa/hr)', 'dP/dt (hPa/hr)'], title="Tốc độ biến thiên (dP/dt)", template="plotly_dark")
            fig2.update_traces(opacity=0.4, selector=dict(name='Raw dP/dt (hPa/hr)'))
            fig2.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig2.update_xaxes(title=None)
            st.plotly_chart(fig2, use_container_width=True)
            
            # --- Astronomical Features Chart ---
            fig_astro = px.line(df_l1, x='Datetime', y=['Solar Elevation (deg)', 'Moon Phase (days)'],
                               title="Thông số Thiên văn (Solar Elevation & Moon Phase)", template="plotly_dark")
            fig_astro.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig_astro.update_xaxes(title=None)
            # Put them on secondary y-axis or just rely on plotly autoscaling
            st.plotly_chart(fig_astro, use_container_width=True)
            
        with tab2:
            st.header("2. Hệ thống Sóng (Boss/Mother/Child)")
            filtered_signals, freqs, power, periods_min, power_valid, exact_peak_period = analyze_layer_2(df_base, fs=fs)
            
            df_waves = df_base[['Datetime']].copy()
            for name, sig in filtered_signals.items():
                df_waves[name] = sig
                
            fig_waves = px.line(df_waves, x='Datetime', y=list(filtered_signals.keys()), 
                                title="Các Dải Sóng (Bandpass Filtered)", template="plotly_dark")
            fig_waves.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=""))
            fig_waves.update_xaxes(title=None)
            st.plotly_chart(fig_waves, use_container_width=True)
            
            df_fft = pd.DataFrame({'Period (minutes)': periods_min, 'Power': power_valid})
            df_fft = df_fft[(df_fft['Period (minutes)'] >= 10) & (df_fft['Period (minutes)'] <= 300)]
            
            fig_fft = px.line(df_fft, x='Period (minutes)', y='Power', log_y=True, 
                              title="Phổ năng lượng (Zero-padded FFT)", template="plotly_dark")
            fig_fft.add_vrect(x0=150, x1=180, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Boss")
            fig_fft.add_vrect(x0=75, x1=85, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Mother")
            fig_fft.add_vrect(x0=35, x1=45, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="Child")
            fig_fft.add_vrect(x0=15, x1=25, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="Micro")
            
            if exact_peak_period is not None:
                fig_fft.add_vline(x=exact_peak_period, line_width=2, line_dash="dash", line_color="white")
                fig_fft.add_annotation(x=exact_peak_period, y=0.95, yref="paper", text=f"Peak: {exact_peak_period:.2f}m", showarrow=True, arrowhead=2, font=dict(color="white"))
                
            fig_fft.update_xaxes(title=None)
            st.plotly_chart(fig_fft, use_container_width=True)
            
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
            st.plotly_chart(fig3, use_container_width=True)
            
            fig3b = px.line(df_l3, x='Datetime', y='Rolling Variance (10m)', title="Rolling Variance (Proxy for Turbulence)", template="plotly_dark")
            fig3b.update_xaxes(title=None)
            st.plotly_chart(fig3b, use_container_width=True)
            
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
            st.plotly_chart(fig4, use_container_width=True)
            
        with tab5:
            st.header("5. Kết nối Hành tinh & External Anchor")
            
            df_l2_baseline_waves = None
            if baseline_folder != "None":
                base_path = os.path.join(DATA_DIR, baseline_folder)
                _, df_base_compare = get_processed_data(base_path, target_fs=fs)
                df_l2_baseline_waves, _, _, _, _, _ = analyze_layer_2(df_base_compare, fs=fs)
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
