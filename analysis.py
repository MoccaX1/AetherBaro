import os
import itertools
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import medfilt, butter, filtfilt, sosfiltfilt, welch, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
from astral import LocationInfo
from astral.sun import elevation
from astral.moon import phase

# --- Preprocessing & Data Loading ---

def parse_system_time(time_str):
    # Example format: "2026-02-20 21:56:34.346 UTC+07:00"
    # We parse the part before UTC
    try:
        clean_str = time_str.split(" UTC")[0].strip()
        return datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        print(f"Error parsing time: {e}")
        return None

def load_and_preprocess_data(folder_path, target_fs=1.0):
    """
    Loads raw CSV data, aligns with real-world datetime,
    applies median filter, interpolates gaps, and generates 32Hz & target resolution dataframes.
    """
    raw_csv = os.path.join(folder_path, "Raw Data.csv")
    time_csv = os.path.join(folder_path, "meta", "time.csv")
    
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing Raw Data.csv in {folder_path}")
        
    df = pd.read_csv(raw_csv)
    
    # Process Datetime
    start_time = None
    if os.path.exists(time_csv):
        df_time = pd.read_csv(time_csv)
        start_row = df_time[df_time['event'] == 'START']
        if not start_row.empty:
            start_time_text = start_row['system time text'].values[0]
            start_time = parse_system_time(start_time_text)
            
    if start_time is None:
        # Fallback to current time if time.csv is corrupted/missing
        start_time = datetime.now()
        
    # Convert Time (s) to absolute Datetime
    # Pandas timedelta accepts seconds via unit='s'
    df['Datetime'] = start_time + pd.to_timedelta(df['Time (s)'], unit='s')
    
    # Ensure sorted by Datetime
    df = df.sort_values('Datetime')
    
    # 1. Median Filter (Spike Removal for Goertek sensor)
    # Apply a small window (e.g., 5 samples) median filter
    df['Pressure (hPa)'] = medfilt(df['Pressure (hPa)'].values, kernel_size=5)
    
    # 2. Gap Filling (Interpolation) by creating a uniform 32Hz grid
    # 32Hz -> ~0.03125 seconds per sample. We'll resample to exact uniform grid
    # to avoid any missing data gaps.
    df = df.set_index('Datetime')
    df = df[~df.index.duplicated(keep='first')] # Remove duplicate timestamps if any
    
    # Create uniform index at 32Hz (using 31.25 ms. Pandas needs exact ms string)
    # 1000/32 = 31.25ms. Let's use 31250 us (microseconds)
    uniform_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='31250us')
    
    # Combine original index with uniform index to preserve data points for interpolation
    combined_index = df.index.union(uniform_index)
    df_combined = df.reindex(combined_index)
    
    # Interpolate values
    df_combined['Pressure (hPa)'] = df_combined['Pressure (hPa)'].interpolate(method='time')
    
    # Extract only the uniform grid points
    df_32hz = df_combined.reindex(uniform_index)
    
    # Recreate Time (s) for convenience
    df_32hz['Time (s)'] = (df_32hz.index - df_32hz.index[0]).total_seconds()
    
    # Reset index for easier plotting later
    df_32hz = df_32hz.reset_index().rename(columns={'index': 'Datetime'})
    
    # 3. Downsample to target freq
    if target_fs == 32.0:
        df_base = df_32hz.copy()
    else:
        freq_str = f"{int(1000/target_fs)}ms"
        df_base = df_32hz.set_index('Datetime').resample(freq_str).mean().reset_index()
    
    return df_32hz, df_base

# --- Layer 1: Synoptic & Tidal ---
def get_lunar_phase_name(phase_days):
    """
    Classifies moon phase into 9 distinct Vietnamese types based on a 29.53-day cycle.
    """
    if phase_days < 0 or phase_days > 29.53:
        return "Không xác định"
    
    # 29.53 days is roughly one synodic month
    # Normalize to 0-360 degrees for easier thresholding based on average times
    deg = (phase_days / 29.53) * 360
    
    if deg < 22.5: # 0 - 22.5
        return "Trăng mới (Sóc)" # 1
    elif deg < 67.5: # 22.5 - 67.5
        return "Trăng non (Lưỡi liềm đầu tháng)" # 2
    elif deg < 112.5: # 67.5 - 112.5
        return "Trăng thượng huyền (Bán nguyệt đầu tháng)" # 3
    elif deg < 157.5: # 112.5 - 157.5
        return "Trăng khuyết đầu tháng (Trương huyền tròn dần)" # 4
    elif deg < 202.5: # 157.5 - 202.5
        return "Trăng tròn (Vọng/Rằm)" # 5
    elif deg < 247.5: # 202.5 - 247.5
        return "Trăng khuyết cuối tháng (Trương huyền khuyết dần)" # 6
    elif deg < 292.5: # 247.5 - 292.5
        return "Trăng hạ huyền (Bán nguyệt cuối tháng)" # 7
    elif deg < 337.5: # 292.5 - 337.5
        return "Trăng tàn (Lưỡi liềm cuối tháng)" # 8
    else: # 337.5 - 360
        return "Trăng tối (Không trăng)" # 9

def analyze_layer_1(df_base, fs=1.0, location_data=None):
    df_res = df_base.copy()
    
    # Default coordinates (HCMC) if no data provided
    lat = 10.7626
    lon = 106.6601
    tz = "Asia/Ho_Chi_Minh"
    city = "Ho Chi Minh City"
    region = "Vietnam"
    
    if location_data:
        lat = location_data.get('Latitude', lat)
        lon = location_data.get('Longitude', lon)
        tz = location_data.get('Timezone', tz)
        city = location_data.get('City', city)
        region = location_data.get('Region', region)

    # 1. We use Gaussian Filter for mathematically smooth low-pass filtering and derivatives.
    pressure_data = df_res['Pressure (hPa)'].interpolate(method='linear').bfill().ffill().values
    
    sigma_10m = 600 * fs
    df_res['Smoothed (1h)'] = gaussian_filter1d(pressure_data, sigma=sigma_10m)
    
    # Use a shorter window (e.g. 1 minutes) for dP/dt so it doesn't get completely flattened
    sigma_1m = 60 * fs
    df_res['dP/dt (hPa/hr)'] = gaussian_filter1d(pressure_data, sigma=sigma_1m, order=1) * (3600.0 * fs)
    df_res['Raw dP/dt (hPa/hr)'] = np.gradient(pressure_data) * (3600.0 * fs)
    
    sigma_30m = 1800 * fs
    df_res['Smoothed (3h)'] = gaussian_filter1d(pressure_data, sigma=sigma_30m)
    
    # Linear trend over entire period
    x = np.arange(len(df_res))
    slope, intercept = np.polyfit(x, pressure_data, 1)
    df_res['Linear Trend'] = slope * x + intercept
    
    # 2. Astronomical Tides (Solar & Lunar)
    loc = LocationInfo(city, region, tz, lat, lon)
    
    # S1 (diurnal solar), S2 (semidiurnal solar), M2 (semidiurnal lunar)
    # Amplitudes are approximate for tropical latitudes (in hPa)
    amp_s1 = 0.5   
    amp_s2 = 1.2   
    amp_m2 = 0.1   
    
    # To optimize execution time for high-frequency data (e.g. 32Hz = 100k+ points),
    # we compute astronomical events on a sparse timeline (every 5 minutes) and interpolate.
    t_seconds = (df_res['Datetime'] - df_res['Datetime'].iloc[0]).dt.total_seconds().values
    
    sparse_step = 300 # 5 minutes
    if len(t_seconds) > 0:
        if t_seconds[-1] < sparse_step:
            sparse_t = np.array([0, t_seconds[-1]]) if t_seconds[-1] > 0 else np.array([0])
        else:
            sparse_t = np.arange(0, t_seconds[-1] + sparse_step, sparse_step)
            # Ensure the last point is covered
            if sparse_t[-1] < t_seconds[-1]:
                sparse_t = np.append(sparse_t, t_seconds[-1])
        
        sparse_dts = [df_res['Datetime'].iloc[0] + pd.Timedelta(seconds=s) for s in sparse_t]
    else:
        sparse_t = []
        sparse_dts = []
        
    theoretic_sparse = []
    sol_elev_sparse = []
    m_phase_sparse = []
    m_elev_sparse = []
    
    # Initialize Skyfield for high-precision lunar positions
    try:
        from skyfield.api import load, wgs84
        ts = load.timescale()
        eph = load('de421.bsp')
        earth, moon = eph['earth'], eph['moon']
        observer_loc = earth + wgs84.latlon(lat, lon, elevation_m=(10 if np.isnan(location_data.get('Elevation', 10)) else location_data.get('Elevation', 10)))
        use_skyfield = True
    except Exception:
        use_skyfield = False
    
    for dt in sparse_dts:
        # tz-aware datetime required by astral
        dt_aware = dt.tz_localize('Asia/Ho_Chi_Minh') if dt.tzinfo is None else dt
        
        # Solar elevation (degrees) -> controls S1, S2 heating cycle
        sol_elev = elevation(loc.observer, dt_aware)
        sol_elev_sparse.append(sol_elev)
        
        # Moon phase (0-27.9 days)
        m_phase = phase(dt_aware)
        m_phase_sparse.append(m_phase)
        
        # Moon Elevation (degrees) via Skyfield
        if use_skyfield:
            t_sf = ts.from_datetime(dt_aware)
            astrometric = observer_loc.at(t_sf).observe(moon)
            alt, az, distance = astrometric.apparent().altaz()
            m_elev = alt.degrees
        else:
            # Fallback mock if skyfield fails to load (e.g., no internet for de421.bsp)
            m_elev = 0.0
            
        m_elev_sparse.append(m_elev)
        
        # Time variables in days for harmonic formulas
        t_hours = dt_aware.hour + dt_aware.minute / 60.0 + dt_aware.second / 3600.0
        
        # S2 wave peaks roughly at 10 AM and 10 PM local time
        tide_s2 = amp_s2 * np.cos(2 * np.pi * (t_hours - 10) / 12)
        # S1 wave peaks roughly at 5 AM (minimum temperature)
        tide_s1 = amp_s1 * np.cos(2 * np.pi * (t_hours - 5) / 24)
        # S3 wave (terdiurnal): 8h period, peaks at ~6 AM, 2 PM, 10 PM
        amp_s3 = amp_s2 * 0.08  # S3 is typically ~8% of S2 amplitude
        tide_s3 = amp_s3 * np.cos(2 * np.pi * (t_hours - 6) / 8)
        
        # M2 wave is driven by the gravitational pull of the moon.
        # It peaks (high tide) when the moon transits the meridian (highest elevation)
        # and also 12.42 hours later (on the opposite side of the earth).
        zenith_angle_rad = np.radians(90.0 - m_elev)
        tide_m2 = amp_m2 * np.cos(2 * zenith_angle_rad)
        
        # Total Theoretical Tide offset
        theoretic_sparse.append(tide_s1 + tide_s2 + tide_s3 + tide_m2)
        
    # Interpolate back to full high-resolution array
    if len(sparse_t) > 1:
        theoretical_tides = np.interp(t_seconds, sparse_t, theoretic_sparse)
        solar_elevations = np.interp(t_seconds, sparse_t, sol_elev_sparse)
        moon_phases = np.interp(t_seconds, sparse_t, m_phase_sparse)
        moon_elevations = np.interp(t_seconds, sparse_t, m_elev_sparse)
    elif len(sparse_t) == 1:
        theoretical_tides = np.full(len(t_seconds), theoretic_sparse[0])
        solar_elevations = np.full(len(t_seconds), sol_elev_sparse[0])
        moon_phases = np.full(len(t_seconds), m_phase_sparse[0])
        moon_elevations = np.full(len(t_seconds), m_elev_sparse[0])
    # Standardize atmospheric tide vertically to match the intercept of the data
    tide_array = np.array(theoretical_tides)
    mean_pressure = np.mean(pressure_data)
    tide_array = tide_array - np.mean(tide_array) + mean_pressure
    
    df_res['Theoretical Tide (Solar+Lunar)'] = tide_array
    
    # 3. Residual Pressure (Actual - Theoretical Tides)
    df_res['Residual Pressure (Synoptic Only)'] = df_res['Pressure (hPa)'] - df_res['Theoretical Tide (Solar+Lunar)'] + mean_pressure
    
    df_res['Solar Elevation (deg)'] = solar_elevations
    df_res['Moon Phase (days)'] = moon_phases
    df_res['Moon Elevation (deg)'] = moon_elevations
    
    metrics = {
        'Synoptic Trend': 'Rising' if slope > 0 else 'Falling',
        'Max dP/dt': df_res['dP/dt (hPa/hr)'].max(),
        'Min dP/dt': df_res['dP/dt (hPa/hr)'].min(),
        'Avg Moon Phase': np.mean(moon_phases),
        'Lunar Phase Name': get_lunar_phase_name(np.mean(moon_phases))
    }
    
    return df_res, metrics

# --- Signal Processing Helpers ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clip high to 0.99 to avoid critical frequency issue if high is too close to Nyquist
    high = min(high, 0.99)
    # Use Second-Order Sections (SOS) for numerical stability with very low frequencies
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    # Filter the data using sosfiltfilt
    y = sosfiltfilt(sos, data)
    return y

def fft_bandpass_filter(data, lowcut, highcut, fs, pad_factor=4):
    """
    Numerically stable zero-phase bandpass filter using FFT.
    Uses zero-padding to resolve frequencies lower than the fundamental 
    frequency of the raw data length (e.g. 3-hour wave in a 2-hour file).
    """
    n_orig = len(data)
    n_padded = n_orig * pad_factor
    
    # Detrend/Demean before padding to prevent massive step-change artifacts at the edges
    mean_val = np.mean(data)
    data_centered = data - mean_val
    
    # Pad with zeros
    data_padded = np.zeros(n_padded)
    # Put data in the middle to minimize edge effects, or just at the start. 
    # Simply padding at the end is standard for FFT resolution interpolation.
    data_padded[:n_orig] = data_centered
    
    yf = np.fft.rfft(data_padded)
    xf = np.fft.rfftfreq(n_padded, d=1/fs)
    
    # Create ideal bandpass mask
    mask = (xf >= lowcut) & (xf <= highcut)
    
    # Smooth the edges to avoid time-domain ringing (Gibbs phenomenon)
    window = np.zeros_like(xf)
    bw = highcut - lowcut
    # Taper dynamically based on bandwidth, minimum of 1 bin
    taper = max(bw * 0.1, xf[1]) 
    
    for i, f in enumerate(xf):
        if lowcut <= f <= highcut:
            window[i] = 1.0
        elif lowcut - taper < f < lowcut and taper > 0:
            window[i] = (f - (lowcut - taper)) / taper
        elif highcut < f < highcut + taper and taper > 0:
            window[i] = 1.0 - (f - highcut) / taper
            
    yf_filtered = yf * window
    y_filtered_padded = np.fft.irfft(yf_filtered, n=n_padded)
    
    # Extract the original time frame
    y_filtered = y_filtered_padded[:n_orig]
    return y_filtered

def compute_zero_padded_fft(data, fs, pad_factor=4):
    """
    Computes zero-padded FFT to increase spectral resolution and avoid leakage.
    pad_factor=4 means the total length will be 4 times the original length.
    """
    n = len(data)
    n_padded = n * pad_factor
    # Remove mean before FFT to avoid huge DC spike
    data_zero_mean = data - np.mean(data)
    
    # Apply Hann window to reduce leakage
    window = np.hanning(n)
    data_windowed = data_zero_mean * window
    
    yf = np.fft.rfft(data_windowed, n=n_padded)
    xf = np.fft.rfftfreq(n_padded, 1/fs)
    
    # Power Spectrum
    power = np.abs(yf)**2 / n
    return xf, power

# --- Layer 2: Wave Spectrum ---
def analyze_layer_2(df_base, fs=1.0):
    from scipy.signal import find_peaks
    data = df_base['Pressure (hPa)'].values
    
    # 1. Zero-padded FFT First to find true peaks
    freqs, power = compute_zero_padded_fft(data, fs, pad_factor=8)
    valid_idx = freqs > 0
    periods_min = (1 / freqs[valid_idx]) / 60
    power_valid = power[valid_idx]
    
    # 2. Define physical search windows for wave types
    # Duration-aware: only search for waves whose period can be resolved by the recording
    duration_s = (df_base['Datetime'].iloc[-1] - df_base['Datetime'].iloc[0]).total_seconds()
    duration_min = duration_s / 60.0
    
    # A wave needs at least 1 full cycle to be detectable.
    # For reliability, we require at least 0.75 of a period.
    max_detectable_period = duration_min / 0.75
    
    # Full wave catalog: from micro-turbulence to tidal harmonics
    # Format: (min_period, max_period, bandwidth_ratio)
    all_search_windows = {
        # Tidal Harmonics (only for long recordings)
        'S3 Tide':  (420, 540, 0.12),   # S3: 8h = 480m. Need > 6h recording
        'S4 Tide':  (320, 420, 0.12),   # S4: 6h = 360m. Need > 4.5h recording
        # Gravity Wave hierarchy
        'Boss':     (140, 320, 0.12),   # Boss: extended to cover 140-320m gap
        'Mother':   (60, 140, 0.15),    # Mother wave
        'Child':    (30, 60, 0.15),     # Child harmonic
        'Micro':    (10, 30, 0.2)       # Micro thermal turbulence
    }
    
    dynamic_bands = {}
    
    for name, (min_p, max_p, bw_ratio) in all_search_windows.items():
        # Skip bands whose minimum period exceeds what the recording can resolve
        if min_p > max_detectable_period:
            continue
            
        # Clamp the search window to the detectable range
        effective_max_p = min(max_p, max_detectable_period)
        
        # Mask out periods within the specific window
        mask = (periods_min >= min_p) & (periods_min <= effective_max_p)
        
        if np.any(mask):
            # Find the period carrying maximum power for labeling
            best_idx = np.argmax(power_valid[mask])
            true_peak = periods_min[mask][best_idx]
        else:
            true_peak = (min_p + effective_max_p) / 2
            
        # USE THE FULL SEARCH WINDOW as the bandpass filter range
        # This ensures ALL energy within the physical band is captured,
        # not just a narrow slice around the peak.
        low_p = min_p
        high_p = effective_max_p
            
        # Convert period (minutes) to frequencies (Hz) for the filter
        low_f = 1 / (high_p * 60)
        high_f = 1 / (low_p * 60)
        
        dynamic_bands[f"{name} ({true_peak:.1f}m)"] = {
            'freqs': (low_f, high_f),
            'period_range': (low_p, high_p),
            'peak': true_peak,
            'base_name': name.split()[0]  # 'S3', 'S4', 'Boss', etc.
        }
        
    # 2.5 Find ALL significant peaks outside our predefined windows (multi-wildcard)
    from scipy.signal import find_peaks
    power_smoothed = gaussian_filter1d(power_valid, sigma=2)
    peaks, properties = find_peaks(power_smoothed, distance=10, prominence=np.max(power_smoothed) * 0.05)
    peak_periods = periods_min[peaks]
    peak_powers = power_smoothed[peaks]
    
    # Sort by power descending
    sorted_idx = np.argsort(peak_powers)[::-1]
    wildcard_count = 0
    max_wildcards = 3  # Allow up to 3 uncategorized peaks
    
    for idx in sorted_idx:
        if wildcard_count >= max_wildcards:
            break
            
        p = peak_periods[idx]
        
        # Skip if too short or already captured by a named band
        if p < 5:
            continue
            
        is_captured = False
        for info in dynamic_bands.values():
            low_p, high_p = info['period_range']
            if low_p <= p <= high_p:
                is_captured = True
                break
                
        if not is_captured:
            bw_ratio = 0.15
            low_p = p * (1 - bw_ratio)
            high_p = p * (1 + bw_ratio)
            low_f = 1 / (high_p * 60)
            high_f = 1 / (low_p * 60)
            
            dynamic_bands[f"Wildcard Peak ({p:.1f}m)"] = {
                'freqs': (low_f, high_f),
                'period_range': (low_p, high_p),
                'peak': p,
                'base_name': 'Wildcard'
            }
            wildcard_count += 1
            
    # --- DEDUPLICATION LOGIC ---
    # Fix the issue where a massive wave exactly on a boundary (e.g. 30.6m) 
    # triggers both 'Micro' (10-30) and 'Child' (30-60).
    bands_list = []
    for k, v in dynamic_bands.items():
        # Find power of this peak to decide which overlapping band wins
        # (closest index to the peak period)
        idx = np.abs(periods_min - v['peak']).argmin()
        peak_power = power_valid[idx] if idx < len(power_valid) else 0
        bands_list.append((k, v, peak_power))
        
    # Sort by period 
    bands_list.sort(key=lambda x: x[1]['peak'])
    
    dedup_bands = {}
    skip_next = False
    
    for i in range(len(bands_list)):
        if skip_next:
            skip_next = False
            continue
            
        k1, v1, pwr1 = bands_list[i]
        
        if i < len(bands_list) - 1:
            k2, v2, pwr2 = bands_list[i+1]
            
            # If peak periods are within 15% of each other, they are the same physical wave
            # leaking across two adjacent theoretical bandpass filters
            ratio = abs(v1['peak'] - v2['peak']) / max(v1['peak'], 1e-5)
            if ratio < 0.15:
                # Keep the one with higher spectral power (closer to the true center)
                if pwr1 >= pwr2:
                    dedup_bands[k1] = v1
                else:
                    dedup_bands[k2] = v2
                skip_next = True
                continue
                
        dedup_bands[k1] = v1
        
    dynamic_bands = dedup_bands

    # 3. Filter the signals using the newly discovered dynamic bands
    filtered_signals = {}
    for label, info in dynamic_bands.items():
        low_f, high_f = info['freqs']
        # Use FFT bandpass to guarantee numeric stability for ultra-low frequencies
        filtered_signals[label] = fft_bandpass_filter(data, low_f, high_f, fs)
        
    # Extract exact global peak period (dynamic range based on file duration)
    max_search = min(max_detectable_period, 600)  # Cap at 10h to avoid synoptic artifacts
    global_mask = (periods_min >= 10) & (periods_min <= max_search)
    if global_mask.any():
        global_peak_idx = np.argmax(power_valid[global_mask])
        exact_peak_period = periods_min[global_mask][global_peak_idx]
    else:
        exact_peak_period = None
    
    return filtered_signals, freqs, power, periods_min, power_valid, exact_peak_period, dynamic_bands

# --- Layer 2 Extensions: Multi-Method Spectral Analysis ---

def _detrend_linear(data, fs):
    """Common linear detrending for all methods."""
    time_s = np.arange(len(data)) / fs
    poly = np.polyfit(time_s, data, 1)
    return data - np.polyval(poly, time_s)

def _find_spectral_peaks(periods, power, min_period=10, max_period=600, prominence_ratio=0.05):
    """Common peak detection on a spectrum. Returns list of dicts with period & amplitude."""
    from scipy.signal import find_peaks
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return []
    p_masked = periods[mask]
    pw_masked = power[mask]
    
    # Sort to ascending period order — required for find_peaks to work correctly.
    # FFT/PSD/STFT all return arrays in descending period order (low freq first).
    # In descending order, short-period waves sit on a trailing slope and find_peaks
    # cannot detect them as local maxima.
    sort_idx = np.argsort(p_masked)
    p_masked = p_masked[sort_idx]
    pw_masked = pw_masked[sort_idx]
    
    # Convert to log10 scale for peak detection
    # This prevents a dominant low-frequency tidal peak (e.g., S3 at power=7000)
    # from setting an impossibly high prominence threshold that silences all other waves.
    # In log-space, S3 (log10=3.9) and Boss (log10=1.5) have comparable prominence.
    pw_log = np.log10(np.maximum(pw_masked, 1e-20))
    pw_smooth = gaussian_filter1d(pw_log, sigma=2)  # sigma=2 for slightly more smoothing
    
    # prominence_ratio is now relative to the log-space range, not linear max
    log_range = pw_smooth.max() - pw_smooth.min()
    min_prominence = log_range * prominence_ratio
    
    peaks, props = find_peaks(pw_smooth, distance=5, prominence=max(min_prominence, 0.05))
    
    results = []
    for idx in peaks:
        period = p_masked[idx]
        pw_val = pw_masked[idx]  # Use original linear power for amplitude
        # Estimate amplitude: sqrt(2 * PSD * df)
        if len(p_masked) > 1:
            df_val = np.abs(np.median(np.diff(1.0 / (p_masked * 60))))
            amp = np.sqrt(2 * pw_val * df_val)
        else:
            amp = np.sqrt(pw_val)
        results.append({
            'period_min': float(period),
            'power': float(pw_val),
            'amplitude': float(amp)
        })
    
    # Sort by power descending, keep top 8
    results.sort(key=lambda x: x['power'], reverse=True)
    return results[:8]


def detect_waves_fft(df_base, fs=1.0):
    """
    Method 1: Zero-padded FFT.
    Returns: dict with 'waves' (list of detected peaks), 'periods', 'power' (for plotting).
    """
    data = df_base['Pressure (hPa)'].values
    freqs, power = compute_zero_padded_fft(data, fs, pad_factor=8)
    valid = freqs > 0
    periods_min = (1.0 / freqs[valid]) / 60.0
    power_valid = power[valid]
    
    duration_s = len(data) / fs
    max_period = min((duration_s / 60.0) / 0.75, 600)
    
    waves = _find_spectral_peaks(periods_min, power_valid,
                                  min_period=10, max_period=max_period,
                                  prominence_ratio=0.03)
    
    return {
        'method': 'FFT',
        'waves': waves,
        'periods': periods_min,
        'power': power_valid,
        'ylabel': 'FFT Power'
    }


def detect_waves_psd(df_base, fs=1.0):
    """
    Method 2: PSD via zero-padded periodogram with log-space smoothing.
    Uses the full N-length window (no Welch averaging) so ALL wave periods are resolved.
    Noise rejection is achieved by stronger Gaussian smoothing in log-space.
    """
    from scipy.signal import periodogram
    data = df_base['Pressure (hPa)'].values
    p_detrend = _detrend_linear(data, fs)
    
    duration_s = len(data) / fs
    max_period = min((duration_s / 60.0) / 0.75, 600)
    
    # Zero-padded periodogram: gives dense frequency grid, same approach as the FFT method
    # but normalized as PSD (power / Hz). Zero-pad 4x for sub-bin resolution.
    nfft = len(p_detrend) * 4
    f_psd, psd_vals = periodogram(p_detrend, fs, nfft=nfft, window='hann')
    
    valid = f_psd > 0
    periods_min = (1.0 / f_psd[valid]) / 60.0
    psd_valid = psd_vals[valid]
    
    # Sort ascending period for _find_spectral_peaks
    sort_idx = np.argsort(periods_min)
    periods_min = periods_min[sort_idx]
    psd_valid = psd_valid[sort_idx]
    
    waves = _find_spectral_peaks(periods_min, psd_valid,
                                  min_period=10, max_period=max_period,
                                  prominence_ratio=0.03)
    
    return {
        'method': 'PSD Welch',
        'waves': waves,
        'periods': periods_min,
        'power': psd_valid,
        'ylabel': 'Power Density (hPa\u00b2/Hz)'
    }


def detect_waves_stft(df_base, fs=1.0):
    """
    Method 3: STFT Spectrogram.
    Detects waves by averaging power across time bins — persistent energy = real wave.
    Returns spectrogram data + detected waves.
    """
    from scipy.signal import spectrogram
    data = df_base['Pressure (hPa)'].values
    p_detrend = _detrend_linear(data, fs)
    
    duration_s = len(data) / fs
    N = len(p_detrend)
    max_period = min((duration_s / 60.0) / 0.75, 600)
    
    # Heatmap: short window for good time resolution (4096 samples = ~68min at 1Hz)
    nperseg_hm = min(4096, N)
    noverlap_hm = nperseg_hm * 3 // 4
    f_hm, t_hm, Sxx_hm = spectrogram(p_detrend, fs, nperseg=nperseg_hm, noverlap=noverlap_hm)
    valid_hm = f_hm > 0
    f_valid = f_hm[valid_hm]
    Sxx_valid_hm = Sxx_hm[valid_hm, :]
    periods_min = (1.0 / f_valid) / 60.0
    time_hours = t_hm / 3600.0
    power_db = 10 * np.log10(np.maximum(Sxx_valid_hm, 1e-20))
    
    # Wave detection: long window + zero-padding for full-range dense spectrum
    # Use N//2 window (max period resolution) + nfft=N*4 (dense bins)
    nperseg_det = min(N // 2, N)
    nfft_det = N * 4
    noverlap_det = nperseg_det * 3 // 4
    f_det, t_det, Sxx_det = spectrogram(p_detrend, fs, nperseg=nperseg_det,
                                         noverlap=noverlap_det, nfft=nfft_det)
    valid_det = f_det > 0
    p_det = (1.0 / f_det[valid_det]) / 60.0
    mean_det = np.mean(Sxx_det[valid_det, :], axis=1)
    
    # Sort ascending
    sort_idx = np.argsort(p_det)
    p_det = p_det[sort_idx]
    mean_det = mean_det[sort_idx]
    
    waves = _find_spectral_peaks(p_det, mean_det,
                                  min_period=10, max_period=max_period,
                                  prominence_ratio=0.03)
    
    return {
        'method': 'STFT',
        'waves': waves,
        'periods': p_det,
        'power': mean_det,
        'ylabel': 'Mean Power (time-averaged)',
        'spectrogram': {
            'time_hours': time_hours,
            'periods_min': periods_min,
            'power_db': power_db
        }
    }


def detect_waves_cwt(df_base, fs=1.0):
    """
    Method 4: Continuous Wavelet Transform (Morlet).
    """
    import pywt
    data = df_base['Pressure (hPa)'].values
    p_detrend = _detrend_linear(data, fs)
    
    duration_s = len(data) / fs
    max_period = min((duration_s / 60.0) / 0.75, 600)
    
    # Define scales corresponding to periods from 10m to max_period
    # Morlet wavelet: period = (1/f) and scale = fs / (2*pi*f * center_freq)
    # For Morlet (cmor1.5-1.0), center_freq ~ 1.0
    min_period_s = 10 * 60   # 10 minutes in seconds
    max_period_s = max_period * 60
    
    # Create 80 logarithmically spaced scales (optimized for speed vs resolution)
    n_scales = 80
    periods_s = np.logspace(np.log10(min_period_s), np.log10(max_period_s), n_scales)
    
    # Downsample aggressively for CWT speed: max 4000 samples
    # CWT FFT method is O(N log N * scales), so keeping N small is key for Streamlit Cloud
    max_samples = 4000
    if len(p_detrend) > max_samples:
        step = max(1, len(p_detrend) // max_samples)
        p_ds = p_detrend[::step]
        fs_ds = fs / step
    else:
        p_ds = p_detrend
        fs_ds = fs
    
    # Convert periods to scales for the morl wavelet
    central_freq = pywt.central_frequency('morl')
    scales = (central_freq * fs_ds) / (1.0 / periods_s)
    
    # Compute CWT using the much faster FFT method instead of time-domain convolution
    # This prevents the Streamlit Cloud timeout issue
    coefficients, frequencies = pywt.cwt(p_ds, scales, 'morl', sampling_period=1.0/fs_ds, method='fft')
    
    # Power = |coefficients|^2
    cwt_power = np.abs(coefficients) ** 2
    
    periods_min = periods_s / 60.0
    time_s_arr = np.arange(len(p_ds)) / fs_ds
    time_hours = time_s_arr / 3600.0
    
    # Time-averaged power for peak detection
    mean_power = np.mean(cwt_power, axis=1)
    
    waves = _find_spectral_peaks(periods_min, mean_power,
                                  min_period=10, max_period=max_period,
                                  prominence_ratio=0.05)
    
    # Scalogram in dB for plotting
    cwt_db = 10 * np.log10(np.maximum(cwt_power, 1e-20))
    
    return {
        'method': 'CWT',
        'waves': waves,
        'periods': periods_min,
        'power': mean_power,
        'ylabel': 'CWT Mean Power',
        'scalogram': {
            'time_hours': time_hours,
            'periods_min': periods_min,
            'power_db': cwt_db
        }
    }


def detect_waves_hht(df_base, fs=1.0):
    """
    Method 5: Hilbert-Huang Transform (EMD + Hilbert).
    Decomposes signal into Intrinsic Mode Functions (IMFs), then computes
    instantaneous frequency and amplitude of each.
    """
    from PyEMD import EMD
    from scipy.signal import hilbert
    
    data = df_base['Pressure (hPa)'].values
    p_detrend = _detrend_linear(data, fs)
    
    duration_s = len(data) / fs
    max_period = min((duration_s / 60.0) / 0.75, 600)
    
    # Downsample for EMD speed
    max_samples = 5000
    if len(p_detrend) > max_samples:
        step = len(p_detrend) // max_samples
        p_ds = p_detrend[::step]
        fs_ds = fs / step
    else:
        p_ds = p_detrend
        fs_ds = fs
    
    # EMD decomposition
    emd = EMD()
    emd.MAX_ITERATION = 100
    try:
        IMFs = emd.emd(p_ds)
    except Exception:
        # EMD can fail on very short or flat data
        return {
            'method': 'HHT/EMD',
            'waves': [],
            'imfs': [],
            'ylabel': 'IMF Amplitude'
        }
    
    # Analyze each IMF
    imf_results = []
    waves = []
    time_hours = np.arange(len(p_ds)) / fs_ds / 3600.0
    
    for i, imf in enumerate(IMFs):
        # Skip the residual (last IMF) if it's just a trend
        if i == len(IMFs) - 1 and len(IMFs) > 2:
            continue
            
        # Hilbert transform for instantaneous frequency
        analytic = hilbert(imf)
        inst_amp = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        
        # Instantaneous frequency (Hz)
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi * (1.0 / fs_ds))
        inst_freq = np.clip(inst_freq, 1e-10, fs_ds / 2)
        
        # Dominant period of this IMF (median of instantaneous period)
        inst_period_min = (1.0 / inst_freq) / 60.0
        median_period = float(np.median(inst_period_min))
        mean_amp = float(np.mean(inst_amp))
        
        imf_results.append({
            'imf_index': i,
            'signal': imf,
            'time_hours': time_hours,
            'median_period_min': median_period,
            'mean_amplitude': mean_amp
        })
        
        # Only count as a detected wave if within our range
        if 10 <= median_period <= max_period:
            waves.append({
                'period_min': median_period,
                'amplitude': mean_amp,
                'power': mean_amp ** 2,
                'imf_index': i
            })
    
    waves.sort(key=lambda x: x['power'], reverse=True)
    
    return {
        'method': 'HHT/EMD',
        'waves': waves[:8],
        'imfs': imf_results,
        'ylabel': 'IMF Amplitude'
    }


def compute_wave_consensus(method_results, tolerance=0.20):
    """
    Multi-method consensus voting.
    Groups detected waves across methods by matching periods within ±tolerance.
    Returns a list of consensus entries sorted by agreement count.
    """
    # Collect all detected periods from all methods
    all_detections = []
    for result in method_results:
        method = result['method']
        for w in result['waves']:
            all_detections.append({
                'method': method,
                'period': w['period_min'],
                'amplitude': w.get('amplitude', 0),
                'power': w.get('power', 0)
            })
    
    if not all_detections:
        return []
    
    # Sort by period
    all_detections.sort(key=lambda x: x['period'])
    
    # Group detections within ±tolerance of each other
    groups = []
    used = set()
    
    for i, det in enumerate(all_detections):
        if i in used:
            continue
        group = [det]
        used.add(i)
        
        for j in range(i + 1, len(all_detections)):
            if j in used:
                continue
            # Check if within tolerance AND from a different method
            ratio = abs(det['period'] - all_detections[j]['period']) / det['period']
            methods_in_group = {d['method'] for d in group}
            if ratio <= tolerance and all_detections[j]['method'] not in methods_in_group:
                group.append(all_detections[j])
                used.add(j)
        
        groups.append(group)
    
    # Build consensus entries
    consensus = []
    for group in groups:
        methods = [d['method'] for d in group]
        periods = [d['period'] for d in group]
        amplitudes = [d['amplitude'] for d in group]
        
        # --- Smart Evidence-Based Scoring System ---
        # Instead of just counting methods, we evaluate the evidence quality:
        # 1. Method Suitability for the specific period band
        # 2. Amplitude vs Hardware Noise Floor (SNR)
        
        n_methods = len(methods)
        avg_period = float(np.median(periods)) # Use median to be robust against outliers
        
        # Spectral methods (FFT, PSD, CWT) inherently return amplitudes that are mathematically 
        # scaled down by windowing, energy spreading, or averaging (often 10x-30x lower than 
        # the true time-domain amplitude). We apply a heuristic calibration factor to bring 
        # the 'average amplitude' back to a physical scale before comparing to hardware specs.
        # HHT is already in physical scale, but the mix pulls the average down.
        # A conservative 15x multiplier aligns the spectral amplitude (e.g., ~0.01 hPa) 
        # with the empirical time-domain amplitude (e.g., ~0.15 hPa).
        calibrated_amp = float(np.mean(amplitudes)) * 15.0
        
        # Determine hardware noise floor based on period band
        # Hardware specs: >1Hz: 0.0056, 1s-1m: 0.0072, >160m: 0.1656
        if avg_period < 1.0: # < 1 min (Turbulence)
            noise_floor = 0.0072
        elif avg_period > 160.0: # > 160 min (VLF Drift)
            noise_floor = 0.1656
        else: # 1 min - 160 min (Gravity Waves band)
            # Log-linear interpolation between 0.0072 (1m) and 0.1656 (160m)
            log_p = np.log10(avg_period)
            log_1 = np.log10(1.0)
            log_160 = np.log10(160.0)
            fraction = (log_p - log_1) / (log_160 - log_1)
            noise_floor = 0.0072 + fraction * (0.1656 - 0.0072)
            
        snr_calibrated = calibrated_amp / noise_floor if noise_floor > 0 else 0
        
        # Calculate Base Score (Max 100) from methods
        score = 0
        for m in methods:
            if m in ['FFT', 'PSD Welch']:
                # Strong evidence for short/medium waves, weaker for long trends
                score += 35 if avg_period < 100 else 15
            elif m == 'STFT':
                # Strong evidence for persistent waves
                score += 25
            elif m == 'CWT':
                # Strong evidence for long waves, okay for short
                score += 30 if avg_period >= 60 else 15
            elif m == 'HHT/EMD':
                # Strongest evidence for distinct physical nonlinear modes
                score += 40
                
        # Apply SNR Multiplier
        # SNR < 1.0: Penalize heavily (likely noise artifact)
        # SNR 1.0 - 2.0: Neutral to slight boost
        # SNR > 2.0: Strong physical evidence, boost score
        if snr_calibrated < 1.0:
            score *= (0.6 + 0.4 * snr_calibrated) # Drops to 60% if SNR is 0
        elif snr_calibrated > 2.0:
            score *= min(1.5, 1.0 + 0.15 * (snr_calibrated - 2.0)) # Up to 50% bonus
            
        score = min(100.0, score) # Cap at 100
        
        # Assign Confidence Tier
        if score >= 80:
            confidence = 'Confirmed'
            icon = '🟢'
        elif score >= 60:
            confidence = 'Likely'
            icon = '🟡'
        elif score >= 40:
            confidence = 'Weak'
            icon = '🟠'
        else:
            confidence = 'Uncertain'
            icon = '⚪'
        
        consensus.append({
            'period_min': avg_period,
            'amplitude': float(np.mean(amplitudes)), # Keep raw mean for display
            'snr': snr_calibrated,
            'score': score,
            'n_methods': n_methods,
            'methods': methods,
            'confidence': confidence,
            'icon': icon
        })
    
    # Sort by number of methods confirming (desc), then by period (asc)
    consensus.sort(key=lambda x: (-x['n_methods'], x['period_min']))
    
    return consensus


def permutation_entropy(time_series, m=3, delay=1):
    """
    Calculate the Permutation Entropy
    m: embedding dimension
    delay: time delay
    """
    n = len(time_series)
    if n < m:
        return np.nan
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)
    
    for i in range(n - delay * (m - 1)):
        # Extract the ordinal pattern
        sorted_index_array = np.argsort(time_series[i:i+delay*m:delay], kind='quicksort')
        for j, p in enumerate(permutations):
            if np.array_equal(sorted_index_array, p):
                c[j] += 1
                break
    
    c = [val for val in c if val != 0]
    p = np.array(c) / sum(c)
    pe = -sum(p * np.log2(p))
    # Normalize by log2(m!) so it's between 0 and 1
    return pe / np.log2(math.factorial(m))

def analyze_layer_3(df_base, fs=1.0):
    # Rolling Permutation Entropy (10 min windows = 600 seconds) to detect "quiet" vs "turbulent" states
    df_res = df_base[['Datetime', 'Pressure (hPa)']].copy()
    
    # We downsample a bit before Rolling PE to save compute time (every 10s)
    step = int(max(1, 10 * fs))
    df_10s = df_res.iloc[::step].copy()
    pe_values = []
    
    # PE is slow, so only compute every N points
    # 60 samples represents 10 minutes when step is 10s
    pe_window = 60
    
    for i in range(len(df_10s)):
        if i < pe_window:
            pe_values.append(np.nan)
        else:
            ts = df_10s['Pressure (hPa)'].iloc[i-pe_window:i].values
            pe_values.append(permutation_entropy(ts, m=3, delay=1))
            
    df_10s['Permutation Entropy'] = pe_values
    df_10s['Rolling Variance (10m)'] = df_10s['Pressure (hPa)'].rolling(pe_window).var()
    
    # Compute global spectral slope
    nperseg = int(min(1024 * fs, len(df_base)))
    freqs, power = welch(df_base['Pressure (hPa)'].values, fs=fs, nperseg=nperseg)
    # Fit log-log slope for mid frequencies
    valid = (freqs > 0.01) & (freqs < 0.1)
    if np.sum(valid) > 2:
        slope, _ = np.polyfit(np.log10(freqs[valid]), np.log10(power[valid]), 1)
    else:
        slope = np.nan
        
    metrics = {
        'Global Spectral Slope': slope,
        'Max Entropy': np.nanmax(pe_values) if len(pe_values) > 0 else np.nan,
        'Min Entropy': np.nanmin(pe_values) if len(pe_values) > 0 else np.nan
    }
    
    return df_10s, metrics

# --- Layer 4: Micro-events (requires 32Hz data) ---
def analyze_layer_4(df_32hz):
    # High-pass filter above 1Hz to catch gusts and turbulence (micro-events)
    fs = 32.0
    nyq = 0.5 * fs
    low = 1.0 / nyq
    b, a = butter(4, low, btype='high')
    
    # Fill any remaining NaNs to prevent filter crash
    data = df_32hz['Pressure (hPa)'].ffill().bfill().values
    highpass = filtfilt(b, a, data)
    
    df_res = df_32hz[['Datetime', 'Pressure (hPa)']].copy()
    df_res['High Frequency (>1Hz) Noise'] = highpass
    
    # Calculate rolling std as a proxy for wind gust intensity (1 second window = 32 points)
    df_res['Gust Proxy (Rolling Std)'] = df_res['High Frequency (>1Hz) Noise'].rolling(32).std()
    
    metrics = {
        'Max Gust Proxy': df_res['Gust Proxy (Rolling Std)'].max(),
        'Avg Gust Proxy': df_res['Gust Proxy (Rolling Std)'].mean(),
        'Pressure Skewness': df_32hz['Pressure (hPa)'].skew()
    }
    return df_res, metrics

# --- Layer 5: Planetary Link ---
def analyze_layer_5(df_l2_current, df_l2_baseline=None, external_mslp=None):
    # Analyze amplitudes of Boss wave
    # Find the dynamic Boss column name (e.g. 'Boss (162.3m)')
    boss_col_current = next((col for col in df_l2_current.columns if col.startswith('Boss')), None)
    
    metrics = {}
    if boss_col_current is not None:
        boss_current = df_l2_current[boss_col_current].max() - df_l2_current[boss_col_current].min()
        metrics['Boss Wave Amplitude (Current)'] = boss_current
    else:
        boss_current = np.nan
        metrics['Boss Wave Amplitude (Current)'] = np.nan
        
    if df_l2_baseline is not None:
        boss_col_base = next((col for col in df_l2_baseline.columns if col.startswith('Boss')), None)
        if boss_col_base is not None:
            boss_base = df_l2_baseline[boss_col_base].max() - df_l2_baseline[boss_col_base].min()
            metrics['Boss Wave Amplitude (Baseline)'] = boss_base
            if not np.isnan(boss_current) and boss_base > 0:
                metrics['Boss Amplitude Ratio'] = boss_current / boss_base
            else:
                metrics['Boss Amplitude Ratio'] = np.nan
        
    if external_mslp is not None:
        # Just simple comparison metric
        metrics['Current MSLP Ref'] = external_mslp
        
    return metrics

# --- Layer 6: Device Evaluation ---
def analyze_device_performance(df_32hz, device_info):
    """
    Evaluates hardware performance based on metadata and raw 32Hz data.
    """
    metrics = {}
    
    # 1. Missing Data / Continuity
    time_diffs = df_32hz['Datetime'].diff().dt.total_seconds()
    max_gap = time_diffs.max()
    expected_gap = 1.0 / 32.0
    metrics['Max Time Gap (s)'] = max_gap
    # Consider gap > 2 * expected as missing data
    metrics['Data Missing Ratio (%)'] = np.sum(time_diffs > (expected_gap * 2)) / len(time_diffs) * 100
    
    # 2. Spectral Decomposition of Residuals (Welch PSD)
    hardware_tolerance = device_info.get('Resolution', 0.01)
    
    time_s = df_32hz['Time (s)'].values
    data = df_32hz['Pressure (hPa)'].ffill().bfill().values
    fs = 32.0
    
    # Detrend to remove the linear Synoptic trend (Rising/Falling pressure over hours)
    # Order 1 (Linear): Removes the gross atmospheric trend slope without destroying
    # the VLF sensor drift curvature that polyfit(2) would erase.
    # Mean subtraction is too weak (leaves 1+ hPa synoptic change in VLF band).
    # Polyfit(2) is too strong (erases real sensor drift completely, giving 0.0).
    # Linear detrend is the Goldilocks zone.
    poly = np.polyfit(time_s, data, 1)
    p_detrend = data - np.polyval(poly, time_s)
    
    # Use Welch method for Power Spectral Density
    from scipy.signal import welch
    # VLF Drift requires resolving frequencies < 1/(60*160) = 0.000104 Hz
    # This requires a window of at least 160 minutes.
    # We will use the full length of the data to ensure maximum low-frequency resolution.
    nperseg = len(p_detrend) 
        
    f, psd = welch(p_detrend, fs, nperseg=nperseg)
    
    def get_band_rms(f_arr, psd_arr, f_low, f_high):
        mask = (f_arr >= f_low) & (f_arr < f_high)
        if not np.any(mask): return 0.0
        variance = np.trapezoid(psd_arr[mask], f_arr[mask])
        return np.sqrt(variance) if variance > 0 else 0.0
        
    # Band definitions based on physics
    rms_white = get_band_rms(f, psd, 1.0, fs/2)            # White Noise (Sensor Electrics): > 1Hz
    rms_turb = get_band_rms(f, psd, 1/60, 1.0)             # Turbulence (Local wind): 1s to 1 min (0.016Hz - 1Hz)
    rms_waves = get_band_rms(f, psd, 1/(60*160), 1/60)     # Mesoscale Waves: 1 min to 160 min (0.0001Hz - 0.016Hz)
    rms_vlf = get_band_rms(f, psd, 0.0, 1/(60*160))        # VLF Drift (Sensor Wander): > 160 min (< 0.0001Hz)
    
    metrics['Empirical White Noise Std (hPa)'] = rms_white
    metrics['Empirical Turbulence RMS (hPa)'] = rms_turb
    metrics['Empirical Waves RMS (hPa)'] = rms_waves
    metrics['Empirical Pink Noise RMS (hPa)'] = rms_vlf # Kept name for UI backward compatibility, but it's now pure VLF drift
    metrics['Total Residual RMS (hPa)'] = np.std(p_detrend)
    
    # Provide the raw white noise signal for plotting
    nyq = 0.5 * fs
    b, a = butter(4, 1.0 / nyq, btype='high')
    noise_signal = filtfilt(b, a, data)
    metrics['White Noise Signal'] = noise_signal
    
    # 4. Empirical Resolution (Minimum non-zero difference)
    unique_vals = np.sort(df_32hz['Pressure (hPa)'].unique())
    diffs = np.diff(unique_vals)
    diffs = diffs[diffs > 1e-6] # Avoid floating point tiny diffs
    if len(diffs) > 0:
        empirical_res = np.min(diffs)
    else:
        empirical_res = hardware_tolerance
    
    metrics['Empirical Resolution (hPa)'] = empirical_res
    
    # Reliability score (0-100)
    score = 100
    if metrics['Data Missing Ratio (%)'] > 0.5:
        score -= min(40, metrics['Data Missing Ratio (%)'] * 10)
    if rms_white > hardware_tolerance:
        penalty = ((rms_white - hardware_tolerance) / hardware_tolerance) * 20
        score -= min(40, penalty)
    
    metrics['Reliability Score'] = max(0, score)
    # White Noise Signal already added above
    
    return metrics

def export_features(folder_path, m1, m2, m3, m4, m5):
    """
    Exports summary metrics to Analysis_Result.csv
    """
    all_metrics = {}
    all_metrics.update(m1)
    
    # m2 has dicts
    all_metrics.update(m3)
    all_metrics.update(m4)
    all_metrics.update(m5)
    
    df_export = pd.DataFrame([all_metrics])
    df_export['Analysis Datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    out_path = os.path.join(folder_path, 'Analysis_Result.csv')
    df_export.to_csv(out_path, index=False)
    return out_path

