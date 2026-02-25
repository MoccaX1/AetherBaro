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

# --- Layer 3: Atmospheric State ---
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

