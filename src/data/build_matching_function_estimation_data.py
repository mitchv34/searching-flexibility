# build_matching_data_bls_unified.py

"""
This script constructs the state-level panel dataset required to estimate the
matching function elasticity (gamma_1).

This version uses a UNIFIED data strategy, fetching ALL required time series
(national and state-level) directly from the BLS API.

It is formatted with `#%%` cells to be "IPython friendly" for interactive
execution in IDEs like VS Code or Spyder.
"""
#%%
# --- 1. Imports and Configuration ---
import pandas as pd
import us
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import json
import requests
import time

print("🔧 Configuring data construction script (Unified BLS API)")

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2025, 12, 31)
OUTPUT_DIR = "/project/high_tech_ind/searching-flexibility/data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "matching_function_data_panel_bls_unified.csv")
API_KEY_FILE = "/project/high_tech_ind/searching-flexibility/api_keys.json"

# State FIPS codes are needed for BLS series IDs
STATE_FIPS_MAP = {s.abbr: s.fips for s in us.states.STATES}
STATE_FIPS_MAP['DC'] = '11'

# --- BLS API Series ID Patterns and Tickers ---
# National Series
NATIONAL_BLS_TICKERS = {
    'U_to_E_flow': 'LNS17100000',         # Unemployed who became Employed (CPS)
    'H_national': 'JTS000000000000000HIL' # JOLTS Total Hires, National
}
# State-Level Series Patterns
BLS_STATE_PATTERNS = {
    'U': 'LASST{}0000000000007', # Unemployment Level (LAUS)
    'V': 'JTS000000{}0000000HIL', # Job Openings (JOLTS)
    'H_total': 'JTS000000{}0000000JOL' # Hires (JOLTS)
}
BLS_API_ENDPOINT = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'


#%%
# --- 2. Helper Functions ---

def load_bls_api_key():
    """Safely loads the BLS API key from a JSON file."""
    if not os.path.exists(API_KEY_FILE):
        print(f"⚠️  API key file not found at '{API_KEY_FILE}'. Public access will be used (rate-limited).")
        return None
    try:
        with open(API_KEY_FILE, 'r') as f:
            keys = json.load(f)
            return keys.get('bls', None)
    except Exception as e:
        print(f"❌ Error loading API key: {e}")
        return None

def fetch_bls_data(series_ids, api_key):
    """Fetches multiple series from the BLS API and returns a clean DataFrame."""
    if not series_ids:
        return pd.DataFrame()
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": series_ids,
        "startyear": str(START_DATE.year),
        "endyear": str(END_DATE.year),
        "registrationkey": api_key,
        "catalog": False # Faster without catalog data
    })
    
    try:
        p = requests.post(BLS_API_ENDPOINT, data=data, headers=headers)
        p.raise_for_status()
        json_data = p.json()
        
        if json_data['status'] != 'REQUEST_SUCCEEDED':
            print(f"❌ BLS API Error: {json_data.get('message', ['No message'])[0]}")
            return None

        data_list = []
        for series in json_data['Results']['series']:
            series_id = series['seriesID']
            for item in series['data']:
                date = pd.to_datetime(f"{item['year']}-{item['period'][1:]}-01")
                value = float(item['value'])
                data_list.append({'series_id': series_id, 'date': date, 'value': value})
        
        if not data_list: return pd.DataFrame()
        df = pd.DataFrame(data_list)
        return df.pivot(index='date', columns='series_id', values='value')
        
    except Exception as e:
        print(f"❌ An error occurred during BLS data fetch: {e}")
        return None

#%%
# --- 3. Main Data Construction Pipeline ---

def calculate_national_share(api_key):
    """Calculates the monthly share of hires from unemployment using BLS API."""
    print("\nStep 1: Fetching national data from BLS API...")
    df = fetch_bls_data(list(NATIONAL_BLS_TICKERS.values()), api_key)
    if df is None: return None
    
    df.rename(columns=lambda x: {v: k for k, v in NATIONAL_BLS_TICKERS.items()}[x], inplace=True)
    df['share_U_to_E'] = df['U_to_E_flow'] / df['H_national']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("✅ National share calculated.")
    return df[['share_U_to_E']]

def build_state_panel(api_key):
    """Builds the full state-level panel (U, V, H) using the BLS API."""
    print("\nStep 2: Building state-level panel from BLS API...")
    
    all_state_series = []
    state_abbr_list = [s.abbr for s in us.states.STATES] + ["DC"]
    
    for state_abbr in state_abbr_list:
        fips = STATE_FIPS_MAP.get(state_abbr)
        if fips:
            for series_type in ['U', 'V', 'H_total']:
                series_id = BLS_STATE_PATTERNS[series_type].format(fips)
                all_state_series.append(series_id)

    # Fetch in chunks to respect API limits (50 series per request)
    chunk_size = 50
    state_panel_wide = pd.DataFrame()
    for i in tqdm(range(0, len(all_state_series), chunk_size), desc="Querying BLS API for state data"):
        chunk = all_state_series[i:i+chunk_size]
        df_chunk = fetch_bls_data(chunk, api_key)
        if df_chunk is not None:
            state_panel_wide = pd.concat([state_panel_wide, df_chunk], axis=1)
        time.sleep(0.5) # Be polite to the API

    if state_panel_wide.empty:
        print("❌ No state-level data could be fetched.")
        return None

    # Reshape the wide data into a long panel format
    state_panel_long = state_panel_wide.stack().reset_index()
    state_panel_long.columns = ['date', 'series_id', 'value']
    
    # Extract state and measure from series_id
    def parse_series_id(sid):
        fips = sid[5:7] if sid.startswith('LASST') else sid[9:11]
        measure_code = sid[-1] if sid.startswith('LASST') else sid[-3:]
        state = next((s.abbr for s in us.states.STATES_AND_TERRITORIES if s.fips == fips), None)
        
        series_type = None
        if measure_code == '7': series_type = 'U'
        elif measure_code == 'JOL': series_type = 'V'
        elif measure_code == 'HIL': series_type = 'H_total'
        
        return state, series_type

    state_panel_long[['state', 'measure']] = state_panel_long['series_id'].apply(lambda x: pd.Series(parse_series_id(x)))
    
    # Pivot to get U, V, and H_total as columns
    final_panel = state_panel_long.pivot_table(index=['date', 'state'], columns='measure', values='value').reset_index()
    
    # Harmonize to get U in thousands of people like the other variables
    # final_panel['U'] = final_panel['U'] * 1000

    print("✅ State-level panel constructed.")
    return final_panel.dropna()

#%%
# --- Cell 4: Main Execution Block ---
print("=" * 60)
print("🚀 Starting Matching Function Data Construction (Unified BLS)")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
api_key = load_bls_api_key()

# --- Pipeline Execution ---
national_share_df = calculate_national_share(api_key)
state_panel_df = build_state_panel(api_key)

#%%
# --- Cell 5: Merge and Calculate Final Variables ---
if national_share_df is not None and state_panel_df is not None:
    print("\nStep 3: Merging national share with state panel...")
    final_df = pd.merge(state_panel_df, national_share_df, left_on='date', right_index=True, how='left')
    print("✅ Merge complete.")

    print("\nStep 4: Calculating final variables (M_it, f_it, theta_it)...")
    final_df['M_it'] = final_df['H_total'] * final_df['share_U_to_E']
    final_df['f_it'] = final_df['M_it'] / final_df['U']
    final_df['theta_it'] = final_df['V'] / final_df['U']
    final_df['ln_f'] = np.log(final_df['f_it'])
    final_df['ln_theta'] = np.log(final_df['theta_it'])
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print("\nStep 4.5: Constraining f_it to be a valid probability [0, 1]...")
    # Count how many observations are affected
    n_before = final_df.shape[0]
    n_over_1 = (final_df['f_it'] > 1.0).sum()

    if n_over_1 > 0:
        print(f"⚠️  Found {n_over_1} observations ({round(n_over_1/n_before*100, 2)}%) where f_it > 1.")
        print("   Capping these values at 0.999 to ensure they are valid probabilities.")
        final_df.loc[final_df['f_it'] > 1.0, 'f_it'] = 0.999
        
        # Recalculate ln_f for the capped values
        final_df['ln_f'] = np.log(final_df['f_it'])
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("✅ Job finding rate constrained successfully.")
    print("✅ Final variables calculated.")

    # --- Step 5: Save and Summarize ---
    print(f"\nStep 5: Saving final dataset to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Dataset saved successfully!")

    print("\n--- Final Data Summary ---")
    print("Shape of the final panel:", final_df.shape)
    print("\nSample of the final data:")
    print(final_df[['state', 'date', 'V', 'U', 'H_total', 'M_it', 'f_it', 'theta_it']].head())
    
    print("\n--- Descriptive Statistics ---")
    print(final_df[['f_it', 'theta_it']].describe())

    print("\n" + "=" * 60)
    print("🎉 Data construction complete!")
    print("=" * 60)
else:
    print("❌ Data construction failed due to errors in fetching data.")
# %%
