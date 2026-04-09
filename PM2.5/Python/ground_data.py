import requests
import pandas as pd
import os
import time
from datetime import datetime

# --- CONFIGURATION (Based on your provided files) ---
API_KEY = "Your_API_Key"
SAVE_PATH = r"Your_Path"
HEADERS = {"X-API-Key": API_KEY}
BASE = "https://api.openaq.org/v3"
PM25_PARAM_ID = 2

# Target Study Period
DATETIME_FROM = "2025-01-01T00:00:00Z"
DATETIME_TO   = "2026-03-31T23:59:59Z"
FILE_TAG      = "Nepal_Full_Study_Jan25_Mar26"

def get_all_nepal_locations():
    """
    Fetches all monitoring locations in Nepal. 
    Uses 'iso' parameter to avoid the 422 error caused by 'countries_id'.
    """
    url = f"{BASE}/locations"
    params = {
        "iso": "NP",              # Correct parameter for country code string
        "parameters_id": PM25_PARAM_ID,
        "limit": 500
    }
    
    print(f"Querying OpenAQ for all Nepal stations...")
    resp = requests.get(url, headers=HEADERS, params=params)
    
    if resp.status_code == 200:
        return resp.json().get("results", [])
    else:
        print(f"ERROR {resp.status_code}: {resp.text}")
        return []

def fetch_measurements(sensor_id, station_name):
    """Paginates through measurements for a specific sensor ID."""
    url = f"{BASE}/sensors/{sensor_id}/measurements"
    params = {
        "datetime_from": DATETIME_FROM,
        "datetime_to":   DATETIME_TO,
        "limit":         1000,
        "page":          1,
    }
    
    all_results = []
    while True:
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"      Error fetching sensor {sensor_id}: {resp.status_code}")
            break
            
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
            
        all_results.extend(results)
        
        # Check if we've reached the end of the data
        if len(results) < params["limit"]:
            break
            
        params["page"] += 1
        time.sleep(0.5) # Prevent hitting rate limits during pagination
        
    return all_results

def main():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    print(f"Starting Extraction: {DATETIME_FROM} to {DATETIME_TO}")
    locations = get_all_nepal_locations()
    print(f"Found {len(locations)} stations in Nepal.\n")

    for loc in locations:
        loc_id = loc.get("id")
        loc_name = loc.get("name", f"Station_{loc_id}").replace("/", "_").replace(" ", "_")
        
        # Each location can have multiple sensors (PM2.5, PM10, etc.)
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("id") == PM25_PARAM_ID:
                s_id = sensor["id"]
                print(f"Processing: {loc_name} (Sensor ID: {s_id})")
                
                records = fetch_measurements(s_id, loc_name)
                
                if records:
                    df = pd.json_normalize(records)
                    
                    # Add station metadata to every row for your ML framework
                    df['station_id'] = loc_id
                    df['station_name'] = loc.get("name")
                    df['latitude'] = loc.get("coordinates", {}).get("latitude")
                    df['longitude'] = loc.get("coordinates", {}).get("longitude")
                    
                    filename = f"Nepal_{loc_name}_Sensor{s_id}_{FILE_TAG}.csv"
                    save_file = os.path.join(SAVE_PATH, filename)
                    df.to_csv(save_file, index=False)
                    print(f"    -> Saved {len(df)} measurements to {filename}")
                else:
                    print(f"    -> No data found for specified dates.")
                
                # Stay within 60 requests/min limit
                time.sleep(1.2) 

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()