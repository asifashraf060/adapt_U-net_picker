import sqlite3
import pickle
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.clients.fdsn.client import Client
from obspy.core.inventory.inventory import Inventory
import seisbench.models as sbm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from tqdm import tqdm
import os

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ S3 CONFIGURATION FOR NCEDC DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════════════════════

# Configure S3 Client for Public NCEDC Access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-west-2')
BUCKET_NAME = 'ncedc-pds'

# Load pretrained PhaseNet for P-wave picks
picker = sbm.PhaseNet.from_pretrained("original")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATABASE CREATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_database(db_path='seismic_data_2.db'):
    """Create SQLite database with required schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create earthquakes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS earthquakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time REAL,
            latitude REAL,
            longitude REAL,
            magnitude REAL,
            depth REAL,
            place TEXT,
            event_id TEXT
        )
    ''')
    
    # Create waveforms table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waveforms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            earthquake_id INTEGER,
            station_code TEXT,
            network TEXT,
            channel TEXT,
            location TEXT,
            waveform_data BLOB,
            sampling_rate REAL,
            p_pick_time REAL,
            eq_time REAL,
            pre_time REAL,
            post_time REAL,
            distance_km REAL,
            FOREIGN KEY (earthquake_id) REFERENCES earthquakes (id)
        )
    ''')
    
    conn.commit()
    return conn

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════════

def process_station(station, start_time, pre_time, post_time, eq_time,
                   network, channel, inventory):
    """Process individual station and return waveform data"""
    station_code = station.code
    
    # Build S3 key
    file_name = f'{station_code}.{network}.{channel}..D.{start_time.year}.{start_time.julday:03d}'
    key = f"continuous_waveforms/{network}/{start_time.year}/{start_time.year}.{start_time.julday:03d}/{file_name}"
    
    try:
        # Stream data from S3
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data_stream = resp['Body']
        buff = BytesIO(data_stream.read())
        buff.seek(0)
        
        # Read waveform
        st_stream = read(buff, format='MSEED')
        st_stream.trim(starttime=eq_time - pre_time, endtime=eq_time + post_time)
        
        if len(st_stream) < 1:
            print(f"  Skipping station {station_code} (no data)")
            return None
        
        # Get single trace
        tr = st_stream[0]
        
        # Remove instrument response
        try:
            tr.remove_response(inventory=inventory, output="DISP", zero_mean=True)
        except Exception as e:
            print(f"  Warning: Could not remove response for {station_code}: {e}")
        
        # Get PhaseNet picks for reference
        picks = picker.classify(st_stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks
        if not picks:
            print(f"  No picks found for station {station_code}")
            return None
        
        # Use first P arrival
        p_time = picks[0].peak_time
        
        # Calculate distance
        station_lat = station.latitude
        station_lon = station.longitude
        
        # Return the processed data
        return {
            'station_code': station_code,
            'waveform': tr.data,
            'sampling_rate': tr.stats.sampling_rate,
            'p_pick_time': p_time.timestamp,
            'eq_time': eq_time.timestamp,
            'pre_time': pre_time,
            'post_time': post_time,
            'station_lat': station_lat,
            'station_lon': station_lon
        }
        
    except Exception as e:
        #print(f"  Error processing station {station_code}: {e}")
        return None

def mine_earthquake_data(eq_time, eq_lat, eq_lon, eq_mag=None, eq_depth=None,
                        radius_km=250, network='NC', channel='HNE', 
                        pre_time=3, post_time=120):
    """Mine seismic data for a single earthquake"""
    
    if not isinstance(eq_time, UTCDateTime):
        eq_time = UTCDateTime(eq_time)
    
    # Define time window
    start_time = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = eq_time.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Get station inventory - try multiple FDSN clients
    print(f"\nProcessing earthquake at {eq_time}")
    print("Getting station inventory...")

    inventory = None
    fdsn_clients = ['IRIS', 'NCEDC', 'SCEDC']

    for client_name in fdsn_clients:
        try:
            print(f"  Trying {client_name} FDSN service...")
            client = Client(client_name)
            inventory = client.get_stations(
                network=network, latitude=eq_lat, longitude=eq_lon,
                starttime=start_time, endtime=end_time, maxradius=radius_km/111.2,
                location='*', channel=channel, level="response"
            )
            print(f"  ✓ Successfully got inventory from {client_name}")
            break
        except Exception as e:
            print(f"  ✗ {client_name} failed: {e}")
            continue

    if not inventory:
        print("  ❌ All FDSN services failed - skipping this earthquake")
        return []
    
    # Get all stations from inventory (no pre-filtering for performance)
    stations = inventory[0].stations if inventory else []
    print(f"Found {len(stations)} stations in inventory")
    
    # Process each station
    waveforms = []
    for station in tqdm(stations, desc="Processing stations"):
        result = process_station(station, start_time, pre_time, post_time, 
                               eq_time, network, channel, inventory)
        if result:
            # Calculate distance from earthquake
            dist_deg = np.sqrt((station.latitude - eq_lat)**2 + 
                              (station.longitude - eq_lon)**2)
            result['distance_km'] = dist_deg * 111.2
            result['network'] = network
            result['channel'] = channel
            result['location'] = ''
            result['eq_lat'] = eq_lat
            result['eq_lon'] = eq_lon
            result['eq_mag'] = eq_mag
            result['eq_depth'] = eq_depth
            waveforms.append(result)
    
    print(f"Successfully processed {len(waveforms)} waveforms")
    return waveforms

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN DATA MINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def main(csv_path='query.csv',
         db_path='seismic_data.db',
         radius_km=300,
         network='NC',
         channel='HNE',
         pre_time=3,
         post_time=120):
    """
    Main data mining function

    Parameters:
    -----------
    csv_path : str
        Path to earthquake catalog CSV file
    db_path : str
        Path to output SQLite database
    radius_km : float
        Search radius in kilometers for stations (applied to all earthquakes)
    network : str
        Network code to use
    channel : str
        Channel code to use
    pre_time : float
        Seconds before earthquake to include
    post_time : float
        Seconds after earthquake to include
    """

    print("="*80)
    print("SEISMIC DATA MINING FOR ML PIPELINE")
    print("="*80)

    # Create database
    print(f"\nCreating database: {db_path}")
    conn = create_database(db_path)
    cursor = conn.cursor()

    # Load earthquakes from CSV
    print(f"Loading earthquakes from {csv_path}")
    eq_df = pd.read_csv(csv_path)
    print(f"Found {len(eq_df)} earthquakes in catalog")

    # Process each earthquake
    total_waveforms = 0

    for idx, eq in eq_df.iterrows():
        print(f"\n{'='*60}")
        print(f"Processing earthquake {idx+1}/{len(eq_df)}")
        print(f"Location: {eq.get('place', 'Unknown')}")
        print(f"Time: {eq['time']}")
        print(f"Magnitude: {eq.get('mag', 'Unknown')}")
        print(f"{'-'*20}")

        try:
            # Mine waveform data
            waveforms = mine_earthquake_data(
                eq_time=eq['time'],
                eq_lat=eq['latitude'],
                eq_lon=eq['longitude'],
                eq_mag=eq.get('mag'),
                eq_depth=eq.get('depth'),
                radius_km=radius_km,
                network=network,
                channel=channel,
                pre_time=pre_time,
                post_time=post_time
            )

            # Insert earthquake record if we have waveforms
            if waveforms:
                eq_time = UTCDateTime(eq['time'])
                cursor.execute('''
                    INSERT INTO earthquakes (event_time, latitude, longitude, magnitude, depth, place, event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (eq_time.timestamp, eq['latitude'], eq['longitude'],
                      eq.get('mag'), eq.get('depth'), eq.get('place', ''), eq.get('id', '')))

                earthquake_id = cursor.lastrowid

                # Insert waveforms into database
                for wf in waveforms:
                    # Serialize waveform data
                    waveform_blob = pickle.dumps(wf['waveform'])

                    cursor.execute('''
                        INSERT INTO waveforms (
                            earthquake_id, station_code, network, channel, location,
                            waveform_data, sampling_rate, p_pick_time, eq_time,
                            pre_time, post_time, distance_km
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        earthquake_id,
                        wf['station_code'],
                        wf['network'],
                        wf['channel'],
                        wf['location'],
                        waveform_blob,
                        wf['sampling_rate'],
                        wf['p_pick_time'],
                        wf['eq_time'],
                        wf['pre_time'],
                        wf['post_time'],
                        wf['distance_km']
                    ))

                conn.commit()
                total_waveforms += len(waveforms)
                print(f"✅ Added {len(waveforms)} waveforms to database")
            else:
                print(f"❌ No waveforms found for this earthquake")

        except Exception as e:
            print(f"❌ Error processing earthquake: {e}")
            conn.rollback()

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATA MINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total earthquakes processed: {len(eq_df)}")
    print(f"Total waveforms in database: {total_waveforms}")

    # Verify database contents
    cursor.execute("SELECT COUNT(*) FROM earthquakes")
    eq_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM waveforms")
    wf_count = cursor.fetchone()[0]

    print(f"\nDatabase verification:")
    print(f"  Earthquakes: {eq_count}")
    print(f"  Waveforms: {wf_count}")

    conn.close()
    print(f"\n✅ Database saved to: {db_path}")

if __name__ == "__main__":
    main(
        csv_path='query.csv',
        db_path='seismic_data_s3_2.db',
        radius_km=50,  # Search radius for stations
        network='NC',   # Network code
        channel='HNZ',  # Channel code
        pre_time=3,     # Seconds before earthquake
        post_time=120   # Seconds after earthquake
    )