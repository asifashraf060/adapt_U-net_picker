import sqlite3
import pickle
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.core.inventory.inventory import Inventory
import seisbench.models as sbm
from tqdm import tqdm
from typing import Optional
import warnings

# Suppress the specific warning about multiple matching responses
warnings.filterwarnings('ignore', message='Found more than one matching response')
# Suppress other non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

# Load pretrained PhaseNet for P-wave picks
picker = sbm.PhaseNet.from_pretrained("original")

# FDSN Network clients to try (in order of preference)
FDSN_CLIENTS = ['NCEDC', 'SCEDC', 'IRIS']

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def station_data_exists(station, eq_time, pre_time, post_time,
                       network: str, location: str, channel: str) -> bool:
    """
    Check if station data exists using FDSN availability service
    This avoids downloading actual waveform data just to check availability
    Falls back to minimal waveform request if availability service fails
    """
    start = eq_time - pre_time
    end = eq_time + post_time

    # Try each FDSN client until one has availability data
    for client_name in FDSN_CLIENTS:
        try:
            client = Client(client_name)
            # First try availability service (most efficient)
            try:
                availability = client.get_availability(
                    network=network,
                    station=station.code,
                    location=location,
                    channel=channel,
                    starttime=start,
                    endtime=end
                )
                # If we get any availability records, data exists
                if availability and len(availability) > 0:
                    return True
            except:
                # Availability service failed, fallback to minimal waveform check
                # Use a very short time window to minimize download
                short_start = eq_time - 1  # Just 1 second before earthquake
                short_end = eq_time + 1    # Just 1 second after earthquake

                st_temp = client.get_waveforms(
                    network=network,
                    station=station.code,
                    location=location,
                    channel=channel,
                    starttime=short_start,
                    endtime=short_end
                )
                if len(st_temp) > 0:
                    return True
        except:
            # This client doesn't have the data, try next one
            continue

    return False

def filter_inventory(inventory: Inventory, eq_time, pre_time, post_time, 
                    network: str, location: str, channel: str) -> Inventory:
    """Filter inventory to only include stations with available data"""
    kept_networks = []
    for net in inventory.networks:
        kept_stns = []
        for st in net.stations:
            if station_data_exists(st, eq_time, pre_time, post_time, 
                                 network, location, channel):
                kept_stns.append(st)
        if kept_stns:
            net.stations = kept_stns
            kept_networks.append(net)
    
    inventory.networks = kept_networks
    return inventory

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATABASE CREATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_database(db_path='seismic_data.db'):
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

    try:
        # Download waveform using OBSpy FDSN client
        # Since filtering already confirmed data exists, try clients more efficiently
        st_stream = None
        client_used = None

        # Try each FDSN client until one works
        for client_name in FDSN_CLIENTS:
            try:
                client = Client(client_name)
                st_stream = client.get_waveforms(
                    network=network,
                    station=station_code,
                    location='*',
                    channel=channel,
                    starttime=eq_time - pre_time,
                    endtime=eq_time + post_time
                )
                if len(st_stream) > 0:
                    client_used = client_name
                    break
            except:
                continue

        if st_stream is None or len(st_stream) < 1:
            return None
        
        # Merge if multiple traces
        st_stream.merge(fill_value='interpolate')
        
        # Get single trace (use the one with best data availability)
        if len(st_stream) > 1:
            # Select trace with most samples
            tr = max(st_stream, key=lambda t: t.stats.npts)
        else:
            tr = st_stream[0]
        
        # Create a more specific inventory selection for this trace
        specific_inv = inventory.select(
            network=tr.stats.network,
            station=tr.stats.station,
            location=tr.stats.location,
            channel=tr.stats.channel,
            time=eq_time
        )
        
        # Remove instrument response
        try:
            if specific_inv:
                # Use specific inventory selection
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr.remove_response(inventory=specific_inv, output="DISP", zero_mean=True)
            else:
                # Fallback to original inventory if specific selection fails
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr.remove_response(inventory=inventory, output="DISP", zero_mean=True)
        except Exception as e:
            pass  # Silently continue if response removal fails
        
        # Get PhaseNet picks for reference
        picks = picker.classify(st_stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks
        if not picks:
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
            'station_lon': station_lon,
            'location': tr.stats.location  # Store actual location code used
        }
        
    except Exception as e:
        return None

def mine_earthquake_data(eq_time, eq_lat, eq_lon, eq_mag=None, eq_depth=None,
                        radius_km=250, networks=['NC'], channels=['HNE'], 
                        pre_time=3, post_time=120):
    """Mine seismic data for a single earthquake"""
    
    if not isinstance(eq_time, UTCDateTime):
        eq_time = UTCDateTime(eq_time)
    
    # Define time window
    start_time = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = eq_time.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Get station inventory - try each FDSN client
    print(f"\nProcessing earthquake at {eq_time}")
    print("Getting station inventory...")
    
    # Convert lists to comma-separated strings for inventory query
    network_str = ','.join(networks)
    channel_str = ','.join(channels)
    
    inventory = None
    for client_name in FDSN_CLIENTS:
        try:
            client = Client(client_name)
            temp_inv = client.get_stations(
                network=network_str, 
                latitude=eq_lat, 
                longitude=eq_lon,
                starttime=start_time, 
                endtime=end_time, 
                maxradius=radius_km/111.2,
                location='*', 
                channel=channel_str, 
                level="response"
            )
            if temp_inv:
                if inventory is None:
                    inventory = temp_inv
                else:
                    # Combine inventories from different sources
                    inventory += temp_inv
        except:
            continue
    
    if not inventory:
        print("No inventory found for this earthquake")
        return []
    
    print("Processing all networks and channels...")
    
    # Process each network and channel combination
    all_waveforms = []
    total_stations = 0
    
    for network in networks:
        for channel in channels:
            # Filter inventory for this specific network/channel
            filtered_inv = inventory.select(network=network, channel=channel)
            
            if not filtered_inv:
                continue
                
            # Further filter for available data
            filtered_inv = filter_inventory(filtered_inv, eq_time, pre_time, post_time, 
                                          network, '*', channel)
            
            stations = []
            for net in filtered_inv.networks:
                stations.extend(net.stations)
            
            if not stations:
                continue
                
            total_stations += len(stations)
            
            # Process each station
            for station in stations:
                result = process_station(station, start_time, pre_time, post_time, 
                                       eq_time, network, channel, inventory)
                if result:
                    # Calculate distance from earthquake
                    dist_deg = np.sqrt((station.latitude - eq_lat)**2 + 
                                      (station.longitude - eq_lon)**2)
                    result['distance_km'] = dist_deg * 111.2
                    result['network'] = network
                    result['channel'] = channel
                    result['location'] = result.get('location', '')  # Use actual location from trace
                    result['eq_lat'] = eq_lat
                    result['eq_lon'] = eq_lon
                    result['eq_mag'] = eq_mag
                    result['eq_depth'] = eq_depth
                    all_waveforms.append(result)
    
    if total_stations > 0:
        print(f"Found {total_stations} stations, successfully processed {len(all_waveforms)} waveforms")
    else:
        print("No stations found with available data")
    return all_waveforms

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN DATA MINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def main(csv_path='query.csv', 
         db_path='seismic_data.db',
         radius_km=300,
         channels=['HNE', 'HNN', 'HNZ'],
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
    channels : list
        List of channel codes to try
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
        print(f"{'='*60}")
        
        try:
            # Mine waveform data for all channels (network wildcard)
            all_waveforms = mine_earthquake_data(
                eq_time=eq['time'],
                eq_lat=eq['latitude'],
                eq_lon=eq['longitude'],
                eq_mag=eq.get('mag'),
                eq_depth=eq.get('depth'),
                radius_km=radius_km,
                channels=channels,  # Only pass channels
                pre_time=pre_time,
                post_time=post_time
            )
            
            # Insert earthquake record if we have waveforms
            if all_waveforms:
                eq_time = UTCDateTime(eq['time'])
                cursor.execute('''
                    INSERT INTO earthquakes (event_time, latitude, longitude, magnitude, depth, place, event_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (eq_time.timestamp, eq['latitude'], eq['longitude'], 
                      eq.get('mag'), eq.get('depth'), eq.get('place', ''), eq.get('id', '')))
                
                earthquake_id = cursor.lastrowid
                
                # Insert waveforms into database
                for wf in all_waveforms:
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
                total_waveforms += len(all_waveforms)
                print(f"✅ Added {len(all_waveforms)} waveforms to database")
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
        db_path='seismic_data_5.db',
        radius_km=100,  # User-specific radius for all earthquakes
        channels=['HHZ'],  # Configurable channels
        pre_time=3,  # Fixed
        post_time=120  # Fixed
    )