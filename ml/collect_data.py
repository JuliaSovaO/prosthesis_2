# ml/collect_new_data.py
import serial
import time
import csv
from datetime import datetime
import sys

def collect_data(gesture_name, duration_seconds=5, samples_per_gesture=500):
    """
    Collect data for a specific gesture
    
    Args:
        gesture_name: Name of the gesture ('rock', 'paper', etc.)
        duration_seconds: How long to collect data
        samples_per_gesture: Target number of samples
    """
    print(f"\n=== Collecting data for: {gesture_name.upper()} ===")
    print(f"Get ready to perform the gesture...")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("START! Perform the gesture now...")
    
    data = []
    start_time = time.time()
    
    # Read data for specified duration
    while len(data) < samples_per_gesture:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line and ',' in line:
                # Format: "ch1,ch2,ch3,ch4,gesture"
                parts = line.split(',')
                if len(parts) >= 4:
                    # Add gesture label
                    parts.append(gesture_name)
                    data.append(parts)
                    
                    # Progress indicator
                    if len(data) % 50 == 0:
                        print(f"Collected {len(data)}/{samples_per_gesture} samples")
        except UnicodeDecodeError:
            continue
        except KeyboardInterrupt:
            print("\nCollection interrupted!")
            return None
    
    elapsed = time.time() - start_time
    print(f"Finished! Collected {len(data)} samples in {elapsed:.1f} seconds")
    print(f"Average rate: {len(data)/elapsed:.1f} Hz")
    
    return data

# Main collection routine
if __name__ == "__main__":
    # Configure serial port (adjust for your system)
    port = '/dev/ttyUSB0'  # or COM3 on Windows
    baudrate = 230400
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection
        
        print("Serial connection established!")
        print("Flushing buffer...")
        ser.flushInput()
        
        # Read and discard any initial data
        while ser.in_waiting:
            ser.readline()
        
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)
    
    # Gestures to collect (in order)
    gestures = [
        'rest',
        'rock',
        'scissors', 
        'paper',
        'fuck',
        'three',
        'four',
        'good',
        'okay',
        'finger-gun'
    ]
    
    all_data = []
    
    for gesture in gestures:
        gesture_data = collect_data(gesture, duration_seconds=10, samples_per_gesture=1000)
        if gesture_data:
            all_data.extend(gesture_data)
        
        # Short break between gestures
        if gesture != gestures[-1]:  # Not the last gesture
            print("\nRest for 5 seconds...")
            time.sleep(5)
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ml/new_data_{timestamp}.csv'
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # No header - just raw data
        writer.writerows(all_data)
    
    print(f"\nData saved to: {filename}")
    print(f"Total samples collected: {len(all_data)}")
    
    ser.close()