import serial
import time
import csv
from datetime import datetime
import os

class EMGDataCollector:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.current_file = None
        self.writer = None
        self.sample_count = 0
        self.gesture_names = [
            "REST", "ROCK", "SCISSORS", "PAPER", "ONE",
            "THREE", "FOUR", "GOOD", "OKAY", "FINGER_GUN"
        ]
        
    def wait_for_start(self):
        print("Waiting for CSV_START command...")
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if line.startswith("CSV_START"):
                filename = line.split(',')[1]
                self.start_new_file(filename)
                return filename
    
    def start_new_file(self, filename):
        if self.current_file:
            self.current_file.close()
        
        # Create data directory if it doesn't exist
        if not os.path.exists('collected_data'):
            os.makedirs('collected_data')
        
        filepath = os.path.join('collected_data', filename)
        self.current_file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.current_file)
        self.writer.writerow(['timestamp_us', 'ch1', 'ch2', 'ch3', 'gesture_id'])
        self.sample_count = 0
        
        print(f"Started recording to: {filepath}")
    
    def process_line(self, line):
        if line.startswith("CSV_END"):
            self.finish_file()
            return False
        elif line.startswith("DATA,") or line[0].isdigit():
            # Parse CSV data
            try:
                parts = line.split(',')
                if len(parts) == 5:
                    self.writer.writerow(parts)
                    self.sample_count += 1
                    
                    # Show progress
                    if self.sample_count % 1000 == 0:
                        print(f"  Samples: {self.sample_count}")
            except:
                pass
        return True
    
    def finish_file(self):
        if self.current_file:
            self.current_file.close()
            print(f"File closed. Total samples: {self.sample_count}")
            self.current_file = None
            self.writer = None
    
    def collect_all_data(self):
        print("=== EMG Data Collection Receiver ===")
        print("Sampling rate: 2000 Hz")
        print("Waiting for STM32 connection...")
        
        # Wait for initial connection
        time.sleep(2)
        self.ser.flushInput()
        
        file_count = 0
        
        try:
            while True:
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                
                print(f"Received: {line}")
                
                if line.startswith("CSV_START"):
                    filename = self.wait_for_start()
                    file_count += 1
                    print(f"File {file_count}: {filename}")
                    
                    # Read until CSV_END
                    collecting = True
                    while collecting:
                        data_line = self.ser.readline().decode('utf-8').strip()
                        if data_line:
                            collecting = self.process_line(data_line)
                
                elif "Gesture" in line:
                    print(f"Status: {line}")
                
                elif "Exiting" in line or "Complete" in line:
                    print("Collection finished.")
                    break
        
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.finish_file()
            self.ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    collector = EMGDataCollector(port='/dev/ttyACM0')  # Adjust port as needed
    collector.collect_all_data()