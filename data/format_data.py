import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ============================================
# PART 1: Clean and merge individual files
# ============================================

def is_valid_line(line):
    """
    Check if a line has exactly 4 numbers separated by commas,
    each between 0 and 9999
    """
    line = line.strip()
    
    pattern = r'^\d{1,4},\d{1,4},\d{1,4},\d{1,4}$'
    
    if not re.match(pattern, line):
        return False
    
    parts = line.split(',')
    nums = [int(p) for p in parts]
    
    if len(nums) != 4:
        return False
    
    return all(0 <= n <= 9999 for n in nums)

def clean_and_merge_files(folder_path, output_filename="all_data.csv"):
    """
    Process all .txt files in folder, clean them, add labels, and merge
    
    Parameters:
    - folder_path: path to folder containing .txt files
    - output_filename: name of output CSV file
    """
    
    folder = Path(folder_path)
    
    # Find all .txt files, exclude existing all_data.csv
    files = list(folder.glob("*.txt"))
    files = [f for f in files if f.name != "all_data.csv" and f.name != "all_data.txt"]
    
    print(f"Found {len(files)} files to process")
    
    all_lines = []
    processed_files = 0
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip()]
        
        valid_lines = [line for line in lines if is_valid_line(line)]
        
        if not valid_lines:
            print(f"Warning: No valid lines in {file_path.name}")
            continue
        
        label = file_path.stem
        
        labeled_lines = [f"{line},{label}" for line in valid_lines]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labeled_lines))
        
        all_lines.extend(labeled_lines)
        processed_files += 1
        
        print(f"Processed: {file_path.name} -> {len(valid_lines)} rows")
    
    output_file = folder / output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    
    print(f"\nDone!")
    print(f"Processed files: {processed_files}")
    print(f"Total rows: {len(all_lines)}")
    print(f"Output saved to: {output_file}")
    
    return output_file

# ============================================
# PART 2: Load and analyze the merged data
# ============================================

def load_merged_data(file_path):
    """Load the merged CSV file"""
    df = pd.read_csv(file_path, header=None, 
                     names=['s1', 's2', 's3', 's4', 'gesture'])
    
    df['s1'] = pd.to_numeric(df['s1'], errors='coerce')
    df['s2'] = pd.to_numeric(df['s2'], errors='coerce')
    df['s3'] = pd.to_numeric(df['s3'], errors='coerce')
    df['s4'] = pd.to_numeric(df['s4'], errors='coerce')
    df['gesture'] = df['gesture'].astype('category')
    
    df = df.dropna()
    
    print(f"Loaded {len(df)} rows, {df['gesture'].nunique()} unique gestures")
    print(f"Gestures: {df['gesture'].cat.categories.tolist()}")
    
    return df

def add_sample_ids(df):
    """Add sample_id column (increments when gesture changes)"""
    df = df.copy()
    gesture_changed = (df['gesture'] != df['gesture'].shift()).fillna(True)
    df['sample_id'] = gesture_changed.cumsum()
    return df

def add_time_steps(df):
    """Add time step (t) column within each sample_id"""
    df = df.copy()
    df['t'] = df.groupby('sample_id').cumcount() + 1
    return df

def create_long_format(df):
    """Convert from wide to long format for easier plotting"""
    df_long = df.melt(
        id_vars=['gesture', 'sample_id', 't'],
        value_vars=['s1', 's2', 's3', 's4'],
        var_name='sensor',
        value_name='value'
    )
    return df_long

# ============================================
# PART 3: Plotting functions
# ============================================

def plot_gesture_signals(df_long, gesture, output_dir="plots"):
    """Plot signal shapes for a specific gesture"""
    Path(output_dir).mkdir(exist_ok=True)
    
    gesture_df = df_long[df_long['gesture'] == gesture]
    first_sample = gesture_df['sample_id'].iloc[0]
    sample_df = gesture_df[gesture_df['sample_id'] == first_sample]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    sensors = ['s1', 's2', 's3', 's4']
    sensor_names = ['Sensor 1 (FCR)', 'Sensor 2 (BR)', 'Sensor 3 (FCU)', 'Sensor 4 (FDS)']
    
    for i, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
        sensor_data = sample_df[sample_df['sensor'] == sensor]
        axes[i].plot(sensor_data['t'], sensor_data['value'], 'b-', linewidth=0.8)
        axes[i].set_ylabel(sensor_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time step')
    fig.suptitle(f'Signal shapes for gesture: {gesture}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gesture_{gesture}_signal.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/gesture_{gesture}_signal.png")

def plot_all_gestures(df_long, output_dir="plots"):
    """Plot first sample of each gesture"""
    gestures = df_long['gesture'].unique()
    
    for gesture in gestures:
        plot_gesture_signals(df_long, gesture, output_dir)
    
    print(f"Plotted {len(gestures)} gestures")

def plot_sensor_density(df_long, sensor, x_max=None, output_dir="plots"):
    """Plot density distribution for a specific sensor"""
    Path(output_dir).mkdir(exist_ok=True)
    
    sensor_data = df_long[df_long['sensor'] == sensor]
    gestures = sensor_data['gesture'].unique()
    
    plt.figure(figsize=(12, 6))
    for gesture in gestures:
        gesture_data = sensor_data[sensor_data['gesture'] == gesture]['value']
        if len(gesture_data) > 0:
            gesture_data.plot.kde(label=gesture, linewidth=1.5)
    
    if x_max:
        plt.xlim(0, x_max)
    
    plt.title(f'Value distributions for different gestures on {sensor.upper()}')
    plt.xlabel('Sensor value')
    plt.ylabel('Density')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_{sensor}.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/density_{sensor}.png")

def plot_all_density_plots(df_long, output_dir="plots"):
    """Create density plots for all sensors"""
    sensors = ['s1', 's2', 's3', 's4']
    x_limits = {'s1': 800, 's2': 500, 's3': 500, 's4': 1500}
    
    for sensor in sensors:
        plot_sensor_density(df_long, sensor, x_limits.get(sensor), output_dir)
    
    print("All density plots created")

# ============================================
# PART 4: Main execution
# ============================================

def main():
    print("="*60)
    print("EMG Data Processing Pipeline")
    print("="*60)
    
    # Step 1: Clean and merge files
    print("\n--- Step 1: Cleaning and merging files ---")
    folder_path = "data/08044"
    
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        print("Please update the folder_path variable")
        return
    
    output_file = clean_and_merge_files(folder_path, "all_data.csv")
    
    # Step 2: Load and analyze
    print("\n--- Step 2: Loading merged data ---")
    df = load_merged_data(output_file)
    
    # Step 3: Add sample IDs and time steps
    print("\n--- Step 3: Adding sample IDs and time steps ---")
    df = add_sample_ids(df)
    df = add_time_steps(df)
    
    # Step 4: Create long format for plotting
    print("\n--- Step 4: Creating long format ---")
    df_long = create_long_format(df)
    
    # Step 5: Generate plots
    print("\n--- Step 5: Generating plots ---")
    plot_all_gestures(df_long)
    plot_all_density_plots(df_long)
    
    # Step 6: Save processed data
    print("\n--- Step 6: Saving processed data ---")
    output_processed = Path(folder_path) / "all_data_processed.csv"
    df.to_csv(output_processed, index=False)
    print(f"Processed data saved to: {output_processed}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total gestures: {df['gesture'].nunique()}")
    print("\nSamples per gesture:")
    gesture_counts = df.groupby('gesture').size()
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count} samples ({count/1000:.1f}s)")
    
    print("\nAll done!")

if __name__ == "__main__":
    main()