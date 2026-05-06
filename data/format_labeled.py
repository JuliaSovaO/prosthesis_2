import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# PART 1: Merge labeled TXT files into CSV

def is_valid_labeled_line(line):
    """
    Check if a line has the format: number,number,number,number,label
    where numbers are between 0 and 9999 and label is non-empty
    """
    line = line.strip()
    
    # Pattern: 4 comma-separated numbers (1-4 digits each), a comma, and a label
    pattern = r'^\d{1,4},\d{1,4},\d{1,4},\d{1,4},.+$'
    
    if not re.match(pattern, line):
        return False
    
    parts = line.split(',')
    nums = [int(p) for p in parts[:4]]
    
    return all(0 <= n <= 9999 for n in nums)

def merge_labeled_files(folder_path, output_filename="all_data.csv"):
    """
    Merge all labeled .txt files into a single CSV without modifying originals
    
    Parameters:
    - folder_path: path to folder containing labeled .txt files
    - output_filename: name of output CSV file
    """
    
    folder = Path(folder_path)
    
    # Find all .txt files
    files = list(folder.glob("*.txt"))
    files = [f for f in files if f.name != output_filename]
    
    print(f"Found {len(files)} files to merge")
    
    all_lines = []
    processed_files = 0
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Validate lines (should already be labeled)
        valid_lines = [line for line in lines if is_valid_labeled_line(line)]
        invalid_count = len(lines) - len(valid_lines)
        
        if not valid_lines:
            print(f"Warning: No valid labeled lines in {file_path.name}")
            continue
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid lines skipped in {file_path.name}")
        
        # Extract label from filename (the gesture name)
        file_label = file_path.stem
        
        # Verify the label in the file matches the filename
        # (optional consistency check)
        for line in valid_lines:
            parts = line.rsplit(',', 1)  # Split on last comma
            line_label = parts[1].strip()
            if line_label != file_label:
                print(f"Warning: Label mismatch in {file_path.name}: "
                      f"found '{line_label}', expected '{file_label}'")
        
        all_lines.extend(valid_lines)
        processed_files += 1
        
        print(f"Merged: {file_path.name} -> {len(valid_lines)} rows")
    
    # Write merged file
    output_file = folder / output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    
    print(f"\nDone!")
    print(f"Files merged: {processed_files}")
    print(f"Total rows: {len(all_lines)}")
    print(f"Output saved to: {output_file}")
    print("Original files were NOT modified")
    
    return output_file

# PART 2: Load and analyze the merged data

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

# PART 3: Plotting functions

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
        axes[i].set_ylabel(sensor_name, fontsize=15)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', labelsize=13)
    
    axes[-1].set_xlabel('Time step')
    fig.suptitle(f'Signal shapes for gesture: {gesture}', fontsize=18)
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
    
    plt.title(f'Value distributions for different gestures on {sensor.upper()}', fontsize=18)
    plt.xlabel('Sensor value', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(loc='upper right', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_{sensor}.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/density_{sensor}.png")

def plot_all_density_plots(df_long, output_dir="plots"):
    """Create density plots for all sensors"""
    sensors = ['s1', 's2', 's3', 's4']
    x_limits = {'s1': 800, 's2': 800, 's3': 1800, 's4': 1500}
    
    for sensor in sensors:
        plot_sensor_density(df_long, sensor, x_limits.get(sensor), output_dir)
    
    print("All density plots created")

# PART 4: Main execution

def main():
    print("="*60)
    print("EMG Data Processing Pipeline (Merge Only)")
    print("="*60)
    
    # Step 1: Merge labeled files
    print("\n--- Step 1: Merging labeled files ---")
    folder_path = "data/06051-mykyta"
    
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        print("Please update the folder_path variable")
        return
    
    output_file = merge_labeled_files(folder_path, "all_data.csv")
    
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