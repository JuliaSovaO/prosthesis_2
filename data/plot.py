import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.signal import find_peaks

def create_detailed_plot(data_path="data/all_g.txt", save_dir="plots"):    
    Path(save_dir).mkdir(exist_ok=True)
    
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    print(f"Loaded {len(df)} samples ({len(df)/1000:.1f} seconds at ~1000Hz)")
    
    combined = (df['ch1'].values + df['ch2'].values) / 2
    
    combined = np.nan_to_num(combined)
    
    sections = [
        (0, min(20000, len(df)), "Section 1: Beginning (Samples 0-20000)"),
        (20000, min(40000, len(df)), "Section 2: Middle (Samples 20000-40000)"),
        (40000, len(df), "Section 3: End (Samples 40000-67882)")
    ]
    
    for start_idx, end_idx, title in sections:
        if start_idx >= len(df):
            continue
            
        fig, ax = plt.subplots(figsize=(20, 12))
        
        x_range = np.arange(start_idx, end_idx)
        y_range = combined[start_idx:end_idx]
        
        if len(y_range) == 0:
            print(f"Warning: No data in section {start_idx}-{end_idx}")
            continue
        
        ax.plot(x_range, y_range, 'b-', linewidth=0.8, alpha=0.7)
        
        mean_val = np.mean(y_range)
        if not np.isnan(mean_val):
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.0f}')
        
        xticks = np.arange(start_idx, end_idx, 500)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks], rotation=45, ha='right', fontsize=8)
        
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(True, which='major', alpha=0.3, linestyle='-')
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        for x in range(start_idx, end_idx, 1000):
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            if x % 5000 == 0:
                ax.text(x, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 1000, 
                       f'{x}', ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        total_samples = len(df)
        expected_blocks = 36
        samples_per_block = total_samples / expected_blocks
        
        expected_sequence = [
            'rock', 'rest', 'rock', 'rest',
            'scissors', 'rest', 'scissors', 'rest',
            'paper', 'rest', 'paper', 'rest',
            'fuck', 'rest', 'fuck', 'rest',
            'three', 'rest', 'three', 'rest',
            'four', 'rest', 'four', 'rest',
            'good', 'rest', 'good', 'rest',
            'okay', 'rest', 'okay', 'rest',
            'finger-gun', 'rest', 'finger-gun', 'rest'
        ]
        
        y_min_display, y_max_display = ax.get_ylim()
        if y_max_display == 0:
            y_max_display = 2000
            
        for i, gesture in enumerate(expected_sequence):
            block_start = int(i * samples_per_block)
            block_end = int((i + 1) * samples_per_block)
            
            if block_end > start_idx and block_start < end_idx:
                color = 'lightgreen' if gesture != 'rest' else 'lightgray'
                alpha_val = 0.3 if gesture != 'rest' else 0.1
                ax.axvspan(max(block_start, start_idx), min(block_end, end_idx), 
                          alpha=alpha_val, color=color)
                
                if gesture != 'rest' and block_start >= start_idx - 1000 and block_start <= end_idx + 1000:
                    mid_x = (block_start + block_end) // 2
                    if start_idx <= mid_x <= end_idx:
                        ax.text(mid_x, y_max_display * 0.95, gesture, 
                               ha='center', fontsize=9, rotation=45,
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        try:
            peaks, properties = find_peaks(y_range, distance=300, prominence=50)
            peak_samples = x_range[peaks]
            peak_values = y_range[peaks]
            
            for sample, value in zip(peak_samples[:20], peak_values[:20]):
                ax.annotate(f'{sample}', xy=(sample, value), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Warning: Peak detection failed for section {start_idx}-{end_idx}: {e}")
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        
        valid_y = y_range[~np.isnan(y_range)]
        if len(valid_y) > 0:
            y_min, y_max = np.percentile(valid_y, [1, 99])
            if not np.isnan(y_min) and not np.isnan(y_max):
                ax.set_ylim(y_min - 50, y_max + 50)
            else:
                ax.set_ylim(0, 2000)
        else:
            ax.set_ylim(0, 2000)
        
        ax.set_title(f'{title}\nUse these sample numbers to identify gesture stable ranges', fontsize=14)
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('EMG Amplitude (CH1+CH2 - reliable sensors)', fontsize=12)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.3, label='Expected Gesture Region'),
            Patch(facecolor='lightgray', alpha=0.2, label='Expected Rest Region'),
            plt.Line2D([0], [0], color='blue', linewidth=1, label='EMG Signal'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Mean Value')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/detailed_plot_{start_idx}_{end_idx}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {save_dir}/detailed_plot_{start_idx}_{end_idx}.png")

def create_range_worksheet(data_path="data/test3.txt"):
    
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    total_samples = len(df)
    
    expected_sequence = [
        'rock', 'rest', 'rock', 'rest',
        'scissors', 'rest', 'scissors', 'rest',
        'paper', 'rest', 'paper', 'rest',
        'fuck', 'rest', 'fuck', 'rest',
        'three', 'rest', 'three', 'rest',
        'four', 'rest', 'four', 'rest',
        'good', 'rest', 'good', 'rest',
        'okay', 'rest', 'okay', 'rest',
        'finger-gun', 'rest', 'finger-gun', 'rest'
    ]
    
    samples_per_block = total_samples / len(expected_sequence)
    
    gesture_num = 1
    for i, gesture in enumerate(expected_sequence):
        expected_start = int(i * samples_per_block)
        expected_end = int((i + 1) * samples_per_block)
        
        if gesture != 'rest':
            print(f"\n#{gesture_num:2d} {gesture.upper():12} | Expected: {expected_start:6d} - {expected_end:6d} | Actual: (_____, _____, '{gesture}')")
            gesture_num += 1
    
def create_simple_plot(data_path="data/all_g.txt"):    
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    combined = (df['ch1'].values + df['ch2'].values) / 2
    combined = np.nan_to_num(combined)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.plot(combined, 'b-', linewidth=0.5, alpha=0.7)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('EMG Amplitude (CH1+CH2)', fontsize=12)
    ax.set_title('Full EMG Signal - Use this to see overall pattern', fontsize=14)
    
    for x in range(0, len(df), 5000):
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.text(x, ax.get_ylim()[1] * 0.95, f'{x}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("plots/full_signal_overview.png", dpi=150)
    plt.show()
    print("Saved: plots/full_signal_overview.png")

def main():
    print("=== Detailed EMG Plot Generator for Gesture Labeling ===\n")
    print("0. Creating overview plot...")
    create_simple_plot()
    print("\n1. Creating detailed plots with sample numbers...")
    create_detailed_plot()
    print("\n2. Creating range worksheet...")
    create_range_worksheet()

if __name__ == "__main__":
    main()