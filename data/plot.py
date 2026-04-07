"""
Detailed EMG Plot with Sample Number Grid - 3 Sections
For easy identification of gesture ranges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.signal import find_peaks

def create_detailed_plot(data_path="data/all_g.txt", save_dir="plots"):
    """Create detailed plots with clear sample number annotations"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    print(f"Loaded {len(df)} samples ({len(df)/1000:.1f} seconds at ~1000Hz)")
    
    # Create combined signal from reliable sensors as numpy array
    combined = (df['ch1'].values + df['ch2'].values) / 2
    
    # Remove any NaN values
    combined = np.nan_to_num(combined)
    
    # Define 3 sections
    sections = [
        (0, min(20000, len(df)), "Section 1: Beginning (Samples 0-20000)"),
        (20000, min(40000, len(df)), "Section 2: Middle (Samples 20000-40000)"),
        (40000, len(df), "Section 3: End (Samples 40000-67882)")
    ]
    
    for start_idx, end_idx, title in sections:
        if start_idx >= len(df):
            continue
            
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Get the data for this section
        x_range = np.arange(start_idx, end_idx)
        y_range = combined[start_idx:end_idx]
        
        # Check if we have valid data
        if len(y_range) == 0:
            print(f"Warning: No data in section {start_idx}-{end_idx}")
            continue
        
        # Plot the signal
        ax.plot(x_range, y_range, 'b-', linewidth=0.8, alpha=0.7)
        
        # Add horizontal line at mean for reference
        mean_val = np.mean(y_range)
        if not np.isnan(mean_val):
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.0f}')
        
        # Add vertical grid lines every 500 samples with labels
        xticks = np.arange(start_idx, end_idx, 500)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks], rotation=45, ha='right', fontsize=8)
        
        # Add minor grid every 100 samples
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(True, which='major', alpha=0.3, linestyle='-')
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        # Add vertical lines every 1000 samples with different style
        for x in range(start_idx, end_idx, 1000):
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            # Add label every 5000 samples
            if x % 5000 == 0:
                ax.text(x, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 1000, 
                       f'{x}', ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add annotations for gesture locations (based on expected sequence)
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
        
        # Add expected gesture regions as background
        y_min_display, y_max_display = ax.get_ylim()
        if y_max_display == 0:
            y_max_display = 2000
            
        for i, gesture in enumerate(expected_sequence):
            block_start = int(i * samples_per_block)
            block_end = int((i + 1) * samples_per_block)
            
            # Only show if in current view range
            if block_end > start_idx and block_start < end_idx:
                color = 'lightgreen' if gesture != 'rest' else 'lightgray'
                alpha_val = 0.3 if gesture != 'rest' else 0.1
                ax.axvspan(max(block_start, start_idx), min(block_end, end_idx), 
                          alpha=alpha_val, color=color)
                
                # Add text label in the middle of the block
                if gesture != 'rest' and block_start >= start_idx - 1000 and block_start <= end_idx + 1000:
                    mid_x = (block_start + block_end) // 2
                    if start_idx <= mid_x <= end_idx:
                        ax.text(mid_x, y_max_display * 0.95, gesture, 
                               ha='center', fontsize=9, rotation=45,
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Find peaks in this section
        try:
            # Adjust parameters for peak detection
            peaks, properties = find_peaks(y_range, distance=300, prominence=50)
            peak_samples = x_range[peaks]
            peak_values = y_range[peaks]
            
            # Annotate peaks with sample numbers
            for sample, value in zip(peak_samples[:20], peak_values[:20]):
                ax.annotate(f'{sample}', xy=(sample, value), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Warning: Peak detection failed for section {start_idx}-{end_idx}: {e}")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        
        # Set y-limits safely
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
        
        # Add legend
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
        
        print(f"✅ Saved: {save_dir}/detailed_plot_{start_idx}_{end_idx}.png")

def create_range_worksheet(data_path="data/all_g.txt"):
    """Create a worksheet to fill in gesture ranges"""
    
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
    
    print("\n" + "="*80)
    print("GESTURE RANGE WORKSHEET")
    print("="*80)
    print("\nLook at the detailed plots and fill in the ACTUAL stable ranges:")
    print("\nFormat: (start_sample, end_sample, 'gesture_name')")
    print("\n" + "-"*80)
    
    gesture_num = 1
    for i, gesture in enumerate(expected_sequence):
        expected_start = int(i * samples_per_block)
        expected_end = int((i + 1) * samples_per_block)
        
        if gesture != 'rest':
            print(f"\n#{gesture_num:2d} {gesture.upper():12} | Expected: {expected_start:6d} - {expected_end:6d} | Actual: (_____, _____, '{gesture}')")
            gesture_num += 1
    
    print("\n" + "="*80)
    
    # Generate Python code template
    print("\n\n# COPY THIS TEMPLATE AND FILL IN ACTUAL NUMBERS:\n")
    print("gesture_ranges = [")
    gesture_num = 1
    for i, gesture in enumerate(expected_sequence):
        expected_start = int(i * samples_per_block)
        expected_end = int((i + 1) * samples_per_block)
        if gesture != 'rest':
            print(f"    # {gesture.upper()} #{gesture_num}")
            print(f"    ({expected_start}, {expected_end}, '{gesture}'),  # ← REPLACE with actual stable range")
            gesture_num += 1
    print("]")
    
    # Also create a CSV template
    template_data = []
    gesture_num = 1
    for i, gesture in enumerate(expected_sequence):
        if gesture != 'rest':
            template_data.append({
                'Number': gesture_num,
                'Gesture': gesture.upper(),
                'Expected_Start': int(i * samples_per_block),
                'Expected_End': int((i + 1) * samples_per_block),
                'Actual_Start': '',
                'Actual_End': '',
                'Duration_ms': ''
            })
            gesture_num += 1
    
    template_df = pd.DataFrame(template_data)
    template_df.to_csv("gesture_ranges_template.csv", index=False)
    print("\n✅ Created 'gesture_ranges_template.csv' - open in Excel/LibreOffice to fill in")

def create_simple_plot(data_path="data/all_g.txt"):
    """Create a simple plot without peak detection for comparison"""
    
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    combined = (df['ch1'].values + df['ch2'].values) / 2
    combined = np.nan_to_num(combined)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot the full signal
    ax.plot(combined, 'b-', linewidth=0.5, alpha=0.7)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('EMG Amplitude (CH1+CH2)', fontsize=12)
    ax.set_title('Full EMG Signal - Use this to see overall pattern', fontsize=14)
    
    # Add vertical lines every 5000 samples
    for x in range(0, len(df), 5000):
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.text(x, ax.get_ylim()[1] * 0.95, f'{x}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("plots/full_signal_overview.png", dpi=150)
    plt.show()
    print("✅ Saved: plots/full_signal_overview.png")

def main():
    print("=== Detailed EMG Plot Generator for Gesture Labeling ===\n")
    
    # First create a simple overview plot
    print("0. Creating overview plot...")
    create_simple_plot()
    
    # Create detailed plots with numbered samples
    print("\n1. Creating detailed plots with sample numbers...")
    create_detailed_plot()
    
    # Create worksheet
    print("\n2. Creating range worksheet...")
    create_range_worksheet()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n✅ Plots saved in 'plots/' folder:")
    print("  - full_signal_overview.png (overall view)")
    print("  - detailed_plot_0_20000.png")
    print("  - detailed_plot_20000_40000.png")
    print("  - detailed_plot_40000_67882.png")
    print("\nInstructions:")
    print("  1. Open the detailed plots")
    print("  2. For each gesture, find the STABLE part (flat top of the peak)")
    print("  3. Note the START and END sample numbers")
    print("  4. Fill in the 'gesture_ranges' list in your labeling script")
    print("\nExample: If rock #1 is stable from sample 1800 to 2200")
    print("  Write: (1800, 2200, 'rock')")
    print("\n  5. Everything else becomes 'rest' automatically")
    
    # Ask if user wants to open the plots
    print("\n" + "-"*60)
    open_plots = input("Open the plots folder? (y/n): ").strip().lower()
    if open_plots == 'y':
        import subprocess
        import os
        subprocess.Popen(['xdg-open', 'plots'])

if __name__ == "__main__":
    main()