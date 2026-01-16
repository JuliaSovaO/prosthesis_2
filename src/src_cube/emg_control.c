#include "emg_control.h"
#include "main.h"
#include <stdio.h>
#include <math.h>

// Constants
#define SAMPLE_RATE_HZ     2000     // As per paper
#define WINDOW_SIZE_MS     250      // 250ms window
#define WINDOW_STEP_MS     25       // 25ms step (90% overlap)
#define SAMPLES_PER_WINDOW (SAMPLE_RATE_HZ * WINDOW_SIZE_MS / 1000)  // 500 samples
#define NUM_CHANNELS       3
#define NUM_FEATURES       6
#define TOTAL_FEATURES     (NUM_CHANNELS * NUM_FEATURES)  // 18

// Data collection state
static bool is_recording = false;
static GestureType current_gesture = GESTURE_REST;
static uint16_t sample_buffer[3][SAMPLE_RATE_HZ * 5];  // 5-second buffer per channel
static uint32_t sample_index = 0;

void EMG_DataCollection_Init(void) {
    is_recording = false;
    sample_index = 0;
    current_gesture = GESTURE_REST;
    printf("EMG Data Collection Initialized\n");
}

void EMG_DataCollection_Start(GestureType gesture) {
    is_recording = true;
    current_gesture = gesture;
    sample_index = 0;
    printf("Recording started for gesture: %d\n", gesture);
}

void EMG_DataCollection_Stop(void) {
    is_recording = false;
    printf("Recording stopped. Collected %lu samples\n", sample_index);
}

void EMG_DataCollection_ProcessSample(void) {
    if (!is_recording || sample_index >= SAMPLE_RATE_HZ * 5) {
        return;
    }
    
    // Get the latest sample from ADC buffer (last sample in DMA buffer)
    int last_sample_idx = (SAMPLES - 1) * ADC_CHANNELS;
    
    sample_buffer[0][sample_index] = adc_buffer[last_sample_idx + 0];  // CH1 (PA0)
    sample_buffer[1][sample_index] = adc_buffer[last_sample_idx + 1];  // CH2 (PA1)
    sample_buffer[2][sample_index] = adc_buffer[last_sample_idx + 2];  // CH3 (PA2)
    
    sample_index++;
    
    // Check if recording time is complete (5 seconds per paper)
    if (sample_index >= SAMPLE_RATE_HZ * 5) {
        EMG_DataCollection_Stop();
    }
}

bool EMG_DataCollection_IsRecording(void) {
    return is_recording;
}

GestureType EMG_DataCollection_GetCurrentGesture(void) {
    return current_gesture;
}

// Helper function to calculate features for one channel
static void CalculateFeatures(uint16_t* window, uint32_t n, TimeDomainFeatures* feat) {
    float sum_sq = 0.0f, sum_abs = 0.0f;
    float wl = 0.0f;
    uint32_t ssc = 0, zc = 0;
    
    // Remove DC offset (approximate)
    float mean = 0;
    for (uint32_t i = 0; i < n; i++) {
        mean += window[i];
    }
    mean /= n;
    
    // Calculate features
    for (uint32_t i = 0; i < n; i++) {
        float x = window[i] - mean;
        sum_abs += fabsf(x);
        sum_sq += x * x;
        
        if (i > 0) {
            wl += fabsf(window[i] - window[i-1]);
        }
        
        // SSC and ZC calculations (simplified)
        if (i > 0 && i < n-1) {
            float x_prev = window[i-1] - mean;
            float x_curr = window[i] - mean;
            float x_next = window[i+1] - mean;
            
            // Slope Sign Change
            if ((x_curr - x_prev) * (x_curr - x_next) > 0) {
                ssc++;
            }
            
            // Zero Crossing with threshold
            if (x_curr * x_next < 0 && fabsf(x_curr - x_next) > 10) {
                zc++;
            }
        }
    }
    
    feat->rms = sqrtf(sum_sq / n);
    feat->var = sum_sq / n;
    feat->mav = sum_abs / n;
    feat->wl = wl;
    feat->ssc = (float)ssc;
    feat->zc = (float)zc;
}

void EMG_DataCollection_ExtractFeatures(uint32_t window_start, TimeDomainFeatures* ch1_feat, 
                                        TimeDomainFeatures* ch2_feat, TimeDomainFeatures* ch3_feat) {
    CalculateFeatures(&sample_buffer[0][window_start], SAMPLES_PER_WINDOW, ch1_feat);
    CalculateFeatures(&sample_buffer[1][window_start], SAMPLES_PER_WINDOW, ch2_feat);
    CalculateFeatures(&sample_buffer[2][window_start], SAMPLES_PER_WINDOW, ch3_feat);
}

void EMG_DataCollection_PrintFeaturesCSV(TimeDomainFeatures* ch1, TimeDomainFeatures* ch2, 
                                         TimeDomainFeatures* ch3) {
    // Output in CSV format for easy import to Python
    printf("%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,"   // CH1: RMS,VAR,MAV,SSC,ZC,WL
           "%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,"   // CH2
           "%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,"   // CH3
           "%d\r\n",                           // Gesture label
           
           ch1->rms, ch1->var, ch1->mav, ch1->ssc, ch1->zc, ch1->wl,
           ch2->rms, ch2->var, ch2->mav, ch2->ssc, ch2->zc, ch2->wl,
           ch3->rms, ch3->var, ch3->mav, ch3->ssc, ch3->zc, ch3->wl,
           current_gesture);
}

uint32_t EMG_DataCollection_GetSampleIndex(void) {
    return sample_index;
}

uint32_t EMG_DataCollection_GetSamplesPerWindow(void) {
    return SAMPLES_PER_WINDOW;
}