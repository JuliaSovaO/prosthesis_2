#ifndef SIGNAL_VALIDATION_H
#define SIGNAL_VALIDATION_H

#include <stdbool.h>
#include <math.h>

// Based on your new data statistics
#define CH1_MIN_VALID 500   // flexor carpi radialis minimum
#define CH1_MAX_VALID 1200  // flexor carpi radialis maximum
#define CH4_MIN_VALID 200   // flexor digitorum superficialis minimum
#define CH4_MAX_VALID 1300  // flexor digitorum superficialis maximum

// Check if raw ADC values are valid based on new data ranges
bool is_valid_signal_range(uint16_t ch1, uint16_t ch2, uint16_t ch3, uint16_t ch4) {
    // Check each channel is within expected range
    if (ch1 < CH1_MIN_VALID || ch1 > CH1_MAX_VALID) return false;
    if (ch4 < CH4_MIN_VALID || ch4 > CH4_MAX_VALID) return false;
    
    // Additional checks based on your data patterns
    // ch1 should generally be higher than ch2
    if (ch1 < ch2 * 2) return false;  // ch1 should be at least 2x ch2
    
    return true;
}

// Check if features are valid before classification
bool are_features_valid(const float* features) {
    // Check MAV values (features 0, 5, 10, 15)
    float ch1_mav = features[0];
    float ch4_mav = features[15];
    
    // Convert MAV to approximate ADC range
    // MAV ~= average absolute value, so multiply by ~1.5 for max range
    float approx_ch1_adc = ch1_mav * 1.5f;
    float approx_ch4_adc = ch4_mav * 1.5f;
    
    if (approx_ch1_adc < CH1_MIN_VALID || approx_ch1_adc > CH1_MAX_VALID) {
        return false;
    }
    if (approx_ch4_adc < CH4_MIN_VALID || approx_ch4_adc > CH4_MAX_VALID) {
        return false;
    }
    
    return true;
}

#endif // SIGNAL_VALIDATION_H