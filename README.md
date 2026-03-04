# Forearm Prosthesis for Training and Self-Experience (ML)
**Team**: [Olena Yakovenko](https://github.com/OlenaYakovenk0), [Yuliia Sova](https://github.com/JuliaSovaO)

### Overview
The FERRARI project is an educational myoelectric prosthesis system that reads muscle activity through EMG sensors, identifies hand gestures, and controls a 3D-printed mechanical hand. Designed for non-amputee users to experience EMG-based prosthesis operation.

### Hardware Components
- **MCU**: STM32F723E-DISCO (216MHz Cortex-M7, 512KB Flash)
- **Sensors**: 2x MyoWare 2.0 + 2x MyoWare 1.0 EMG sensors
- **Actuation**: PCA9685 servo controller driving 5 servos
- **Power**: Split supply (6V servos / 3.3V logic) with common ground reference

### Pin Connections
| Component | Interface | Pins |
|-----------|-----------|------|
| EMG Sensor 1 | ADC2_IN10 | PC0 |
| EMG Sensor 2 | ADC2_IN11 | PC1 |
| EMG Sensor 3 | ADC2_IN4 | PA4 |
| EMG Sensor 4 | ADC2_IN14 | PC4 |
| PCA9685 | I2C2 | PH4(SCL), PH5(SDA) |
| Debug UART | USART6 | PC6(TX), PC7(RX) |

### Sensor Placement (Anatomical)
- **A0 (PC0)** - Flexor carpi radialis (wrist flexion/hand closing)
- **A1 (PC1)** - Brachioradialis (forearm activation patterns)
- **A2 (PA4)** - Flexor carpi ulnaris (power grip)
- **A3 (PC4)** - Flexor digitorum superficialis (finger flexion)

### Gesture Set (10 classes)
| Gesture | Description |
|---------|-------------|
| rock | All fingers closed (fist) |
| scissors | Index+middle open, others closed |
| paper | All fingers open |
| one | Middle finger open, others closed |
| three | Index+middle+ring open, others closed |
| four | Only thumb closed |
| good | Only thumb open |
| okay | Index+thumb circle, others open |
| finger-gun | Index+thumb open, others closed |
| rest | Relaxed, no contraction |

### Signal Processing
- **Moving Average Filter**: 50-sample window with circular buffer ($O(1)$ per sample)
- **Sampling Rate**: ~1000 Hz per channel
- **Output Format**: `CH1,CH2,CH3,CH4` (CSV over UART at 921600 baud)

### Feature Extraction (24 features)
| Feature | Formula |
|---------|---------|
| RMS | $\sqrt{\frac{1}{N}\sum x_i^2}$ |
| Variance | $\frac{1}{N}\sum (x_i - \mu)^2$ |
| MAV | $\frac{1}{N}\sum |x_i|$ |
| Zero Crossing | $\sum [\text{sgn}(x_i x_{i+1}) \cap |x_i - x_{i+1}| \geq \epsilon]$ |
| Slope Sign Change | $\sum f[(x_i - x_{i-1})(x_i - x_{i+1})]$ |
| Waveform Length | $\sum |x_{i+1} - x_i|$ |