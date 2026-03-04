# prosthesis_2
Forearm prosthesis using 4 Mioware Muscle Sensors.

- ﬂexor carpi radialis а0
- brachioradialis а1
- ﬂexor carpi ulnaris а2
- flexor digitorum superficialis а3

    'rock': 0,           # all closed
    'scissors': 1,       # index, middle opened, others closed
    'paper': 2,          # all opened
    'fuck': 3,           # middle finger opened, others closed
    'three': 4,          # index, middle, ring opened, others closed
    'four': 5,           # only thumb closed
    'good': 6,           # only thumb opened
    'okay': 7,           # index and thumb make circle, others opened
    'finger-gun': 8,     # index, thumb opened, others closed
    'rest': 9            # relaxed

ph4 I2C2_SCL
ph5 I2C2_SDA

PA2 USART2_TX
PA3 USART2_RX

PC0 ADC2_IN10
PC1 ADC2_IN11
PA4 ADC2_IN4
PC4 ADC2_IN14