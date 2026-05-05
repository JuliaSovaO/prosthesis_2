# prosthesis_2
Forearm prosthesis using 4 Mioware Muscle Sensors.

- ﬂexor carpi radialis а0  gr new
- brachioradialis а1  yel
- ﬂexor carpi ulnaris а2  red
- flexor digitorum superficialis а3

    'rock': 0,     thumb under      # all closed       okay fuck
    'scissors': 1,   напр    # index, middle opened, others closed  rest
    'paper': 2,          # all opened    four good okay
    'fuck': 3,           # middle finger opened, others closed   rest four good
    'three': 4,          # index, middle, ring opened, others closed  four good
    'four': 5,           # only thumb closed   good fuck scissors
    'good': 6,           # only thumb opened     thumb    fuck good okay
    'okay': 7,    thumb       # index and thumb make circle, others opened   four rest
    'finger-gun': 8,     # index, thumb opened, others closed   rock okay
    'rest': 9            # relaxed    okay good

ph4 I2C2_SCL
ph5 I2C2_SDA

PA2 USART2_TX
PA3 USART2_RX

PC0 ADC2_IN10  g
PC1 ADC2_IN11  y
PA4 ADC2_IN4   p
PC4 ADC2_IN14  o