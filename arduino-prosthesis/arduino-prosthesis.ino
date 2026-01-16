// MyoWare Muscle Sensors Reader

const int sensorPin1 = A0; 
const int sensorPin2 = A1; 
const int sensorPin3 = A2; 
int sensorValue1 = 0;       
int sensorValue2 = 0;  
int sensorValue3 = 0;  

void setup() {
  Serial.begin(9600);
}

void loop() {
  sensorValue1 = analogRead(sensorPin1);
  sensorValue2 = analogRead(sensorPin2);
  sensorValue3 = analogRead(sensorPin3);

  
  Serial.print(sensorValue1);     
  Serial.print(",");         
  Serial.print(sensorValue2);   
  Serial.print(",");            
  Serial.print(sensorValue3);    
  Serial.print("\r\n");            

  
  delay(10);  // 10ms = ~100Hz sampling
}