#include "src/model.h"
#include "yes.c"
#include "no.c"

void testInference(int8_t input_data[1920]) {
  int8_t output[4];
  TVMExecute(input_data, output);
  
  for(int i = 0; i < 4; i++) {
    Serial.print(output[i]);
    Serial.print(", ");
  }
  Serial.println();
}

void setup() {
  Serial.begin(9600);
  TVMInitialize();

  Serial.println("Yes results:");
  testInference(yes_data);
  
  Serial.println("No results:");
  testInference(no_data);

  Serial.end();
}
void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
}
