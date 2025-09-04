#include <Arduino.h>
#include <TMCStepper.h>

// put function declarations here:
int myFunction(int, int);

#define SCK_PIN 13
#define CIPO_PIN 12
#define COPI_PIN 11
#define STEP_PIN 10
#define DIR_PIN 9
#define CS_PIN 8
#define EN_PIN 7

#define RAMP_DURATION 1000  // Ramp-up duration in milliseconds
#define START_FREQ 1000     // Starting frequency in Hz
#define TARGET_FREQ 10000   // Target frequency in Hz

#define POTENTIOMETER_PIN A6

#include <FastLED.h>
#define NUM_LEDS 9
#define DATA_PIN 6

CRGB leds[NUM_LEDS];

  
TMC2130Stepper driver = TMC2130Stepper(CS_PIN);




void rampFrequency() {
  unsigned long startTime = millis();
  unsigned long currentTime;
  float progress;
  uint16_t topValue;

  while ((currentTime = millis()) - startTime < RAMP_DURATION) {
    progress = (float)(currentTime - startTime) / RAMP_DURATION;
    float currentFreq = START_FREQ + progress * (TARGET_FREQ - START_FREQ);
    
    // Calculate and set the TOP value for the current frequency
    topValue = (F_CPU / currentFreq) - 1;
    ICR1 = topValue;
    
    // Set compare value for 50% duty cycle
    OCR1B = topValue / 2;
    
    // Small delay to control the ramp-up rate
    delayMicroseconds(100);
  }
  
  // Ensure we reach the exact target frequency
  topValue = (F_CPU / TARGET_FREQ) - 1;
  ICR1 = topValue;
  OCR1B = topValue / 2;
}

void setFreq(uint32_t freq) {
  // Clear Timer/Counter Control Registers
  // TCCR1A = 0;
  // TCCR1B = 0;

  uint16_t topValue = (F_CPU / freq) - 1;
  ICR1 = topValue;
  OCR1B = topValue / 2;

  Serial.println(topValue);
}



void setup() {
  Serial.begin(115200);

  // Init pins
  pinMode(EN_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT  );
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW); //Activate driver (LOW active)
  digitalWrite(DIR_PIN, LOW); //LOW or HIGH
  digitalWrite(CS_PIN, HIGH);
  


  // Clear Timer/Counter Control Registers
  TCCR1A = 0;
  TCCR1B = 0;
  
  // Set non-inverting mode on OC1B (pin 10)
  TCCR1A |= (1 << COM1B1);
  
  // Set fast PWM mode with ICR1 as TOP
  TCCR1A |= (1 << WGM11);
  TCCR1B |= (1 << WGM12) | (1 << WGM13);
  
  // Set prescaler to 1
  TCCR1B |= (1 << CS10);
  
  // // Set TOP value for 25 kHz frequency
  // // ICR1 = 639;
  // ICR1 = 958;
  
  // Ensure we reach the exact target frequency
  uint16_t topValue = (F_CPU / TARGET_FREQ) - 1;
  ICR1 = topValue;
  OCR1B = topValue / 2;

  Serial.println(topValue);


  // // Set compare value for 50% duty cycle
  // OCR1B = 319;


  
  // Init TMC 2130
  SPI.begin();                    // SPI drivers
  driver.begin();                 //  SPI: Init CS pins and possible SW SPI pins
  
  // Test Driver Connection
  Serial.print(F("\nTesting connection..."));
  uint8_t result = driver.test_connection();
  if (result) {
      Serial.println(F("failed!"));
      Serial.print(F("Likely cause: "));
      switch(result) {
          case 1: Serial.println(F("loose connection")); break;
          case 2: Serial.println(F("Likely cause: no power")); break;
      }
      Serial.println(F("Fix the problem and reset board."));
      abort();
  }
  Serial.println(F("OK"));

  // Set driver parameters
  driver.toff(5);                 // Enables driver in software
  // driver.rms_current(600);        // Set motor RMS current (default 600)
  driver.rms_current(800);        // Set motor RMS current (default 600)
  driver.microsteps(64);          // Set microsteps 

  driver.en_pwm_mode(true);       // Toggle stealthChop on TMC2130/2160/5130/5160
  // driver.en_spreadCycle(false);   // Toggle spreadCycle on TMC2208/2209/2224
  driver.pwm_autoscale(true);     // Needed for stealthChop

  // Start the frequency ramp-up
  // rampFrequency();


  // Init LED array
  // FastLED.addLeds<WS2811, DATA_PIN, RGB>(leds, NUM_LEDS);
  FastLED.addLeds<WS2812B, DATA_PIN, RGB>(leds, NUM_LEDS);
  
  // setFreq(10000);

  // for (uint32_t ii=0; ii<15000; ii+=200) {
  //   setFreq(ii);
  //   delay(100);
  // }


  // for (uint32_t ii=6000; ii<7500; ii+=100) {
  //   setFreq(ii);
  //   delay(100);
  // }
}


#define STEP_DELAY 40


uint32_t setFreqVal = 0;
int testLed = 0;
void loop() {
  // Calculate new stepper output

  // // Update stepper
  // digitalWrite(STEP_PIN, HIGH);
  // delayMicroseconds(STEP_DELAY);
  // digitalWrite(STEP_PIN, LOW);
  // delayMicroseconds(STEP_DELAY);

  for(int ii=0; ii<NUM_LEDS; ii++){
    int iterDiff = abs(ii-testLed)%NUM_LEDS;
    switch (iterDiff)
    {
    case 0:
      leds[ii] = CRGB(0, 100, 60);
      break;

    case 1:
    case NUM_LEDS-1:
      leds[ii] = CRGB(10, 100, 40);
      break;

    case 2:
    case NUM_LEDS-2:
      leds[ii] = CRGB(20, 100, 10);
      break;

    case 3:
    case NUM_LEDS-3:
      leds[ii] = CRGB(50, 100, 0);
      break;
    
    default:
      leds[ii] = CRGB(100, 100, 0);
      break;
    }
  }
  
  FastLED.show();
  testLed += 1;
  if (testLed > NUM_LEDS) testLed = 0;

  if (setFreqVal < 70000/4) {
    setFreqVal += 500;
    setFreq(setFreqVal);
  }

  delay(100);




  // delayMicroseconds(4*1e6*(16*200/TARGET_FREQ));
  // delay(35*2);

  // Write to LED


  // setFreq(1000);

  // #define TARGET_FREQ 5000   // Target frequency in Hz

  // for (uint32_t ii=500; ii<10000; ii+=200) {
  //   setFreq(ii);
  //   delay(100);
  // }

  // delay(2000);

  // for (uint32_t ii=20000; ii>=500; ii-=200) {
  //   setFreq(ii);
  //   delay(100);
  // }

  // delay(2000);



}

