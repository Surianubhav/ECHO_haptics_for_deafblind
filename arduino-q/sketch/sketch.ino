/*
 * ECHO Project - High-Throughput Wireless Actuation Interface
 * Board: Uno Q (Qualcomm ARM-Zephyr Toolchain Architecture)
 * Description: Ingests 4-word AI CSV strings via a background TCP connection 
 *              and drives haptic pins / Braille grid outputs dynamically.
 */

#include <Arduino.h>
#include <Arduino_RouterBridge.h>
#include <Arduino_RPClite.h>
#include "BrailleVocabulary.h" // Contains BRAILLE_ALPHABET array mappings

// ==========================================
// User Configuration Settings
// ==========================================
const char* SERVER_IP = "192.168.137.1";
const uint16_t SERVER_PORT = 5000;

// ==========================================
// Hardware Pin Layout & Timing Definitions
// ==========================================
const int PIN_THUMB  = 2;
const int PIN_INDEX  = 3;
const int PIN_MIDDLE = 10; 
const int GRID_PINS[6] = {4, 5, 6, 7, 8, 9}; 

const int TIME_SINGLE = 400;
const int TIME_DOUBLE = 200; 
const int TIME_LONG   = 1000;
const int CHAR_DELAY  = 800;  

// ==========================================
// Operational Constraints & Constants
// ==========================================
const unsigned long RECONNECT_INTERVAL = 5000;    
const size_t MAX_PACKET_SIZE           = 128;     

BridgeTCPClient<> client(Bridge);

enum ConnectionState {
  STATE_INIT,
  STATE_CONNECTING_TCP,
  STATE_CONNECTED
};

ConnectionState currentState = STATE_INIT;
unsigned long lastReconnectAttempt = 0;

char networkBuffer[MAX_PACKET_SIZE];
size_t bufferIndex = 0;

// Forward Declarations
void manageNetworkState();
bool ensureTCPConnected();
void readNetworkData();
void extractCSVPayload(char* lineBuffer);
void processPrediction(String intent, String subject, String action, String concept);
void executePulse(int pinTarget);
void handleThumb(String type);
void handleIndex(String subject);
void handleMiddle(String action);
void handleObject(String obj);
void renderWordToGrid(const char* word);
void renderWordToGrid(String word);
void displayBrailleMask(uint8_t mask);
void clearGrid();

void setup() {
  Serial.begin(9600);
  
  unsigned long start = millis();
  while (!Serial && (millis() - start < 2000));
  
  Serial.println(F("\n=============================================="));
  Serial.println(F("ECHO Production TCP Receiver Init Sequence..."));
  Serial.println(F("=============================================="));
  
  // Initialize hardware pins
  pinMode(PIN_THUMB, OUTPUT);
  pinMode(PIN_INDEX, OUTPUT);
  pinMode(PIN_MIDDLE, OUTPUT);
  for (int i = 0; i < 6; i++) {
    pinMode(GRID_PINS[i], OUTPUT);
  }
  
  Serial.println(F("ECHO Actuator Nodes Initialized."));
  
  // Initialize the native communication bridge layer
  Bridge.begin();
  currentState = STATE_INIT;
}

void loop() {
  manageNetworkState();
  
  if (currentState == STATE_CONNECTED) {
    readNetworkData();
  }
}

void manageNetworkState() {
  unsigned long currentMillis = millis();
  
  switch (currentState) {
    case STATE_INIT:
      Serial.println(F("[Bridge] Comm channel ready. Entering TCP pipeline setup..."));
      currentState = STATE_CONNECTING_TCP;
      lastReconnectAttempt = currentMillis - RECONNECT_INTERVAL; 
      break;
      
    case STATE_CONNECTING_TCP:
      if (currentMillis - lastReconnectAttempt >= RECONNECT_INTERVAL) {
        lastReconnectAttempt = currentMillis;
        Serial.print(F("[TCP] Attempting socket connection to server at "));
        Serial.print(SERVER_IP); Serial.print(F(":")); Serial.println(SERVER_PORT);
        
        if (ensureTCPConnected()) {
          Serial.println(F("[TCP] Connected to Python server. Stream ingestion active."));
          currentState = STATE_CONNECTED;
          bufferIndex = 0; 
        } else {
          Serial.println(F("[TCP] Connection failed. Retrying in next interval loop..."));
        }
      }
      break;
      
    case STATE_CONNECTED:
      if (client.connected() == 0) {
        Serial.println(F("[TCP] Connection dropped by remote host. Moving to recovery loop..."));
        client.stop();
        currentState = STATE_CONNECTING_TCP;
        lastReconnectAttempt = currentMillis;
      }
      break;
  }
}

bool ensureTCPConnected() {
  return (client.connect(SERVER_IP, SERVER_PORT) >= 0);
}

void readNetworkData() {
  size_t len = client.available();
  if (len > 0) {
    for (size_t i = 0; i < len; i++) {
      char incomingChar = (char)client.read();
      
      if (incomingChar == '\n') {
        networkBuffer[bufferIndex] = '\0'; 
        
        if (bufferIndex > 0) {
          extractCSVPayload(networkBuffer);
        }
        
        bufferIndex = 0; 
        return;
      }
      else if (incomingChar == '\r') {
        continue;
      }
      else {
        if (bufferIndex < MAX_PACKET_SIZE - 1) {
          networkBuffer[bufferIndex++] = incomingChar;
        } else {
          Serial.println(F("[Error] Line buffer overflow. Discarding corrupt stream data."));
          bufferIndex = 0;
        }
      }
    }
  }
}

void extractCSVPayload(char* lineBuffer) {
  char* tokens[4];
  int tokenCount = 0;
  
  char* token = strtok(lineBuffer, ",");
  while (token != NULL && tokenCount < 4) {
    tokens[tokenCount++] = token;
    token = strtok(NULL, ",");
  }
  
  if (tokenCount != 4) {
    Serial.print(F("[Parsing Error] Expected 4 items, received "));
    Serial.print(tokenCount);
    Serial.println(F(". Dropping packet."));
    return;
  }
  
  processPrediction(
    String(tokens[0]), 
    String(tokens[1]), 
    String(tokens[2]), 
    String(tokens[3])
  );
}

// ==========================================
// Hardware Actuation Mapping Functions
// ==========================================

void processPrediction(String intent, String subject, String action, String concept) {
  intent.trim();
  subject.trim();
  action.trim();
  concept.trim();

  Serial.println(F("\n--- Executing Wireless AI Sequence ---"));
  Serial.print(F("Intent:  ")); Serial.println(intent);
  Serial.print(F("Subject: ")); Serial.println(subject);
  Serial.print(F("Action:  ")); Serial.println(action);
  Serial.print(F("Concept: ")); Serial.println(concept);
  Serial.println(F("--------------------------------------"));
  
  // Actuate pins sequentially based on parsed network arguments
  handleThumb(intent);
  handleIndex(subject);
  handleMiddle(action);
  handleObject(concept);
  
  Serial.println(F("Sequence Complete. Ready for next packet.\n"));
}

void executePulse(int pinTarget) {
  for (int i = 0; i < 10; i++) {
    digitalWrite(pinTarget, HIGH);
    delay(40); 
    digitalWrite(pinTarget, LOW);
    delay(40);
  }
}

void handleThumb(String type) {
  if (type.equalsIgnoreCase("alert")) {
    executePulse(PIN_THUMB);
  } 
  else if (type.equalsIgnoreCase("request")) {
    digitalWrite(PIN_THUMB, HIGH); delay(TIME_LONG);
    digitalWrite(PIN_THUMB, LOW);
  } 
  else if (type.equalsIgnoreCase("question")) {
    digitalWrite(PIN_THUMB, HIGH); delay(TIME_SINGLE);
    digitalWrite(PIN_THUMB, LOW);
  } 
  else if (type.equalsIgnoreCase("statement")) {
    digitalWrite(PIN_THUMB, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_THUMB, LOW); delay(TIME_DOUBLE);
    digitalWrite(PIN_THUMB, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_THUMB, LOW);
  }
  delay(500); 
}

void handleIndex(String subject) {
  if (subject.equalsIgnoreCase("you")) {
    digitalWrite(PIN_INDEX, HIGH); delay(TIME_SINGLE);
    digitalWrite(PIN_INDEX, LOW);
    delay(500);
  } 
  else if (subject.equalsIgnoreCase("i")) {
    digitalWrite(PIN_INDEX, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_INDEX, LOW); delay(TIME_DOUBLE);
    digitalWrite(PIN_INDEX, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_INDEX, LOW);
    delay(500);
  } 
  else if (subject.equalsIgnoreCase("he") || subject.equalsIgnoreCase("she") || 
           subject.equalsIgnoreCase("they") || subject.equalsIgnoreCase("it")) {
    digitalWrite(PIN_INDEX, HIGH); delay(TIME_LONG);
    digitalWrite(PIN_INDEX, LOW);
    delay(500);
  } 
  else {
    char shortcutBuffer[10];
    if (getGrade2Shortcut(subject, shortcutBuffer, sizeof(shortcutBuffer))) {
      renderWordToGrid(shortcutBuffer);
    } else {
      renderWordToGrid(subject);
    }
    delay(1000); 
  }
}

void handleMiddle(String action) {
  if (action.equalsIgnoreCase("need")) {
    digitalWrite(PIN_MIDDLE, HIGH); delay(TIME_LONG);
    digitalWrite(PIN_MIDDLE, LOW);
  } 
  else if (action.equalsIgnoreCase("do")) {
    digitalWrite(PIN_MIDDLE, HIGH); delay(TIME_SINGLE);
    digitalWrite(PIN_MIDDLE, LOW);
  } 
  else if (action.equalsIgnoreCase("move")) {
    digitalWrite(PIN_MIDDLE, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_MIDDLE, LOW); delay(TIME_DOUBLE);
    digitalWrite(PIN_MIDDLE, HIGH); delay(TIME_DOUBLE);
    digitalWrite(PIN_MIDDLE, LOW);
  } 
  else if (action.equalsIgnoreCase("danger")) {
    executePulse(PIN_MIDDLE);
  } 
  delay(500);
}

void handleObject(String obj) {
  char shortcutBuffer[10];
  if (getGrade2Shortcut(obj, shortcutBuffer, sizeof(shortcutBuffer))) {
    renderWordToGrid(shortcutBuffer);
  } else {
    renderWordToGrid(obj);
  }
}

void renderWordToGrid(const char* word) {
  int len = strlen(word);
  for (int i = 0; i < len; i++) {
    char c = tolower(word[i]);
    if (c >= 'a' && c <= 'z') {
      uint8_t mask = BRAILLE_ALPHABET[c - 'a'];
      displayBrailleMask(mask);
      delay(CHAR_DELAY);
      clearGrid();
      delay(200); 
    }
  }
}

void renderWordToGrid(String word) {
  renderWordToGrid(word.c_str());
}

void displayBrailleMask(uint8_t mask) {
  for (int dot = 0; dot < 6; dot++) {
    if ((mask >> dot) & 0x01) {
      digitalWrite(GRID_PINS[dot], HIGH);
    } else {
      digitalWrite(GRID_PINS[dot], LOW);
    }
  }
}

void clearGrid() {
  for (int i = 0; i < 6; i++) {
    digitalWrite(GRID_PINS[i], LOW);
  }
}