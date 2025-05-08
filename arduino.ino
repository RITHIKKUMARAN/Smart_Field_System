#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

const char* ssid     = "  ";
const char* password = "  ";

const char* serverUrl = "http://IP_ADDRESS:8002/data";

#define DHT_PIN      4
#define MOISTURE_PIN 34

#define DHT_TYPE     DHT22

DHT dht(DHT_PIN, DHT_TYPE);

const char* weatherApiKey = " ";
const char* city          = " ";
const String weatherUrl   = "http://api.openweathermap.org/data/2.5/weather?q=" + String(city) + "&appid=" + String(weatherApiKey);

const float HUMIDITY_DRY_THRESHOLD = 40.0;
const int   ALERT_INTERVAL         = 3600000;

unsigned long lastAlertTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(MOISTURE_PIN, INPUT);
  dht.begin();

  // Configure ADC for moisture sensor
  analogReadResolution(12); // ESP32-WROOM-32E supports 12-bit ADC

  connectWiFi();
}

void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi. Restarting...");
    ESP.restart();
  }
}

void loop() {
  // Reconnect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();
  }

  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT22 sensor!");
    delay(2000);
    return;
  }

  int rawMoisture = analogRead(MOISTURE_PIN);
  float moisture = map(rawMoisture, 4095, 1000, 0, 100); // Adjusted for 12-bit ADC and typical sensor range
  moisture = constrain(moisture, 0, 100);

  bool rainPredicted = checkWeather();

  StaticJsonDocument<200> doc;
  doc["moisture"]      = round(moisture * 10) / 10.0; // Round to 1 decimal
  doc["temperature"]   = round(temperature * 10) / 10.0;
  doc["humidity"]      = round(humidity * 10) / 10.0;
  doc["rain_predicted"] = rainPredicted;
  doc["timestamp"]     = String(millis());

  String jsonData;
  serializeJson(doc, jsonData);

  HTTPClient http;
  Serial.println("Sending data to: " + String(serverUrl));
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(jsonData);
  Serial.println("HTTP Code: " + String(httpCode));
  if (httpCode == HTTP_CODE_OK) {
    Serial.println("Data sent successfully");
  } else {
    Serial.println("Error sending data: " + http.errorToString(httpCode));
  }
  http.end();

  delay(60000); // Wait 60 seconds before next reading
}

bool checkWeather() {
  HTTPClient http;
  http.begin(weatherUrl);
  int httpCode = http.GET();

  bool rainPredicted = false;
  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload);
    if (!error) {
      String weather = doc["weather"][0]["main"];
      rainPredicted = (weather == "Rain");
    } else {
      Serial.println("JSON deserialization failed: " + String(error.c_str()));
    }
  } else {
    Serial.println("Weather API request failed: " + String(httpCode));
  }
  http.end();
  return rainPredicted;
}
