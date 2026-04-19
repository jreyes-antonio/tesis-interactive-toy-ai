#include "esp_camera.h"
#include <WiFi.h>
#include "wifi_credentials.h" // Importamos tus credenciales seguras

// ==========================================
// Configuración de pines para cámara AI THINKER
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
// ==========================================

void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  
  // Utilizamos formato JPEG para optimizar el envío de paquetes de video
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Si la placa tiene PSRAM instalada
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA; // Calidad media equilibrada (640x480)
    config.jpeg_quality = 10;
    config.fb_count = 2; // Doble buffer para video fluido
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Iniciar la cámara
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Fallo crítico en inicialización de la cámara, error 0x%x", err);
    return;
  }

  // Conexión al Wi-Fi usando la configuración del archivo externo y seguro
  WiFi.begin(ssid, password);
  Serial.println("Conectando al Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("¡Completado! Wi-Fi conectado");

  // Iniciar el servidor
  startCameraServer();

  Serial.print("Servidor de Streaming levantado.\n");
  Serial.print("Usa tu cámara visualizando en el navegador en la siguiente dirección: ");
  Serial.print("http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
}

void loop() {
  // El loop puede quedarse completamente vacío.
  // El microcontrolador trabaja en segundo plano a través del startCameraServer y eventos del ESP-IDF.
  delay(10000);
}

