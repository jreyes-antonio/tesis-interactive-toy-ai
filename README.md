# Proyecto de Tesis: Interactive Toy AI

Este repositorio contiene el código fuente para el proyecto de tesis de Jorge Reyes.

## Arquitectura del Proyecto

El prototipo consiste en enviar una señal de video desde una **ESP32-CAM** a una computadora, donde un modelo entrenado con **PyTorch** se encarga de analizar las imágenes y detectar colores.

### 📁 Estructura de Directorios

- `data/`: Almacenamiento de imágenes de entrenamiento y prueba (ignorado en Git por su peso). 
  - `data/raw/`: Imágenes originales.
  - `data/processed/`: Imágenes listas para la red neuronal.
- `notebooks/`: Entornos de prueba en Jupyter para el diseño y entrenamiento del modelo con PyTorch.
- `scripts/`: Herramientas en Python para conectar la cámara, recopilar imágenes de forma masiva, etc.
- `models/`: Dónde se guardan los pesos y modelos (`.pth`, `.pt`) exportados tras el entrenamiento.
- `esp32_firmware/esp32_camera_stream/`: Sketch de Arduino (`.ino`) para que la ESP32 se conecte al Wi-Fi y exponga el video.

## Instalación del Entorno
Para el lado de la computadora:
```bash
pip install -r requirements.txt
```
