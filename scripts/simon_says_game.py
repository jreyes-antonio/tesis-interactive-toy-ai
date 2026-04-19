import os
import time
import random
import threading
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import pyttsx3

# Configuración
STREAM_URL = "http://192.168.31.106:81/stream"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'color_classifier.pth')

def get_valid_classes():
    valid_classes = []
    class_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for cls in class_dirs:
        if len(os.listdir(os.path.join(DATA_DIR, cls))) >= 30:
            valid_classes.append(cls)
    valid_classes.sort()
    return valid_classes

# Inicializar motor de voz
try:
    engine = pyttsx3.init()
    # Ajustar para voz en español (si está instalada en Windows)
    voices = engine.getProperty('voices')
    for voice in voices:
        if "es" in voice.id.lower() or "spanish" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
except:
    engine = None

def speak(text):
    if engine is None: return
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    # Usamos hilos para que el juego no se congele mientras la computadora habla
    threading.Thread(target=run_speech).start()

def load_ai_model(classes):
    print("Cargando el cerebro neuronal (MobileNetV2)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Cargamos la estructura
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    # Adaptamos la última capa al número de colores que leyó válidos
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    # Inyectamos los pesos matemáticos
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() # Modo evaluación (Inferencia pura)
    return model, device

def main():
    print("==================================================")
    print("     INICIANDO JUEGO DE IA - SIMON DICE")
    print("==================================================")

    classes = get_valid_classes()
    if len(classes) == 0:
        print("ERROR: No hay clases válidas. ¡Entrena el modelo primero!")
        return
        
    model, device = load_ai_model(classes)
    
    # Mismo pipeline exacto usado en el entrenamiento (sin aumento de datos)
    transform = transforms.Compose([
        transforms.ToPILImage(), # Conversor raw a PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Conectando al feed de la ESP32: {STREAM_URL}")
    try:
        stream = urllib.request.urlopen(STREAM_URL)
    except Exception as e:
        print(f"Error Crítico: No se pudo conectar a la cámara. {e}")
        return

    bytes_data = b''
    
    # Motor de estados lógicos
    target_color = random.choice(classes)
    score = 0
    state = "INICIAR"
    frames_in_state = 0
    
    speak("Bienvenido a Simón Dice Inteligente. Vamos a jugar un rato.")
    time.sleep(2)
    state = "NUEVA_RONDA"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        try:
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    h, w = frame.shape[:2]
                    cx, cy = w//2, h//2
                    box_size = 150 # Área de escaneo de 300x300 en el medio
                    
                    # Dibujar cuadro visor para el usuario
                    cv2.rectangle(frame, (cx-box_size, cy-box_size), (cx+box_size, cy+box_size), (0, 255, 0), 2)
                    
                    # Recortar zona interior (Aísla ruido externo)
                    roi = frame[cy-box_size:cy+box_size, cx-box_size:cx+box_size]
                    
                    # PREDICCIÓN CON INTELIGENCIA ARTIFICIAL
                    # cvtColor arregla el factor de color entre OpenCV (BGR) a PyTorch (RGB)
                    tensor_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(tensor_rgb).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        max_prob, predicted = torch.max(probabilities, 0)
                        
                        predicted_color = classes[predicted.item()]
                        prob_percent = max_prob.item() * 100
                        
                    # Panel HUD Izquierdo
                    cv2.putText(frame, f"IA Ve: {predicted_color.upper()}", (10, 30), font, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, f"Certeza: {prob_percent:.1f}%", (10, 60), font, 0.6, (200, 200, 200), 1)
                    cv2.putText(frame, f"Puntos: {score}", (10, h - 20), font, 0.9, (0, 255, 255), 2)
                    
                    # Máquina de estados del flujo de juego
                    if state == "NUEVA_RONDA":
                        target_color = random.choice(classes)
                        msg = f"Simon dice, muestrame algo de color {target_color}"
                        print(f"JUEGO: {msg}")
                        speak(msg)
                        frames_in_state = 0
                        state = "JUGANDO"
                        
                    elif state == "JUGANDO":
                        cv2.putText(frame, f"OBJETIVO: {target_color.upper()}", (cx - 110, cy - 165), font, 0.8, (0, 0, 255), 2)
                        
                        # Debe de atinar al color pedido con una confianza altísima (>85%)
                        if predicted_color == target_color and prob_percent > 85.0:
                            frames_in_state += 1
                            # Exigimos que la IA perciba el objeto al menos 10 fotogramas continuos para evitar falsos "parpadeos"
                            if frames_in_state > 10: 
                                score += 1
                                msg = f"¡Excelente! Super bien hecho, eso es color {target_color}."
                                print(msg)
                                speak(msg)
                                state = "ACIERTO"
                                frames_in_state = 0
                        else:
                            # Castigamos reiniciar el contador si mueve el objeto fuera de la lente
                            frames_in_state = 0 
                            
                    elif state == "ACIERTO":
                        cv2.putText(frame, "¡CORRECTO!", (cx - 90, cy), font, 1.2, (0, 255, 0), 3)
                        frames_in_state += 1
                        if frames_in_state > 60: # Pausa visual de unos fotogramas (aprox. 3 a 5 seg de delay de festejo)
                            state = "NUEVA_RONDA"

                    cv2.imshow("Cerebro CNN - Simon Dice (Presiona 'q' para salir)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Terminando el juego de forma segura...")
                        break
        except Exception as e:
            print("Perdida de red del stream:", e)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
