import os
import time
import random
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import pygame

# Configuración
STREAM_URL = "http://192.168.31.106:81/stream"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'color_classifier.pth')
AUDIO_DIR = os.path.join(SCRIPT_DIR, '..', 'assets', 'audio')

def get_valid_classes():
    valid_classes = []
    class_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for cls in class_dirs:
        if len(os.listdir(os.path.join(DATA_DIR, cls))) >= 30:
            valid_classes.append(cls)
    valid_classes.sort()
    return valid_classes

# Motor de Reproducción de MP3 Fijos (Alta velocidad, Cero lag)
pygame.mixer.init()
def play_audio(filename):
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    if os.path.exists(audio_path):
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    else:
        print(f"Advertencia: No se encontró el archivo de audio '{filename}.mp3'")

def load_ai_model(classes):
    print("Cargando el cerebro neuronal (MobileNetV2)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
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
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
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
    max_score = 3
    state = "INICIAR"
    frames_in_state = 0
    
    play_audio("bienvenida")
    time.sleep(3)
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
                    box_size = 150 
                    
                    cv2.rectangle(frame, (cx-box_size, cy-box_size), (cx+box_size, cy+box_size), (0, 255, 0), 2)
                    roi = frame[cy-box_size:cy+box_size, cx-box_size:cx+box_size]
                    
                    tensor_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(tensor_rgb).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        max_prob, predicted = torch.max(probabilities, 0)
                        
                        predicted_color = classes[predicted.item()]
                        prob_percent = max_prob.item() * 100
                        
                    cv2.putText(frame, f"IA Ve: {predicted_color.upper()}", (10, 30), font, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, f"Certeza: {prob_percent:.1f}%", (10, 60), font, 0.6, (200, 200, 200), 1)
                    cv2.putText(frame, f"Puntos: {score}/{max_score}", (10, h - 20), font, 0.9, (0, 255, 255), 2)
                    
                    if state == "NUEVA_RONDA":
                        target_color = random.choice(classes)
                        print(f"JUEGO: Busca el color {target_color}")
                        play_audio(f"buscar_{target_color}")
                        frames_in_state = 0
                        state = "JUGANDO"
                        
                    elif state == "JUGANDO":
                        cv2.putText(frame, f"OBJETIVO: {target_color.upper()}", (cx - 110, cy - 165), font, 0.8, (0, 0, 255), 2)
                        
                        if predicted_color == target_color and prob_percent > 75.0:
                            frames_in_state += 1
                            if frames_in_state > 5: 
                                score += 1
                                print(f"¡Excelente! Super bien hecho, eso es color {target_color}.")
                                play_audio("correcto")
                                state = "ACIERTO"
                                frames_in_state = 0
                        else:
                            frames_in_state = 0 
                            
                    elif state == "ACIERTO":
                        cv2.putText(frame, "¡CORRECTO!", (cx - 90, cy), font, 1.2, (0, 255, 0), 3)
                        frames_in_state += 1
                        if frames_in_state > 60: 
                            if score >= max_score:
                                state = "FIN_JUEGO"
                                frames_in_state = 0
                            else:
                                state = "NUEVA_RONDA"
                    
                    elif state == "FIN_JUEGO":
                        cv2.putText(frame, "¡JUEGO TERMINADO!", (cx - 140, cy), font, 1.2, (255, 0, 255), 3)
                        cv2.putText(frame, "Ganaste todas las rondas", (cx - 130, cy + 50), font, 0.8, (255, 255, 255), 2)
                        if frames_in_state == 0:
                            print(f"JUEGO: Felicitaciones, ganaste.")
                            play_audio("fin_juego")
                            frames_in_state += 1

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
