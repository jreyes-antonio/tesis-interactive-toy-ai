import cv2
import os
import time
import urllib.request
import numpy as np

# --- CONFIGURACIÓN PRINCIPAL ---
STREAM_URL = "http://192.168.31.106:81/stream"
BASE_DIR = os.path.join("data", "raw")

# Mapeo de teclas a carpetas de colores
KEY_MAPPING = {
    'r': 'rojo',
    'b': 'azul',
    'y': 'amarillo',
    'g': 'verde',
    'w': 'blanco',
    'n': 'negro',
    'f': 'fondo',
}

def ensure_directories():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    for _, folder_name in KEY_MAPPING.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def main():
    ensure_directories()
    
    print(f"\n================ RESUMEN DE DATASET ================")
    for key, color in KEY_MAPPING.items():
        folder_path = os.path.join(BASE_DIR, color)
        count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        print(f" [{color.upper()}]: {count} imágenes")
    print(f"====================================================\n")
    
    print(f"==================================================")
    print(f" Conectando al flujo de video de ESP32-CAM...")
    print(f" URL: {STREAM_URL}")
    print(f"==================================================")
    
    # En Windows, a veces cv2.VideoCapture() no posee el conector necesario para streams
    # web tipo multipart/x-mixed-replace. Usaremos urllib para bajar los fragmentos byte a byte.
    try:
        stream = urllib.request.urlopen(STREAM_URL)
    except Exception as e:
        print(f"Error: No se pudo conectar a la cámara. {e}")
        return

    print("\n--- INSTRUCCIONES DE CAPTURA ---")
    for key, color in KEY_MAPPING.items():
        print(f" Presiona '{key}' para guardar foto en -> {color}")
    print(" Presiona 'q' o 'ESC' para salir del programa.")
    print("----------------------------------\n")

    bytes_data = b''
    
    while True:
        try:
            # Leemos porción de datos del servidor por red
            bytes_data += stream.read(1024)
            
            # Las imágenes JPEG siempre empiezan con ffd8 y terminan en ffd9
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            # Si encontramos un cuadro de imagen completo
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Convertimos de arreglo de bytes puramente a imagen de CV2
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Mostrar la ventanita en tu computadora
                    cv2.imshow("Recoleccion de Datos ESP32 (Presiona 'q' para salir)", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    key_char = chr(key).lower() if key < 256 else ''
                    
                    # Manejo de teclas para guardado
                    if key_char in KEY_MAPPING:
                        target_color = KEY_MAPPING[key_char]
                        folder_path = os.path.join(BASE_DIR, target_color)
                        
                        timestamp = int(time.time() * 1000)
                        filename = f"img_{timestamp}.jpg"
                        filepath = os.path.join(folder_path, filename)
                        
                        cv2.imwrite(filepath, frame)
                        
                        # Contar el total actualizado en la carpeta
                        current_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
                        print(f" [✓] Guardado en '{target_color}': {filename} (Total en clase: {current_count})")
                        
                        # Señal de flashesito visual
                        flash = frame.copy()
                        flash.fill(255)
                        cv2.imshow("Recoleccion de Datos ESP32 (Presiona 'q' para salir)", flash)
                        cv2.waitKey(50)
                        
                    elif key == ord('q') or key == 27:
                        print("\nSaliendo del programa...")
                        break
        except Exception as e:
            print("Se perdió la conexión con el stream:", e)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
