import cv2
import os
import time

# --- CONFIGURACIÓN PRINCIPAL ---
STREAM_URL = "http://192.168.31.106:81/stream"
BASE_DIR = os.path.join("data", "raw")

# Mapeo de teclas a carpetas de colores
KEY_MAPPING = {
    'r': 'rojo',
    'b': 'azul',     # (b) de blue
    'y': 'amarillo', # (y) de yellow
    'g': 'verde',    # (g) de green
    'w': 'blanco',   # (w) de white
    'n': 'negro',    # (n) de black
    'f': 'fondo',    # (f) de fondo/vacio
}

def ensure_directories():
    """Crea las carpetas base de colores si no existen."""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    for _, folder_name in KEY_MAPPING.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def main():
    ensure_directories()
    
    print(f"==================================================")
    print(f" Conectando al flujo de video de ESP32-CAM...")
    print(f" URL: {STREAM_URL}")
    print(f"==================================================")
    
    # Inicia la captura desde la URL del MJPEG
    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print("Error: No se pudo conectar a la cámara. Revisa que esté encendida y en la misma red.")
        return

    print("\n--- INSTRUCCIONES DE CAPTURA ---")
    for key, color in KEY_MAPPING.items():
        print(f" Presiona '{key}' para guardar foto en -> {color}")
    print(" Presiona 'q' o 'ESC' para salir del programa.")
    print("----------------------------------\n")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Se perdió el frame de video. Intentando de nuevo...")
            time.sleep(0.5)
            continue
            
        # Monitorear a tiempo real
        cv2.imshow("Recoleccion de Datos ESP32 (Presiona 'q' para salir)", frame)
        
        # Leer tecla pulsada (esperando 1ms por frame)
        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if key < 256 else ''
        
        # Guardar imagen si es una tecla mapeada
        if key_char in KEY_MAPPING:
            target_color = KEY_MAPPING[key_char]
            folder_path = os.path.join(BASE_DIR, target_color)
            
            # Nombre unívoco por estampa de tiempo
            timestamp = int(time.time() * 1000)
            filename = f"img_{timestamp}.jpg"
            filepath = os.path.join(folder_path, filename)
            
            # Guardado
            cv2.imwrite(filepath, frame)
            
            # Feedback visual y en consola
            print(f" [✓] Guardado en '{target_color}': {filename}")
            
            # Flash visual blanco para notificar que la foto se tomó
            flash = frame.copy()
            flash.fill(255)
            cv2.imshow("Recoleccion de Datos ESP32 (Presiona 'q' para salir)", flash)
            cv2.waitKey(100) # Dejar el flash 100ms
            
        elif key == ord('q') or key == 27:
            # Salir presionando q o Escape
            print("\nSaliendo del programa...")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
