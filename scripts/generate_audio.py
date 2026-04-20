import os
import asyncio
import edge_tts

# Directorios
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, '..', 'assets', 'audio')

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Voz seleccionada: es-MX-DaliaNeural (Muy amigable, clara y de tono calmado/joven)
# Otras opciones si prefieres cambiarla despues: es-ES-AlvaroNeural (Hombre), es-MX-JorgeNeural
VOICE = "es-MX-DaliaNeural"

# Diccionario de frases que necesitamos para el Juguete
AUDIOS = {
    "bienvenida": "¡Hola! Bienvenido a Simón Dice. Vamos a jugar un rato.",
    "fin_juego": "Eeeee, terminaste mi juego, felicitaciones lo hiciste super bien.",
    "correcto": "¡Excelente! Super bien hecho.",
    
    # Todas las posibles variaciones de color generadas de manera offline
    "buscar_rojo": "Simon dice, muestrame algo de color rojo",
    "buscar_azul": "Simon dice, muestrame algo de color azul",
    "buscar_verde": "Simon dice, muestrame algo de color verde",
    "buscar_amarillo": "Simon dice, muestrame algo de color amarillo",
    "buscar_blanco": "Simon dice, muestrame algo de color blanco",
    "buscar_negro": "Simon dice, muestrame algo de color negro",
}

async def generate():
    print(f"Generando archivos MP3 Neurales en:\n  -> {os.path.abspath(AUDIO_DIR)}\n")
    for filename, text in AUDIOS.items():
        output_file = os.path.join(AUDIO_DIR, f"{filename}.mp3")
        # Inicia conexión con Microsoft Neural Services para grabar los MP3
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)
        print(f" [OK] Archivo MP3 guardado: {filename}.mp3")
    print("\n¡Listo! Tienes todo el sistema de audio estructurado para tu Tesis.")

if __name__ == "__main__":
    asyncio.run(generate())
