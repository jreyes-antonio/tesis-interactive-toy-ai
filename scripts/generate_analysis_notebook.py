import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(SCRIPT_DIR, '..', 'notebooks', 'Tesis_Analisis_Resultados.ipynb')

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(text.split("\n"))-1 else line for i, line in enumerate(text.split("\n"))]
    })

# --- CONTENIDO DEL NOTEBOOK ---

add_md("# Evaluación Estadística del Modelo Neuronal CNN\nEste cuaderno interactivo (Jupyter Notebook) genera automáticamente las visualizaciones académicas requeridas para la tesis de grado, analizando el rendimiento del modelo clasificador de colores para el Juguete Interactivo basado en ESP32.")

add_md("## 1. Importación de Librerías y Definición de Entorno Científico")
add_code("""import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuración Visual para Gráficas de Tesis (A todo color y alta definición)
sns.set_theme(style="whitegrid", context="talk")

DATA_DIR = '../data/raw'
MODEL_PATH = '../models/color_classifier.pth'
HISTORY_PATH = '../models/training_history.json'""")

add_md("## 2. Curvas de Aprendizaje (Learning Curves)\nLos siguientes gráficos documentan la evolución del Error (Loss) y la Precisión (Accuracy) a través de las iteraciones de la Red.")
add_code("""if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de Función de Pérdida (Loss)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Pérdida en Entrenamiento', linewidth=3)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Pérdida en Validación (Examen Ciego)', linewidth=3)
    ax1.set_title('Evolución del Error (Loss)', fontsize=16, pad=15)
    ax1.set_xlabel('Iteraciones (Epochs)')
    ax1.set_ylabel('Pérdida (Entropía Cruzada)')
    ax1.legend()
    
    # Gráfico de Precisión (Accuracy)
    ax2.plot(epochs, history['train_acc'], '#1f77b4', label='Precisión Entrenamiento', linewidth=3)
    ax2.plot(epochs, history['val_acc'], '#2ca02c', label='Precisión Validación', linewidth=3)
    ax2.set_title('Evolución de Precisión (Accuracy)', fontsize=16, pad=15)
    ax2.set_xlabel('Iteraciones (Epochs)')
    ax2.set_ylabel('Aciertos (%)')
    ax2.set_ylim([0, 105])
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("El archivo de telemetría no se encontró. Asegúrate de ejecutar train_model.py primero.")""")


add_md("## 3. Recreación del Entorno de Validación\nPara poder trazar matrices cruzadas, debemos pedirle a la Inteligencia Artificial que vuelva a tomar el examen completo una vez más y registrar exactamente en dónde se equivocó de color.")
add_code("""# Cargar clases filtradas
valid_classes = []
class_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
for cls in class_dirs:
    if len(os.listdir(os.path.join(DATA_DIR, cls))) >= 30:
        valid_classes.append(cls)
valid_classes.sort()

# Custom loader idéntico al de entrenamiento
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        return valid_classes, class_to_idx

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Cargando dataset y modelo neuronal en memoria...")
dataset = FilteredImageFolder(root=DATA_DIR, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(valid_classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Evaluación realizada sobre {len(all_labels)} imágenes totales.")""")

add_md("## 4. Matriz de Confusión (Confusion Matrix)\nVisualiza qué clases interactúan matemáticamente y cuáles tienen fugas de falsos positivos.")
add_code("""cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
            xticklabels=[c.upper() for c in valid_classes],
            yticklabels=[c.upper() for c in valid_classes],
            cbar_kws={'label': 'Frecuencia'})

plt.title('Matriz de Confusión Sensorial de la Red', fontsize=18, pad=20)
plt.ylabel('Clase de Objeto REAL')
plt.xlabel('Clase PREDICHA por la I.A.')
plt.tight_layout()
plt.show()""")

add_md("## 5. Reporte Académico Final (Métricas Precision y Recall)\nEste reporte desglosa estadísticamente la fiabilidad ponderada según la teoría formal del Machine Learning.")
add_code("""report = classification_report(all_labels, all_preds, target_names=valid_classes)
print(report)""")


notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

note_dir = os.path.dirname(NOTEBOOK_PATH)
if not os.path.exists(note_dir):
    os.makedirs(note_dir)

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("¡Notebook de Estadísticas creado de maravilla!")
