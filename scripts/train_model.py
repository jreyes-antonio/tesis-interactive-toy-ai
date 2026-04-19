import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# Configuración
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'color_classifier.pth')
MIN_IMAGES_REQUIRED = 30 # Ignorar folders con menos fotos para evitar colapsos
BATCH_SIZE = 16
EPOCHS = 10

def main():
    print("==============================================")
    print(" INICIANDO PROCESO DE MACHINE LEARNING")
    print("==============================================")

    # 1. Definir cómo la IA verá y modificará las fotos antes de estudiar (Data Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Giramos un poco las fotos para variedad
            transforms.RandomRotation(15),     # Pequeñas rotaciones
            transforms.ToTensor(),             # La convertimos en números (Tensores)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalización estándar de ImageNet
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Custom wrapper para poder ignorar silenciosamente carpetas vacías/con 10 fotos
    class SmartDataset(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                return self.transform(sample), target
            return sample, target

    # 2. Carga y Filtro Inteligente de Clases
    print("\n[1/5] Cargando y analizando directorio de imágenes...")
    
    # Pre-limpieza: contar antes de leer
    valid_classes = []
    class_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for cls in class_dirs:
        count = len(os.listdir(os.path.join(DATA_DIR, cls)))
        if count >= MIN_IMAGES_REQUIRED:
            valid_classes.append(cls)
        else:
            print(f"  -> Ignorando '{cls}' (Solo tiene {count} imágenes. Requiere {MIN_IMAGES_REQUIRED} mínimo)")
            
    if len(valid_classes) < 2:
        print("\nERROR: ¡Necesitas al menos 2 categorías válidas para enseñar a diferenciar colores!")
        print("Asegúrate de recolectar también tu clase 'fondo' (importantísimo).")
        return

    # Usar un custom loader que fuerza a PyTorch a ni siquiera mirar las carpetas inválidas/vacías
    class FilteredImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            valid_classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
            return valid_classes, class_to_idx

    full_dataset = FilteredImageFolder(root=DATA_DIR)
    
    # Todos los índices cargados ahora son enteramente válidos y pre-filtrados
    valid_idx = list(range(len(full_dataset)))

    # 3. Separación de Datos (75% entrenar, 25% poner a prueba/examen)
    train_idx, val_idx = train_test_split(valid_idx, test_size=0.25, random_state=42)

    # Convertimos los conjuntos a Datasets re-entrenables
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, transform=None):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            path, label = self.base.samples[self.indices[idx]]
            from PIL import Image
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label

    train_dataset = FilteredDataset(full_dataset, train_idx, data_transforms['train'])
    val_dataset = FilteredDataset(full_dataset, val_idx, data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[2/5] Clases admitidas: {valid_classes}")
    print(f"      - Exámenes para Estudiar: {len(train_dataset)} fotos")
    print(f"      - Exámenes de Prueba Ciega: {len(val_dataset)} fotos")

    # 4. Magia Pura: TRANSFER LEARNING
    print("\n[3/5] Descargando cerebro experto pre-entrenado (MobileNetV2)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"      -> Entrenando usando: {device.type.upper()}")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Congelar o fijar el "cerebro principal" para que no olvide cómo ver líneas y formas generales
    for param in model.parameters():
        param.requires_grad = False
        
    # Cambiamos la ultimísima capa (que originalmete identificaba 1000 animales) por nuestra cantidad de colores
    num_ftrs = model.classifier[1].in_features
    # Hay que mapear contra el total ORIGINAL del ImageFolder para que las probabilidades calcen con index
    model.classifier[1] = nn.Linear(num_ftrs, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 5. El Bucle Escolar (Entrenamiento Cíclico)
    print("\n[4/5] Arrancando los ciclos de aprendizaje...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Modo Estudiar
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # Modo Examen Ciego (Validación)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = val_corrects.double() / len(val_dataset)
        
        print(f" - Época {epoch+1}/{EPOCHS} -> Precisión Examen: {val_acc*100:.1f}%")

    # 6. Embalaje
    print("\n[5/5] ¡Entrenamiento completado! Guardando archivo de cerebro en disco...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" -> Modelo guardado en {MODEL_SAVE_PATH}")
    print("LISTO. Ya podemos empezar a programar la lógica del juego Simón Dice.")

if __name__ == '__main__':
    main()
