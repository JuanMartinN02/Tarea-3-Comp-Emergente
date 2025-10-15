import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Selección del tipo de red
network_type = input("Ingrese el tipo de red (rectangular/convolucional) (por defecto rectangular): ") or "rectangular"

# Configuración de la red
num_hidden_layers = int(input("Ingrese el número de capas ocultas (por defecto 2): ") or 2)
hidden_nodes = int(input("Ingrese el número de nodos por capa oculta (por defecto 512): ") or 512)
num_epochs = int(input("Ingrese el número de épocas de entrenamiento (por defecto 10): ") or 10)
batch_size = int(input("Ingrese el tamaño del batch (por defecto 64): ") or 64)
learning_rate = float(input("Ingrese la tasa de aprendizaje (por defecto 0.001): ") or 0.001)

# Carga del conjunto de datos Fashion MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Definición de las redes ---

# Red totalmente conectada (rectangular)
class FashionMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_hidden_layers=2, num_classes=10):
        super(FashionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

# Red convolucional con interpolación lineal de neuronas
class FashionConvNet(nn.Module):
    def __init__(self, num_hidden_layers=2, num_classes=10):
        super(FashionConvNet, self).__init__()

        # Bloques convolucionales simples
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 x 14 x 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32 x 7 x 7
        )

        # Tamaño de entrada de las capas totalmente conectadas
        input_size = 32 * 7 * 7  # 1568

        # Interpolación lineal de neuronas entre 784 y 10
        interpolated = np.linspace(784, 10, num_hidden_layers + 2)[1:-1].astype(int).tolist()

        # Construcción de las capas totalmente conectadas
        layers = []
        in_size = input_size
        for h in interpolated:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Inicialización del modelo
if network_type.lower().startswith("conv"):
    model = FashionConvNet(num_hidden_layers=num_hidden_layers, num_classes=10)
else:
    model = FashionMLP(input_size=784, hidden_size=hidden_nodes, num_hidden_layers=num_hidden_layers, num_classes=10)

# Definición de la función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Bucle de entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {running_loss/len(train_loader):.4f}")

# Evaluación del modelo
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")