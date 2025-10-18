import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# Definicion de la red rectangular
class FashionMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_hidden_layers=2, num_classes=10):
        super(FashionMLP, self).__init__()
        capas = []
        capas.append(nn.Linear(input_size, hidden_size))
        capas.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            capas.append(nn.Linear(hidden_size, hidden_size))
            capas.append(nn.ReLU())
        capas.append(nn.Linear(hidden_size, num_classes))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.red(x)

# Definicion de la red convolucional
class FashionConvNet(nn.Module):
    def __init__(self, num_hidden_layers=2, num_classes=10):
        super(FashionConvNet, self).__init__()
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        entrada_fc = 32 * 7 * 7
        interpolado = np.linspace(784, 10, num_hidden_layers + 2)[1:-1].astype(int).tolist()
        capas = []
        actual = entrada_fc
        for h in interpolado:
            capas.append(nn.Linear(actual, h))
            capas.append(nn.ReLU())
            actual = h
        capas.append(nn.Linear(actual, num_classes))
        self.clasificador = nn.Sequential(*capas)

    def forward(self, x):
        x = self.caracteristicas(x)
        x = x.view(x.size(0), -1)
        return self.clasificador(x)

# Cargar datos
def cargar_datos(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Entrenamiento
def entrenar(modelo, train_loader, criterio, optimizador, num_epochs):
    for epoca in range(num_epochs):
        modelo.train()
        perdida_acumulada = 0.0
        for imagenes, etiquetas in train_loader:
            optimizador.zero_grad()
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item()
        print(f"Epoca [{epoca+1}/{num_epochs}], Perdida promedio: {perdida_acumulada/len(train_loader):.4f}")

# Evaluacion
def evaluar(modelo, test_loader):
    modelo.eval()
    correctas = 0
    total = 0
    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            salidas = modelo(imagenes)
            _, predicho = torch.max(salidas.data, 1)
            total += etiquetas.size(0)
            correctas += (predicho == etiquetas).sum().item()
    precision = 100 * correctas / total
    print(f"Precision en el conjunto de prueba: {precision:.2f}%")

# Guardar modelo
def guardar_modelo_completo(modelo, nombre_archivo, tipo, num_hidden_layers, hidden_nodes):
    datos = {
        'modelo': modelo.state_dict(),
        'tipo': tipo,
        'num_hidden_layers': num_hidden_layers,
        'hidden_nodes': hidden_nodes
    }
    torch.save(datos, nombre_archivo)
    print(f"Modelo guardado en '{nombre_archivo}'.")

# Cargar modelo
def cargar_modelo_completo(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        print("Archivo no encontrado.")
        return None

    datos = torch.load(nombre_archivo)
    tipo = datos['tipo']
    num_hidden_layers = datos['num_hidden_layers']
    hidden_nodes = datos['hidden_nodes']

    if tipo == "convolucional":
        modelo = FashionConvNet(num_hidden_layers=num_hidden_layers)
    else:
        modelo = FashionMLP(hidden_size=hidden_nodes, num_hidden_layers=num_hidden_layers)

    modelo.load_state_dict(datos['modelo'])
    modelo.eval()
    print(f"Modelo '{nombre_archivo}' cargado exitosamente.")
    return modelo
# Menu principal
def main():
    modelo = None
    train_loader, test_loader = None, None

    while True:
        print("\nMenu Principal")
        print("1. Crear nueva red")
        print("2. Cargar red desde archivo")
        print("3. Entrenar red")
        print("4. Evaluar red")
        print("5. Guardar red")
        print("6. Salir")

        opcion = input("Seleccione una opcion: ")

        if opcion == "1":
            tipo = input("Tipo de red (rectangular/convolucional) (por defecto rectangular): ") or "rectangular"
            try:
                num_hidden_layers = int(input("Numero de capas ocultas (por defecto 2): ") or 2)
                hidden_nodes = int(input("Nodos por capa oculta (solo rectangular) (por defecto 512): ") or 512)
                batch_size = int(input("Tamaño del batch (por defecto 64): ") or 64)
                learning_rate = float(input("Tasa de aprendizaje (por defecto 0.001): ") or 0.001)
            except ValueError:
                print("Entrada invalida.")
                continue

            train_loader, test_loader = cargar_datos(batch_size)
            if tipo.lower().startswith("conv"):
                modelo = FashionConvNet(num_hidden_layers=num_hidden_layers)
            else:
                modelo = FashionMLP(hidden_size=hidden_nodes, num_hidden_layers=num_hidden_layers)
            criterio = nn.CrossEntropyLoss()
            optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)
            print("Red creada exitosamente.")

        elif opcion == "2":
            archivo = input("Ingrese el nombre del archivo (.pth): ")
            modelo = cargar_modelo_completo(archivo)
            if modelo:
                # Recuperar metadatos del archivo
                datos = torch.load(archivo)
                tipo = datos['tipo']
                num_hidden_layers = datos['num_hidden_layers']
                hidden_nodes = datos['hidden_nodes']
                batch_size = 64  

                train_loader, test_loader = cargar_datos(batch_size)
                criterio = nn.CrossEntropyLoss()
                learning_rate = 0.001  
                optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

        elif opcion == "3":
            if modelo and train_loader:
                try:
                    num_epochs = int(input("Numero de epocas (por defecto 10): ") or 10)
                except ValueError:
                    print("Entrada invalida.")
                    continue
                entrenar(modelo, train_loader, criterio, optimizador, num_epochs)
            else:
                print("Primero debe crear o cargar una red.")

        elif opcion == "4":
            if modelo and test_loader:
                evaluar(modelo, test_loader)
            else:
                print("Primero debe crear o cargar una red.")

        elif opcion == "5":
            if modelo:
                archivo = input("Nombre del archivo para guardar (.pth): ")
                guardar_modelo_completo(modelo, archivo, tipo, num_hidden_layers, hidden_nodes)
            else:
                print("No hay red para guardar.")

        elif opcion == "6":
            print("Saliendo del programa.")
            break

        else:
            print("Opcion invalida. Intente de nuevo.")

main()