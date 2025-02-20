import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# === 1. Параметры ===
BATCH_SIZE = 32
IMAGE_SIZE = 128  # Размер картинок 128x128
NUM_CLASSES = 3  # 3 класса (Stalker, Minecraft, Реальное фото)
CONFIDENCE_THRESHOLD = 0.9  # Новый, более строгий порог уверенности
SAVE_DIR = "lastversion"

# === 2. Загрузка и предобработка данных ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Масштабирование
    transforms.ToTensor(),  # Перевод в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация
])

train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# === 3. Определение модели ===
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# === 4. Инициализация модели ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Классы и их индексы:", train_dataset.class_to_idx)

# === 5. Обучение модели ===
EPOCHS = 200
SAVE_EVERY = 10  # Сохраняем модель каждые 10 эпох

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

    # Сохранение модели каждые 10 эпох
    if (epoch + 1) % SAVE_EVERY == 0:
        model_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Модель сохранена: {model_path}")

# === 6. Оценка точности модели ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")


# === 7. Улучшенная функция предсказания ===
def predict_image(image_path, model, confidence_threshold=CONFIDENCE_THRESHOLD):
    model.eval()

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Преобразование

    with torch.no_grad():
        outputs = model(image)

        # Применяем температурное масштабирование
        T = 2.0  # Чем выше, тем мягче распределение
        probabilities = F.softmax(outputs / T, dim=1)

        # Сортируем вероятности
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        # Берём топ-1 и топ-2 вероятности
        top1_prob = sorted_probs[0][0].item()
        top2_prob = sorted_probs[0][1].item()

        # Если уверенность ниже абсолютного порога
        if top1_prob < confidence_threshold:
            return "Не уверен, что это Stalker, Minecraft или реальное фото.", top1_prob

        # Если разница между топ-1 и топ-2 слишком маленькая (менее 0.2), значит модель не уверена
        if top1_prob - top2_prob < 0.2:
            return "Не уверен, слишком похожи на два класса.", top1_prob

        # Классы
        class_names = ['Minecraft', 'Реальное фото', 'Stalker']

        return class_names[sorted_indices[0][0].item()], top1_prob


# === 8. Пример использования ===
image_path = 'random_foto/fortnite.jpg'
label, conf = predict_image(image_path, model)
print(f"Класс: {label}, Уверенность: {conf:.2f}")
