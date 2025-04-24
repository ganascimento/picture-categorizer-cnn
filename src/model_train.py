import torch
import torch.nn as nn
import config
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from image_classification_cnn import ImageClassificationNN
from PIL import Image

torch.classes.__path__ = [os.path.dirname(os.path.abspath(torch.__file__))]


class ModelTrain:
    def __init__(self, load=False):
        self.__set_device()
        self.__load_data()
        self.__get_model(load)

    def train(self, num_epochs=10):
        lossFunction = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = lossFunction(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_value = f"{running_loss/len(self.train_loader):.4f}"

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value}")

            yield (epoch + 1, loss_value)

    def evaluate(self, image):
        self.model.eval()
        image_convert = Image.open(image).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        image_transformed = transform(image_convert)
        image_test = image_transformed.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_test)
            _, predicted = torch.max(output, 1)

        return self.test_dataset.classes[predicted.item()]

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return (correct / total) * 100

    def save_model(self):
        if not os.path.exists(config.DEFAULT_PATH_TRAIN_DATA):
            os.makedirs(config.DEFAULT_PATH_TRAIN_DATA)
        torch.save(self.model.state_dict(), config.TRAIN_DATA_FULL_PATH)
        print(f"Model saved to {config.TRAIN_DATA_FULL_PATH}")

    def __set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chosen device: {self.device}")

    def __load_data(self):
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        # Get dataset
        self.train_dataset = datasets.CIFAR10(
            root=config.DEFAULT_PATH_DATASET, train=True, download=True, transform=train_transform
        )
        self.test_dataset = datasets.CIFAR10(root=config.DEFAULT_PATH_DATASET, train=False, download=True, transform=transform)

        # Divide into batches
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    def __get_model(self, load):
        self.model = ImageClassificationNN().to(self.device)

        if load:
            data_train = torch.load(config.TRAIN_DATA_FULL_PATH)
            self.model.load_state_dict(data_train)
            print(self.validate())
