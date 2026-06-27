import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc1(x)


# Loss function
def manual_cross_entropy_loss(logits, targets):
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    shifted = logits - max_logits
    log_probs = shifted - torch.log(torch.sum(torch.exp(shifted), dim=1, keepdim=True))

    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    target_log_probs = log_probs[idx, targets]

    return -torch.mean(target_log_probs)


# Hyper Parameters
lr = 1e-4
epochs =  10                                                                                      
batch = 32

# Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classification().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Dataset
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

train_losses, test_losses = [], []
train_accs, test_accs = [], []

print(f"[{device}] Train Start")

for epoch in range(epochs):
    # Train
    model.train()
    total_train_loss, correct_train = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = manual_cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == labels).sum().item()

    avg_train_loss = total_train_loss / len(train_dataset)
    train_acc = correct_train / len(train_dataset)

    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # Test
    model.eval()
    total_test_loss, correct_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = manual_cross_entropy_loss(outputs, labels)
            total_test_loss += loss.item() * len(labels)

            preds = torch.argmax(outputs, dim=1)
            correct_test += (preds == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_dataset)
    test_acc = correct_test / len(test_dataset)

    test_losses.append(avg_test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}")

# Graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='s')
plt.title('Cross-Entropy Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', marker='o')
plt.plot(test_accs, label='Test Accuracy', marker='s')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_result.png', dpi=150)
plt.show()