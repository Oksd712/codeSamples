import torch
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Normalize the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset.transform = transform
test_dataset.transform = transform

# Split the dataset into training and testing sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearClassifier, self).__init__()

        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.linear(x)
        return x

# Create the model
model = LinearClassifier(28*28, 10)

# Choose a loss function for classification tasks
criterion = torch.nn.CrossEntropyLoss()

# Choose an optimizer for gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create a learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update the learning rate scheduler
    scheduler.step()

    # Print the loss every 100 iterations
    if epoch % 1 == 0:
        print('Epoch: {}/{} | Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Evaluate the model on the testing dataset
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print('Accuracy on testing set: {:.2f}%'.format(accuracy))

evaluate(model, test_loader)
