import torch
import torch.optim as optim
from car_classification.model import get_model
from car_classification.data import get_data_loaders

def main():
    train_dir = 'cars_train'
    test_dir = 'cars_test'
    
    train_loader, test_loader, num_classes = get_data_loaders(train_dir, test_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    main()
