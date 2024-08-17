import torch
from car_classification.model import get_model
from car_classification.data import get_data_loaders

def main():
    train_dir = 'path_to_train_data'
    test_dir = 'path_to_test_data'
    
    _, test_loader, num_classes = get_data_loaders(train_dir, test_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes).to(device)
    model.load_state_dict(torch.load('model.pth'))
    
    model.eval()
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (preds == labels).sum().item()
    
    accuracy = corrects / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
