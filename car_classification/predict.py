# import torch
# import cv2
# from car_classification.model import get_model
# from car_classification.data import transform

# def main():
#     model_path = 'model.pth'
#     model = get_model(num_classes=196)  # Set the correct number of classes
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         image = transform(frame).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
#         with torch.no_grad():
#             output = model(image)
#             _, predicted = torch.max(output, 1)
        
#         label = train_dataset.classes[predicted.item()]
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         cv2.imshow('Car Detection', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


import torch
import cv2
import torchvision.transforms as transforms
from car_classification.model import get_model
from PIL import Image

def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size expected by the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = preprocess_frame(frame).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main():
    num_classes = 196  # Number of car classes in your dataset
    model_path = 'path/to/your/trained/model.pth'
    
    model = load_model(model_path, num_classes)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Dictionary mapping class indices to car names
    class_names = {0: '2012 Tesla Model S', 1: '2012 BMW M3 Coupe',}  # Add all 196 classes
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the class of the car in the frame
        predicted_class = predict(model, frame)
        car_name = class_names.get(predicted_class, "Unknown")
        
        # Display the prediction on the frame
        cv2.putText(frame, car_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Car Model Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
