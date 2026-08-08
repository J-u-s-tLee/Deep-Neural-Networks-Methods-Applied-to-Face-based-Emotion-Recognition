import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import numpy as np


# Suppress specific warning messages
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_excel(csv_file, sheet_name='label', header=0)   # Labels of all the images
        self.root_dir = root_dir # Directory to the images
        self.transform = transform # Transformations applied to the image
        self.classes = 10 # Number of different labels
        
    def __len__(self):
        return len(self.data) # Number of input labels = images
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0]) # Constructs a path for each image
        image = Image.open(img_name) # Open each image
        
        # Convert the image to RGB to input in the model
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Extract the label to the corresponding idx
        label = self.data.iloc[idx, 12]
        
        # Apply transformation to the input images so they can be used as input to the model
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training set
train_dataset = CustomDataset(csv_file='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+\\train\\label.xlsx',
                               root_dir='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+ Landmarks\\train',
                               transform=data_transform)

# Create validation set
validation_dataset = CustomDataset(csv_file='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+\\validation\\label.xlsx',
                               root_dir='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+ Landmarks\\validation',
                               transform=data_transform)

# Create test set
test_dataset = CustomDataset(csv_file='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+\\test\\label.xlsx',
                               root_dir='C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+ Landmarks\\test',
                               transform=data_transform)

# Data loader - batches for manage memory efficiently, shuffles so the model doesn't learn the order
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # len = 893 = total number of batches
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load pre-trained ResNet-18 model (transfer learning)
resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features # Number of input features
resnet18.fc = nn.Linear(num_ftrs, train_dataset.classes)  # Modify the last fully connected layer to match the number of classes


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # Loss function (same as the dataset paper)
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9) # Will update the weights/parameters during training to minimize loss
                                                                     # lr = learning rate, momentum = helps accelerate convergence by averaging 
                                                                     # gradients over time
# Train the model
num_epochs = 20  # Define the number of epochs

for epoch in range(num_epochs):
    
    resnet18.train()
    running_loss = 0.0
    running_train = 0
    running_val = 0
    batch = 0;
    running_train_bal = 0;
    running_val_bal = 0;
    running_test = 0;
    running_test_bal = 0;
    
    for inputs_train, labels_train in train_loader:

        optimizer.zero_grad() # Resets the gradient (slope of loss function for each parameter) of the optimizer
        outputs_train = resnet18(inputs_train) # Pass a batch of 32 images through the model
        _, preds_train = torch.max(outputs_train,1)
        loss = criterion(outputs_train, labels_train) # Calculating the loss using CrossEntropyLoss
        loss.backward() # Computes gradients of the loss function with respect to all parameters in the model, 
                        # enabling the optimization algorithm to adjust the model's parameters during training to minimize the loss.
        optimizer.step() # Updating the model parameters based on the computed gradients, driving the model 
                         # towards minimizing the loss function and improving its performance on the training data.
        
        running_loss += loss.item() # To track cumulative loss across batches
        running_train += metrics.accuracy_score(labels_train, preds_train)
        running_train_bal += metrics.balanced_accuracy_score(labels_train, preds_train)
        batch+=1;
        print(f"\rTraining: {batch * 100 /len(train_loader):.2f}% ", end='', flush=True)

    batch = 0;
    print()
    resnet18.eval()
    with torch.no_grad():
    
        for inputs_val, labels_val in validation_loader:
             
             outputs_val = resnet18(inputs_val)
             _, preds_val = torch.max(outputs_val,1)
             running_val += metrics.accuracy_score(labels_val, preds_val)
             running_val_bal += metrics.balanced_accuracy_score(labels_val, preds_val)
             batch+=1;
             print(f"\rValidation: {batch * 100 /len(validation_loader):.2f}% ", end='', flush=True)
             
    epoch_loss = running_loss / len(train_loader)
    epoch_acc_train = (running_train * 100) / len(train_loader)
    epoch_acc_val = (running_val * 100) / len(validation_loader) # Accuracy da validação diz se o modelo está a generalizar
    epoch_acc_train_bal = (running_train_bal* 100) /  len(train_loader)
    epoch_acc_val_bal = (running_val_bal * 100) / len(validation_loader)
    
    print()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.2f}, Train Acc: {epoch_acc_train:.2f}%, Train Bal Acc: {epoch_acc_train_bal:.2f}%, Validation Acc: {epoch_acc_val:.2f}%, Val Bal Acc: {epoch_acc_val_bal:.2f}%")

print('Training Complete!')

# Lists to store all the labels and predictions
all_labels_test = []
all_preds_test = []

with torch.no_grad():
    batch = 0;
    
    for inputs_test, labels_test in test_loader:
             
             outputs_test = resnet18(inputs_test)
             _, preds_test = torch.max(outputs_test,1)
             running_test += metrics.accuracy_score(labels_test, preds_test)
             running_test_bal += metrics.balanced_accuracy_score(labels_test, preds_test)
             
             # Save labels and predictions for confusion matrix
             all_labels_test.extend(labels_test.numpy())
             all_preds_test.extend(preds_test.numpy())
    
             batch += 1;
             print(f"\rTesting: {batch * 100 /len(test_loader):.2f}% ", end='', flush=True)
             
    Acc_test = (running_test * 100) / len(test_loader)
    Acc_test_bal = (running_test_bal * 100) / len(test_loader)
    print()
    print(f"Test Acc: {Acc_test:.2f}%, Test Bal Acc: {Acc_test_bal:.2f}%")
    
    # Define your class labels
    class_labels = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt', 'Unknown']

    # Compute the confusion matrix
    confusion_matrix = confusion_matrix(all_labels_test, all_preds_test, normalize='true')
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Display the numbers in the confusion matrix cells
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.show()
        
print('Test Complete!')
