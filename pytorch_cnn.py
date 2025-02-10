import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor 
from timeit import default_timer as timer 
import time



# CNN's Architecture
class FashionMNISTModel(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, #the number of channels in the input data (color channels) 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3, 
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.classifier(x)
            return x


# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # Data's file location
    train=True, # get training data (if its true its for training else if its false is for testing)
    download=True, # download data if it doesn't exist on disk (true or false)
    transform=ToTensor(), # images come as PIL format, Therefore using #torchvision.transfroms.ToTensor()  to turn them   into  tensors
    target_transform=None 
)
    
# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor(),
    target_transform=None

)

# Setup the batch size hyperparameter
BATCH_SIZE = 32


# Creating Dataloaders for Training and Evaluation
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
    
)



# Getting class names that correspond to the images 
class_names = train_data.classes



cnn_model = FashionMNISTModel(input_shape=1,    # input_shape= number_of_color_channels (FMNIST's gray pictures have 1 color channel)  
    hidden_units=10, 
    output_shape=len(class_names)).to('cpu')




# Number of Epochs
Epochs=3

# Optimizer
optimizer=torch.optim.SGD(cnn_model.parameters(),lr=0.01)

# Loss function
loss_fn=nn.CrossEntropyLoss()

# Accuracy function
def accuracy_fn(predictions,labels):
   correct=torch.eq(predictions,labels).sum().item()
   acc = (correct / len(predictions)) * 100 
   return acc

# Training function
def train_func() :
   
    cnn_model.to('cpu')

    
    # Calculating the Training time
    Start_time= time.time()
    # Training Loop
    for epoch in range(Epochs):
    # Set model to training mode 
     cnn_model.train()
    
     
    
     # Creating a  nested loop, passing each batch to train the model
     total_train_loss=0
     for batch_idx, (data, target) in enumerate(train_dataloader):
        
        # Forward pass
        predictions=cnn_model(data)
        # Calculating train_loss for each bach
        train_loss=loss_fn(predictions,target)
        # Zero grad
        optimizer.zero_grad()
        # Loss backward
        train_loss.backward()
        # Optimizer step  
        optimizer.step() 
        # Calculating  total train loss
        total_train_loss+=train_loss

    if epoch==Epochs-1:
     End_time=time.time()
     print(f'Total_train_loss is:{total_train_loss}')
     print(f'Training time is : {End_time-Start_time}')
    
# Evaluation function
def eval_func():
   # Setting the model for evaluation
   cnn_model.eval()
   total_test_loss = 0
   test_loss = 0
   
   with torch.no_grad():  # Disable gradient calculation for inference    
    for batch_idx,(data,target) in enumerate(test_dataloader):
     # Forward pass
     predictions=cnn_model(data)
     probabilities = torch.softmax(predictions, dim=1)  # Apply softmax to get probabilities
     predicted_classes = torch.argmax(probabilities, dim=1)  # Get class with highest probability
    
     # Calculating total  loss 
     test_loss = loss_fn(predictions, target)
     total_test_loss += test_loss.item()
    

    acc=accuracy_fn(predicted_classes,target)

    # Printing total_test_loss and accuracy
    print(f'accuracy is :{acc}')
    print(f'total_test_loss is:{total_test_loss}')


train_func()
eval_func()




