import torch
from torch.utils.data import DataLoader,DistributedSampler
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor 
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from pyspark.ml.torch.distributor import TorchDistributor
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler,Sampler
from pyspark.sql import SparkSession

from  pyspark  import SparkConf

'''
import os
When there isnt a spark-submit script,  this block of code is used

# Setting enviromental variables
os.environ['PYSPARK_PYTHON'] = '/mnt/c/Users/Μακης/Documents/vscode_python/mlib_spark/venv/bin/python'
os.environ['SPARK_HOME'] = '/mnt/c/Spark/spark-3.5.0-bin-hadoop3'
os.environ['MASTER_ADDR'] = 'localhost' #refers to the ip adress of the machine that rank 0 process runs
os.environ['MASTER_PORT'] = '12355'  
os.environ['WORLD_SIZE'] = '2'   # Total number of processes
os.environ['RANK'] = '0'  # Rank of the current process
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["LOCAL_RANK"]='0'  # Refers to the ID of the current CPU (or process) on the local machine. 


# Starting Spark session
spark = SparkSession.builder \
    .appName("PyTorch_CNN_distributed_training") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
'''

# Creating an empty spark in order to define its configurations via Spark-Submit
empty_config = SparkConf()
spark = SparkSession.builder.config(conf=empty_config).getOrCreate() 
executors_n = int(spark.sparkContext.getConf().get('spark.executor.instances'))


# CNN's Architecture
class FashionMNISTModel(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, # the number of channels in the input data (color channels) 
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
    download=True, # download data if it doesn't exist on disk 
    transform=ToTensor(), # images come as PIL format, Therefore using #torchvision.transfroms.ToTensor()  to turn them   into  tensors
    target_transform=None 
)
     
# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None

)

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Create a DistributedSampler to partition unique data to each process
sampler = DistributedSampler(train_data, num_replicas=2, rank=0)  #ensures the dataset is chunked across the cpu cores without any overlapping sample

#Creating Dataloaders for Training and Evaluation
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    sampler=sampler
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False   
)

# Getting  class names that correspond to the images 
class_names = train_data.classes

cnn_model = FashionMNISTModel(input_shape=1,    #input_shape= number_of_color_channels , gray pictures have 1  therefore  its value
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

    # Initializing distributed training
    dist.init_process_group('gloo')
    cnn_model.to('cpu')

    # Implement data parallelism at the module level (each process have a model replica)
    DDP_model=DDP(cnn_model)

    # Calculating the Training time
    Start_time= time.time()
     
    
    # Set model to training mode 
    cnn_model.train()
    # Training Loop
    for epoch in range(Epochs):
       
     # Shuffle data at each epoch 
     sampler.set_epoch(epoch)

     total_train_loss=0 
     # Creating a  nested loop, passing each batch to train the model
     for batch_idx, (data, target) in enumerate(test_dataloader):
        
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
        # Caluclating total train loss 
        total_train_loss+=train_loss
    
     # Getting the Total Training time and total_train_loss
    if epoch==Epochs-1:
     End_time=time.time()
     print(f'Total_train_loss is:{total_train_loss}')
     print(f'Training time is : {End_time-Start_time}')
    # Cleanup the process group 
    dist.destroy_process_group() 
   
    

   
# Evaluationg function
def eval_func(model):
  

    # Setting the model for evaluation
   model.eval()
   total_test_loss = 0
   test_loss = 0
   
   with torch.no_grad():  # Disable gradient calculation for inference     
    for batch_idx,(data,target) in enumerate(test_dataloader):
     # Forward pass
     predictions=model(data)
     probabilities = torch.softmax(predictions, dim=1)  # Applying softmax to get probabilities
     predicted_classes = torch.argmax(probabilities, dim=1)   # Getting class with highest probability
    
     
     #Calculating total_test_loss 
     test_loss = loss_fn(predictions, target)
     total_test_loss += test_loss.item()

  # Calculating  Model's Accuracy 
   acc=accuracy_fn(predicted_classes,target)

   # Printing total_test_loss and accuracy
   print(f'total_test_loss is:{total_test_loss}')
   print(f'accuracy is :{acc}')

# Using TorchDistributor Api to run the training_function via Apache Spark 
distributor = TorchDistributor(
    num_processes=executors_n,  # Number of Spark executors 
    local_mode=False,  # Training on Cluster
    use_gpu=False)
model = distributor.run(train_func) 
spark.stop() # End of Spark Application


