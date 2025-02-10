import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,random_split
import os
from pyspark.sql import SparkSession
import torch.optim as optim 
import time


'''
There is no need for spark in this case.
It is only  used to access local data and not for distributed Training
'''
# Setting enviromental variables
os.environ['PYSPARK_PYTHON'] = '/mnt/c/Users/Μακης/Documents/vscode_python/mlib_spark/venv/bin/python'
os.environ['SPARK_HOME'] = '/mnt/c/Spark/spark-3.5.0-bin-hadoop3'
os.environ['MASTER_ADDR'] = 'localhost' # Refers to the ip address of the machine  that rank 0 process runs
os.environ['MASTER_PORT'] = '12355'  
os.environ['WORLD_SIZE'] = '1'   # Total number of processes
os.environ['RANK'] = '0'  #  Refers to the ID of the current CPU (or process) on the local machine. 
os.environ['OMP_NUM_THREADS'] = '1'

# Starting Spark session
spark = SparkSession.builder \
    .appName("Sequential_PyTorch_MLP") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# MLP's Architecture : 3 Layers  (input data has 4 features and  3 labels)
class Multilayer_perceptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=4, out_features=5)  
        self.layer_2 = nn.Linear(in_features=5,out_features=4)
        self.layer_3= nn.Linear(in_features=4,out_features=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
      
        return self.layer_3(self.sigmoid(self.layer_2( self.sigmoid(self.layer_1(x)))))

# Defining a function to calculate model's accuracy (#num_of correct predictions/#labels)
def accuracy_fn(labels, predictions):
    correct = torch.eq(labels, predictions).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(predictions)) * 100 
    return  acc

# Load training data (spark's MLlib sample data)
train_data = spark.read.format("libsvm")\
    .load("{}/data/mllib/sample_multiclass_classification_data.txt".format(os.environ['SPARK_HOME']))


# Converting the data to pandas dataframe for better handling
pandas_data = train_data.toPandas()  

# Getting the features that are saved as sparse vectors and turning them to tensors 
sparse_vectors = pandas_data['features'].apply(lambda x: torch.tensor(x.toArray(), dtype=torch.float32)) #turning the sparse_vectors to dense
tensor_list = sparse_vectors.tolist() 
X_train = torch.stack(tensor_list) 

# Storing the labels and turning them to tensors
target_data = pandas_data['label'].values 
target_tensor = torch.tensor(data = target_data, dtype=torch.long) # Labels
features_tensor = X_train #4 features  


# Creating the  dataset
dataset = TensorDataset(features_tensor, target_tensor)

# Randomly split the dataset into 60% training and 40% test sets
train_size = int(0.6 * len(dataset)) 
test_size = len(dataset) - train_size  

# Performing the random split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=30,shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

def train_func():
 
# Making device agnostic code
 device = "cuda" if torch.cuda.is_available() else "cpu"
 # Creating and passing the model to device
 multilayer_model = Multilayer_perceptor().to(device)
 # Loss_function for multiclass classification
 loss_fn=nn.CrossEntropyLoss() 
 # Οptimizer 
 optimizer = optim.SGD(multilayer_model.parameters(), lr=0.03)
 # Νumber of epochs
 epochs = 100
 # Set model to training mode 
 multilayer_model.train()  
 # Starting  time  of the training
 start_time = time.time() 
 # Training loop
 for epoch in range(epochs):
  
  
  # Creating a  nested loop, passing each batch to train the model
  for batch in train_loader:
     features, labels = batch 
     # Forward pass (predictions)
     predictions = multilayer_model(features) 
     # Calculate loss 
     loss=loss_fn(predictions, labels)
     # Optimizer zero grad # Zero the gradients
     optimizer.zero_grad()
     # Loss backward
     loss.backward()
     # Optimizer step  
     optimizer.step() 
     
 # Printing Total Training_time
  if(epoch==99):
    end_time = time.time()
    print(f"PyTorch Training Time: {end_time - start_time:.2f} seconds")



 # Evaluation Loop

 # Setting the model on evaluation mode
 multilayer_model.eval()

 # Initializing variable to accumulate total test loss 
 total_test_loss = 0

 with torch.no_grad():  # Disable gradient calculation for inference  
     for batch in test_loader:
      features, labels = batch  
      #Forward pass to get predictions
      predictions = multilayer_model(features)  
      probabilities = torch.softmax(predictions, dim=1)  # Applying softmax to get probabilities
      predicted_classes = torch.argmax(probabilities, dim=1)  # Getting class with highest probability
        
      # Calculating total test_loss  
      test_loss = loss_fn(predictions, labels)
      total_test_loss += test_loss.item()
      print(test_loss) 

     
train_func()
