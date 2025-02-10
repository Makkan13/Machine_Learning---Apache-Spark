import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import init_process_group,destroy_process_group 
from torch.utils.data import DataLoader, TensorDataset,random_split,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from pyspark.sql import SparkSession
import torch.optim as optim 
from pyspark.ml.torch.distributor import TorchDistributor
import time
from  pyspark  import SparkConf

'''
# Setting enviromental variables
os.environ['PYSPARK_PYTHON'] = '/mnt/c/Users/Μακης/Documents/vscode_python/mlib_spark/venv/bin/python'
os.environ['SPARK_HOME'] = '/mnt/c/Spark/spark-3.5.0-bin-hadoop3'
os.environ['MASTER_ADDR'] = 'localhost' #refers to the ip adress of the machine that rank 0 process runs
os.environ['MASTER_PORT'] = '12355'  
os.environ['WORLD_SIZE'] = '2'   # Total number of processes
os.environ['RANK'] = '0'   # Refers to the ID of the current CPU (or process) on the local machine.
os.environ['OMP_NUM_THREADS'] = '1'
'''

'''
# Starting Spark session
spark = SparkSession.builder \
    .appName("MLP_TORCH_DISTRIBUTOR_TRAINING") \
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

# Converting the training data  to pandas dataframe for better handling
pandas_data = train_data.toPandas()  

# Getting the features that are saved as sparse vectors and turning them to tensors  
sparse_vectors = pandas_data['features'].apply(lambda x: torch.tensor(x.toArray(), dtype=torch.float32)) #turning the sparse_vectors to dense
tensor_list = sparse_vectors.tolist() 
X_train = torch.stack(tensor_list) 

# Storing the labels and turning them to tensor
target_data = pandas_data['label'].values 
target_tensor = torch.tensor(data = target_data, dtype=torch.long) 
features_tensor = X_train #4 features  

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Creating the  dataset
dataset = TensorDataset(features_tensor, target_tensor)

# Randomly split the dataset into 60% training and 40% test sets
train_size = int(0.6 * len(dataset))  
test_size = len(dataset) - train_size  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create a DistributedSampler to partition unique data to each process
sampler = DistributedSampler(train_dataset, num_replicas=2, rank=0) #ensures the dataset is chunked across the cpu cores without any overlapping sample

#Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=30,sampler=sampler)  
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# Training Function
def train_func():
 
 # Creating and passing the model to device
 multilayer_model = Multilayer_perceptor().to(device)

 # Initializing distributed training 
 dist.init_process_group(backend='gloo') 

# Implement data parallelism at the module level (each process has a model replica)
 DDP_model = DDP(multilayer_model)   

 # Loss_function for multiclass classification
 loss_fn=nn.CrossEntropyLoss() 
 # Optimizer 
 optimizer = optim.SGD(DDP_model.parameters(), lr=0.03)
 # Number of epochs
 epochs = 100
 
# Set model to training mode 
 multilayer_model.train()  
# Starting  time  of the training
 start_time = time.time() 

# Training loop 
 for epoch in range(epochs):
 
  # Shuffle data at each epoch 
  sampler.set_epoch(epoch)  # At the beginning of each epoch it  is necessary to make shuffling work properly across multiple epochs
  
  # Creating a  nested loop, passing each batch to train the model
  for batch in train_loader:
     features, labels = batch 
     # Forward pass (predictions)
     predictions = multilayer_model(features) #predictions/outputs
     # Calculate loss (per batch) 
     loss=loss_fn(predictions, labels)
     # Zero the gradients
     optimizer.zero_grad()
     # Loss backward
     loss.backward()
     # Optimizer step  
     optimizer.step() 

 # End time of training
  if(epoch==99):
    end_time = time.time()
    print(f"PyTorch Training Time: {end_time - start_time:.2f} seconds")
 # Cleanup the process group
 dist.destroy_process_group()
  
# Evaluation Loop

 # Setting the model on evaluation mode
 multilayer_model.eval()

# Initialize variable to accumulate total test loss  
 total_test_loss = 0

 with torch.no_grad():  # Disable gradient calculation for inference  
     for batch in test_loader:
      features, labels = batch  

      #Forward pass to get predictions
      predictions = multilayer_model(features) 
       
      probabilities = torch.softmax(predictions, dim=1)  # Apply softmax to get probabilities
      predicted_classes = torch.argmax(probabilities, dim=1)  # Get class with highest probability
      
      # Calculate total_test_loss 
      test_loss = loss_fn(predictions, labels)
      total_test_loss += test_loss.item()

 # Printing the total Test_loss and Accuracy
 if epoch == epochs - 1:
   print("Test loss on last batch:", test_loss.item())
   accuracy=accuracy_fn(labels, predicted_classes)    
   print('accuracy is:',accuracy)

 
# Running the TorchDistributor API to execute the training function with multiple processes  via Apache Spark (distributed training)
distributor = TorchDistributor(
    num_processes=executors_n , # Number of  Spark Executors
    local_mode=True, # Training on Cluster
    use_gpu=False)
model = distributor.run(train_func) 
spark.stop() #End of Spark Application


