
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torch import optim
import torchdata 
from torchdata.datapipes.iter import IterableWrapper,ShardingFilter 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from  pyspark  import SparkConf

'''
import os
When there isnt a spark-submit script,  this block of code is used

NUM_WORKERS = 4
# Setting enviromental variables 
os.environ['PYSPARK_PYTHON'] = '/mnt/c/Users/Μακης/Documents/vscode_python/mlib_spark/venv/bin/python'
os.environ['SPARK_HOME'] = '/mnt/c/Spark/spark-3.5.0-bin-hadoop3'
os.environ['MASTER_ADDR'] = 'localhost' #refers to the ip address of the machine that rank 0 process runs
os.environ['MASTER_PORT'] = '12355'  
os.environ['WORLD_SIZE'] = str(NUM_WORKERS)   # Total number of processes
os.environ['RANK'] = '0'  #  # Refers to the ID of the current CPU (or process) on the local machine.
os.environ['OMP_NUM_THREADS'] = '1'

# Starting Spark session and configuring spark's resources
spark = SparkSession.builder \
    .appName("Pytorch_DDP_Transformer") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.cores", "2") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

'''

# Creating an empty spark in order to define its configurations via Spark-Submit
empty_config = SparkConf()
spark = SparkSession.builder.config(conf=empty_config).getOrCreate() 
executors_n = int(spark.sparkContext.getConf().get('spark.executor.instances'))


# Defining function to calculate model's Accuracy
def accuracy_fn(predictions,labels):
   correct=torch.eq(predictions,labels).sum().item()
   acc = (correct / len(predictions)) * 100 
   return acc

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device to GPU if available, otherwise fallback to CPU

# Defining the model's hyperparameters 
learning_rate = 0.001  # Learning rate for the optimizer
epochs = 3  # Number of training epochs
batch_size = 32  # Batch size for training
max_len = 128  # Maximum length of input sequences 
hidden_size = 32 # Size of the hidden layers
num_layers = 2 # Number of LSTM layers
output_dim = 4 #AG_NEWS has 4 labels 

# Getting  AG_news dataset fron TorchText.datasets  to  train-evaluate the model   
train_dataset =  AG_NEWS(root = 'data', split='train')
test_dataset = AG_NEWS(root = 'data', split='test') 



# Defining the tokenizer that will convert the senteces to tokens  
tokenizer = get_tokenizer('basic_english') 

# Text tokenization of batches using  the tokenizer
text_tokenizer = lambda batch: [tokenizer(x) for x in batch] #list comprehension creates  a list of all the sentences in each batch

# Vocabulary building generator
def yield_tokens (data_iter):   
    for _,text in data_iter:
        yield tokenizer(text)


# Creating Vocabulary, in which   each token corresponds to an indice (dictionary -> tokens:indices) 
vocab = build_vocab_from_iterator(yield_tokens(train_dataset),
        min_freq=1, # Minimum appearances for a token 
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],  # Special case tokens
        special_first=True)  # Place special tokens first in the vocabulary


# If a token is not found in the vocabulary, it will be replaced with the '<unk>' token.
vocab.set_default_index(vocab['<unk>'])

text_tranform = T.Sequential( 
    # Using VocabTransform to convert input batch of tokens into corresponding token ids, based on the given vocabulary
    T.VocabTransform(vocab=vocab), 
    # Adding <sos> at the beginning of each sentence. 1 is used because the index for <sos> in the vocabulary is 1.
    T.AddToken(1, begin=True),
    # Cropping the sentence if it is longer than the max length
    T.Truncate(max_seq_len=max_len),
    # Adding <eos> at the end of each sentence. 2 is used because the index for <eos> in the vocabulary is 2.
    T.AddToken(2, begin=False),
    # Converting the list of sentences  to  tensor.
    T.ToTensor(padding_value=0) 
)

# LSTM's Architecture
class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size, output_size, num_layers=2): 
        super(LSTM, self).__init__()
        
        # Create an embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings =len(vocab), embedding_dim= hidden_size, padding_idx = vocab['0'] )  
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.5) #batch_first: Set True if the input shape is (batch_size, sequence_length, input_size).
        
        # Define the output fully connected layer
        self.fc_out = nn.Linear(32, output_size) 
       

    def forward(self, input_seq, hidden_in, mem_in):
        
        # Convert token indices to dense vectors
        input_embs = self.embedding(input_seq)
        
        # Pass the embeddings through the LSTM layer
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
       
        
        # Pass the LSTM output through the fully connected layer to get the final output
        return self.fc_out(output), hidden_out, mem_out

# Creating  the text_classifier model
text_classifier = LSTM(vocab_size = len(vocab),embedding_dim = hidden_size ,hidden_size=hidden_size,output_size = output_dim )

# Defining optimizer and loss function
optimizer = optim.Adam(text_classifier.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() 



# Wrapping dataset in IterableWrapper to convert it to a IterDataPipe
train_datapipe = IterableWrapper(train_dataset)

# Applying Sharding_filter  to chunk the data in a unique way for each process in distributed training
sharded_datapipe = train_datapipe.sharding_filter()

# Creating  DataLoaders for training 
data_loader_train = DataLoader(train_datapipe, batch_size=batch_size, num_workers=executors_n ,shuffle=True, drop_last=True)

# Defing training_function
def train_func() :
    dist.init_process_group('gloo')
    text_classifier.to('cpu')
    # Implement data parallelism at the module level (each process has a model replica)
    DDP_model=DDP(text_classifier)
    # Set the model to training mode
    text_classifier.train()     
    # Calculating the Training time
    Start_time= time.time()

    for epoch in range (epochs):
      
        train_acc = 0
  
        
       
        for batch_idx, (labels, texts)   in enumerate(data_loader_train):
         
         #batch_size
         bs = labels.shape[0]

         # Performing the tokenization and  converting the tokens to tensors
         text_tokens = text_tranform(text_tokenizer(texts)).to(device)
        
         labels = (labels - 1).to(device)  #making the labels range  from 0 to 3
          # Initialize hidden and memory states
         hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
         memory = torch.zeros(num_layers, bs, hidden_size, device=device)
        
        # Forward pass through the model
         pred, hidden, memory = text_classifier(text_tokens, hidden, memory)

        # Calculate the loss
         loss = loss_fn(pred[:, -1, :], labels)

        # Backpropagation and Optimization
         optimizer.zero_grad() #zero the gradients
         loss.backward()
         optimizer.step()
  
        # Getting the number of correct predicts in each batch
         train_acc += (pred[:, -1, :].argmax(1) == labels).sum() 
    
    # Getting the Total Training time     
    if epoch == epochs-1:
     End_time=time.time()
     print(f'Training time is : {End_time-Start_time}')    

    # Ending the distributed training and clearing up  the resources
    dist.destroy_process_group()




# Using TorchDistributor Api to run the training_function via Apache Spark
distributor = TorchDistributor(
    num_processes=executors_n , # Number of Spark executors 
    local_mode=False, # Training on Cluster
    use_gpu=False) 
model = distributor.run(train_func) 






