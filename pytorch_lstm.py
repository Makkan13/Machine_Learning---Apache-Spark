import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torch import optim
import time





# Defining a function to calculate model's accuracy (#num_of correct predictions/#labels)
def accuracy_fn(predictions,labels):
   correct=torch.eq(predictions,labels).sum().item()
   acc = (correct / len(predictions)) * 100 
   return acc



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the hyperparameters

# Number of training epochs
epochs = 3 
 # Batch size for training
batch_size = 32 
# Maximum length of input sequences
max_len = 128  
# Define the size of the hidden layer and number of LSTM layers
hidden_size = 32
# Define the size of the hidden layer and number of LSTM layers
num_layers = 2
# Number of classes for classification
output_dim = 4 
# Learning rate for the optimizer
learning_rate = 0.001  

#Using  dataset AG_NEWS from  torchtext.datasets for training-testing 
train_dataset =  AG_NEWS(root = 'data', split='train')  # returns datapipe
test_dataset = AG_NEWS(root = 'data', split='test')# returns datapipe  


# Defining the tokenizer that will convert the senteces to tokens  (basic_english : each word is a token )
tokenizer = get_tokenizer('basic_english')  


# Vocabulary building generator
def yield_tokens (data_iter):   
    for _,text in data_iter:
        yield tokenizer(text)


#Vocabulary dictionary
vocab = build_vocab_from_iterator(yield_tokens(train_dataset),
        min_freq=1, #specials=["<unk>"])  
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],  # Special case tokens
        special_first=True)  # Place special tokens first in the vocabulary

# Set the default index of the vocabulary to the index of the '<unk>' token. 
vocab.set_default_index(vocab['<unk>'])


text_tranform = T.Sequential( 
    # Using VocabTransform to convert input batch of tokens into corresponding token ids, based on the given vocabulary
    T.VocabTransform(vocab=vocab), 
    # Add <sos> at the beginning of each sentence. 1 is used because the index for <sos> in the vocabulary is 1.
    T.AddToken(1, begin=True),
    # Crop the sentence if it is longer than the max length
    T.Truncate(max_seq_len=max_len),

    # Add <eos> at the end of each sentence. 2 is used because the index for <eos> in the vocabulary is 2.
    T.AddToken(2, begin=False),

    # Convert the list of lists to  tensor. 
    T.ToTensor(padding_value=0) 
)

# text tokenization for batches using get_tokinzer 
text_tokenizer = lambda batch: [tokenizer(x) for x in batch] #list comprehension creates  a list of all the sentences in each batch

# Creating data loaders for training and testing
data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , drop_last=True)
data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False )



class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size, output_size, num_layers=2): 
        super(LSTM, self).__init__()
        
        # Create an embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings =vocab_size, embedding_dim= embedding_dim, padding_idx = vocab['0'] )   
        
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.5) # batch_first: When  True  the input shape is (batch_size, sequence_length, input_size).
        
        # Define the output fully connected layer
        self.fc_out = nn.Linear(32, output_size) 
         # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)  


    def forward(self, input_seq, hidden_in, mem_in):
        
        # Convert token indices to dense vectors
        input_embs = self.embedding(input_seq)
        # Pass the embeddings through the LSTM layer
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
        # Pass the LSTM output through the fully connected layer to get the final output
        return self.fc_out(output), hidden_out, mem_out




text_classifier = LSTM(vocab_size = len(vocab),embedding_dim = hidden_size,hidden_size=hidden_size,output_size = output_dim)  

# Setting Optimizer-Loss_function
optimizer = optim.Adam(text_classifier.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() #criterion

# Training function
def train_func() :

   # Set the model to training mode
    text_classifier.train()  
    Start_time = time.time()
   # Training loop    
    for epoch in range (epochs):
        
        
        
        # Creating a  nested loop, passing each batch to train the model
        for batch_idx, (labels, texts)   in enumerate(data_loader_train):
       
         #Batch_size
         bs = labels.shape[0]
      
          
         text_tokens = text_tranform(text_tokenizer(texts)).to(device)
         
         labels = (labels - 1).to(device) # making the labels start from 0 to 3

         # Initializing hidden and memory states
         hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
         memory = torch.zeros(num_layers, bs, hidden_size, device=device)
        
         # Forward pass through the model
         pred, hidden, memory = text_classifier(text_tokens, hidden, memory)

         # Calculate the loss
         loss = loss_fn(pred[:, -1, :], labels)
            
         # Zero Grad
         optimizer.zero_grad()
         # Backpropagation 
         loss.backward()
         # Optimizer step
         optimizer.step()
        
        
        
         if batch_idx % 100 ==0 : 
          # Select the last time step for each sequence
          final_predictions = pred[:, -1, :]  

          # Convert logits to predicted classes
          predicted_classes = final_predictions.argmax(dim=1)  
          
          # Calculating mode's Accuracy
          acc=accuracy_fn(predicted_classes,labels)
          print(acc)
    
    #Printing the total Training time
    if epoch == epochs-1:
      End_time=time.time()
     
      print(f'Training time is : {End_time-Start_time}')  

train_func() 















