from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
import torchtext.transforms as T
from torch.utils.data import DataLoader 
import torch
from torch import nn
import time
from transformer import Transformer



# Getting Training -Evaluation-Validation   datasets
train_dataset, valid_dataset, test_dataset = Multi30k(root="data", 
                                                    split=('train', 'valid', 'test'),
                                                    language_pair=('en', 'de')) 


# Tokenizers 
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# Text tokenization for batches using english tokenizer
def en_text_tokenizer(batch):
    # Ensure input is iterable (list or tuple)
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")
    
    return [list(en_tokenizer(text)) for text in batch]

# Text tokenization for batches using german tokenizer

def de_text_tokenizer(batch):
    # Ensure input is iterable (list or tuple)
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")
    
    return [list(de_tokenizer(text)) for text in batch]

#Generators_vocab_builders
def get_engish_vocab_generator(dataset): 
     for src, trg in dataset:
      yield en_tokenizer(src)


def get_german_vocab_generator(dataset):
 for src, trg in dataset:
      yield de_tokenizer(trg)


#English vocab
en_vocab = build_vocab_from_iterator(get_engish_vocab_generator(train_dataset),
                                              min_freq=1, #specials=["<unk>"])  
                                              specials=['<pad>', '<sos>', '<eos>', '<unk>'],  # Special case tokens
                                              special_first=True)  # Place special tokens first in the vocabulary

# Set the default index of the English vocabulary to the index of the '<unk>' token. 
en_vocab.set_default_index(en_vocab['<unk>'])

de_vocab = build_vocab_from_iterator(get_german_vocab_generator(train_dataset),
                                              min_freq=1, #specials=["<unk>"])  
                                              specials=['<pad>', '<sos>', '<eos>', '<unk>'],  # Special case tokens
                                              special_first=True)  # Place special tokens first in the vocabulary

# Set the default index of the German vocabulary to the index of the '<unk>' token. 
de_vocab.set_default_index(en_vocab['<unk>'])

# Encoder's input Transformation
en_text_tranform = T.Sequential( 
    # Using VocabTransform to convert input batch of tokens into corresponding token ids, based on the given vocabulary
    T.VocabTransform(vocab=en_vocab), 
    # Add <sos> at the beginning of each sentence. 1 is used because the index for <sos> in the vocabulary is 1.
    T.AddToken(1, begin=True),
    # Crop the sentence if it is longer than the max length
    T.Truncate(max_seq_len=199),
    # Add <eos> at the end of each sentence. 2 is used because the index for <eos> in the vocabulary is 2.
    T.AddToken(2, begin=False),
    # Convert the list of lists to  tensor. 
   
    T.ToTensor(padding_value=0),
    T.PadTransform(max_length=200, pad_value=de_vocab['<pad>']),  # Εnsuring that all sentences are the same length.
)

# Decoder's input Transformation
de_text_tranform = T.Sequential( 
    # Using VocabTransform to convert input batch of tokens into corresponding token ids, based on the given vocabulary
    T.VocabTransform(vocab=de_vocab), 
    # Add <sos> at the beginning of each sentence. 1 is used because the index for <sos> in the vocabulary is 1.
    T.AddToken(1, begin=True),
    # Crop the sentence if it is longer than the max length
    T.Truncate(max_seq_len=199),
    # Add <eos> at the end of each sentence. 2 is used because the index for <eos> in the vocabulary is 2.
    T.AddToken(2, begin=False),
    # Convert the list of lists to  tensor. 
    T.ToTensor(padding_value=0),
    T.PadTransform(max_length=200, pad_value=de_vocab['<pad>']),  # Εnsuring that all sentences are the same length.
)


# Function to create look-ahead mask
def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)) == 0
    return mask.unsqueeze(0).unsqueeze(0)


# Hyperparameters for  Transformer model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
d_model = 512  
ffn_hidden = 1024 
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
de_vocab_size= len(de_vocab) 
batch_size = 32
epochs = 1 

# Creating the Transformer model
model = Transformer(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,de_vocab_size,en_vocab,de_vocab)



# Loss_function & Optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=0,
                                reduction='none')


optimizer= torch.optim.Adam(model.parameters(), lr=0.001)



# Passing the model to the corresponding device
model.to(device)

# Setting the model on training mode
model.train()

# Creating Dataloader for training 
train_dataloader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,drop_last=True,)


# Training Loop

Start_time_ = time.time() # Start time of Training
for epoch in range(epochs):



 Start_time = time.time() # Timing the training of 100 batches
  
 for batch_idx,(src,trg) in enumerate(train_dataloader): 
      
      
     # Tokenization of batches
      xa = en_text_tokenizer(src)
      ya = de_text_tokenizer(trg)
 
    # Transforming  tokens  to indiced  tensors
      src_tensor = en_text_tranform(xa)
      trg_tensor =  de_text_tranform(ya)
     
    
      padding_mask_x = (src_tensor != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # Encoder padding mask
      padding_mask_y = (trg_tensor != de_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # Decoder padding mask
      
      # Look-ahead mask for the target sequence
      look_ahead_mask = create_look_ahead_mask(trg_tensor.size(1)).to(device)
     

      # Forward pass
      german_predictions = model(src_tensor,     # en_batch,
                                 trg_tensor,     # de_batch,
                                 padding_mask_x, # encoder_self_attention_mask 
                                 look_ahead_mask,# decoder_self_attention_mask
                                 look_ahead_mask # decoder_cross_attention_mask
                                    )
      
    

      # Calculating the loss
      loss = loss_fn(
       german_predictions.view(-1, german_predictions.size(-1)),  
       trg_tensor.view(-1) 
        )

      valid_indices = (trg_tensor.view(-1) != de_vocab['<pad>'])  # Getting the valid indices ignoring the padding tokens
      loss = loss[valid_indices].sum() / valid_indices.sum()  # Comparing the predictions with the correct indiced tokens to get the loss 
      print(loss)
   
      # Zero Grad
      optimizer.zero_grad()
  
    # Backward pass and optimization
      loss.backward()
      optimizer.step() 

 
      if batch_idx >1 and  batch_idx %100 ==0 :
            print('accuracy - loss is : ',loss) 
            print(f"Iteration {batch_idx} : {loss.item()}") 
           
            End_time=time.time()
            # Printing Training time every 100 batches
            print(f'Training time of 100 batches  is : {End_time-Start_time}') 

# Printing Total Training time            
End_time_ = time.time()
print(f'Total Training time is : {End_time_-Start_time_}') 


