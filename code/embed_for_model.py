import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import transformers
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import math

tqdm.pandas()
print("Cuda is available: ", torch.cuda.is_available())
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define model
tokenizer = transformers.LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

model = transformers.LongformerModel.from_pretrained("allenai/longformer-base-4096")

model.to(DEV)


def embed_sequence(text):
        
    encoding = tokenizer.encode_plus(text, return_tensors="pt", max_length=4096)

    encoding = encoding.to(DEV)
    
    global_attention_mask = global_attention_mask = torch.tensor([([0]*encoding["input_ids"].shape[-1])]) #make all zeros for local attention, all 1s for all global

    global_attention_mask = global_attention_mask.to(DEV)
    
    encoding["global_attention_mask"] = global_attention_mask
    
    with torch.no_grad():
        o = model(**encoding)
    
    sentence_embedding = o.last_hidden_state[:,0]
    
    return sentence_embedding

def create_row_vector(row):
    return [row[c] for c in row.index]

model_df = pd.read_csv('../data/peripatetic_hater_training_sample.csv', lineterminator='\n')

unk_embedding = embed_sequence('UNK') #placeholder embedding for when the parent text is missing

target_x = torch.empty(size=(len(model_df), 768))

i = 0
for text in tqdm(model_df['text'].to_list()): #embed the target
    target_split = str(text).split(' </s> ')
    valid_posts = [post for post in target_split if post != 'nan'] #clean out missing posts
    final_text = ' </s> '.join(valid_posts)
    if final_text != '':
        target_x[i] = embed_sequence(final_text)

    else:
        target_x[i] = (unk_embedding)

    i += 1



context_x = torch.empty(size=(len(model_df), 768))

i = 0

for text in tqdm(model_df['parent_text'].to_list()):  #embed the parent
    target_split = str(text).split(' </s> ')
    valid_posts = [post for post in target_split if post != 'nan'] #clean out missing posts
    final_text = ' </s> '.join(valid_posts)
    if final_text != '':
        context_x[i] = embed_sequence(final_text)

    else:
        context_x[i] = unk_embedding
        
    i += 1


categories = ['0', '1', '2' ,'3', '4', '7']

y = torch.tensor(np.array(list(model_df[categories].apply(create_row_vector, axis=1))))

y = (y > 0).int()

torch.save(context_x, '../data/parent_embeddings.pt')
torch.save(target_x, '../data/target_embeddings.pt')


torch.save(y, '../data/response.pt')

categories = [0, 1, 2, 3, 4, 7]

subreddit_type_tensor = torch.from_numpy(pd.get_dummies(model_df['og_category'])[categories].values) # get the category the user started in

torch.save(subreddit_type_tensor, '../data/target_subreddit_types.pt')

parent_subreddit_types = torch.from_numpy(model_df[['parent_' + str(category) for category in categories]].values) #get the categories the parents posted in
torch.save(parent_subreddit_types, '../data/parent_subreddit_types.pt')
