import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import pandas as pd
import sys

#Set TRAINING_DATA either to 'parent,' 'target,' or 'both'
TRAINING_DATA = sys.argv[1]

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class HateMigrationNN(nn.Module):

    def __init__(self, num_features=1536, additional_features=6, learning_rate=1e-4, optimizer=torch.optim.AdamW, loss_fn=nn.BCELoss(), device='cpu'):
        super(HateMigrationNN, self).__init__()

        self.fc1 = nn.Linear(num_features, num_features // 2)
        self.fc2 = nn.Linear((num_features // 2) + additional_features, num_features // 4)
        self.fc3 = nn.Linear((num_features//4), 3)

        self.device = device
        self.to(device)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn.to(self.device)

    def linear_pass(self, input, input_dim):
        fc = nn.Linear(input_dim, input_dim // 2)
        return fc(input)

    def forward(self, x, x2):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.6)
        x = torch.cat((x, x2), dim=1)


        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.6)

        x = self.fc3(x)
        output = torch.sigmoid(x)

        return output

    def fit(self, train_loader, valid_loader, results_file, epochs=1000, seed=0):
        for epoch in range(epochs):
            self.train() 

            LOSS_train = 0.0
            PRECISION_train, RECALL_train, F1_train, ACCURACY_train = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
            roc_scores = np.zeros(3)
            full_preds = []
            full_labels = []

            for j, data in enumerate(train_loader, 0):

                # Get the inputs; data is a list of [inputs, labels]
                inputs, input2, labels = data
                inputs = inputs.to(self.device)
                input2 = input2.to(self.device)
                labels = labels.to(self.device)
                labels = labels.float()
                full_labels.append(labels)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward (aka predict)
                outputs = self.forward(inputs, input2)
                predictions = np.array([float(k >= 0.5) for sublist in outputs for k in sublist]).reshape(outputs.shape)
                full_preds.append(outputs)

                # Loss + Backprop
                loss = self.loss_fn(outputs, labels)
                LOSS_train += loss.item()
                loss.backward()
                self.optimizer.step()

                # Metrics
            
            # Average the metrics based on the number of batches
            LOSS_train /= len(train_loader)
            PRECISION_train /= len(train_loader)
            RECALL_train /= len(train_loader)
            F1_train /= len(train_loader)
            ACCURACY_train /= len(train_loader)
            
            #preds_arr = torch.cat(full_preds).detach().numpy()
            #labels_arr = torch.cat(full_labels).detach().numpy()

            #roc_scores = roc_auc_score(labels_arr, preds_arr, average=None)
            
            # Validation set
            self.eval()
            with torch.no_grad():

                LOSS_valid = 0.0
                PRECISION_valid, RECALL_valid, F1_valid, ACCURACY_valid = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
                roc_scores_val = np.zeros(3)
                full_preds_val = []
                full_labels_val = []
                full_input_val = []

                for k, data in enumerate(valid_loader, 0):

                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, input2, labels = data
                    inputs = inputs.to(self.device)
                    input2 = input2.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.float()
                    full_labels_val.append(labels)
                    full_input_val.append(input2)


                    # Forward (aka predict)
                    outputs = self.forward(inputs, input2)
                    predictions = np.array([float(k >= 0.5) for sublist in outputs for k in sublist]).reshape(outputs.shape)
                    full_preds_val.append(outputs)

                    # Loss
                    loss = self.loss_fn(outputs, labels)
                    LOSS_valid += loss.item()

                    # Metrics
                    for l in range(3):
                        (precision, recall, f1, accuracy) = self.get_metrics(predictions[:, l], labels[:, l], for_class=1)
                        PRECISION_valid[l] += precision
                        RECALL_valid[l] += recall
                        F1_valid[l] += f1
                        ACCURACY_valid[l] += accuracy

                # Average the metrics based on the number of batches
                LOSS_valid /= len(valid_loader)
                PRECISION_valid /= len(valid_loader)
                RECALL_valid /= len(valid_loader)
                F1_valid /= len(valid_loader)
                ACCURACY_valid /= len(valid_loader)

                preds_arr_val = torch.cat(full_preds_val).detach().cpu().numpy()
                labels_arr_val = torch.cat(full_labels_val).detach().cpu().numpy()
                input_arr_val = torch.cat(full_input_val).detach().cpu().numpy()

                roc_scores_val = roc_auc_score(labels_arr_val, preds_arr_val, average=None)

			##################
			### STATISTICS ###
			##################


        logging.info('Finished Training')

        roc_scores_subset = []
        for i in range(3):
            labels_s = labels_arr_val[input_arr_val[:, i] == 0][:, i]
            preds_s = preds_arr_val[input_arr_val[:, i] == 0][:, i]
            roc_scores_subset.append(roc_auc_score(labels_s, preds_s))

        with open(results_file, 'a+') as f:
            f.write(str(seed))
            for score in roc_scores_val:
                f.write(',' + str(score))

            for score in roc_scores_subset:
                f.write(',' + str(score))

            f.write('\n')



    def get_metrics(self, preds, labels, for_class=1):
        TP, FP, TN, FN = 0, 0, 0, 0

		# Iterate over all predictions
        for idx in range(len(preds)):
            # If we predicted the sample to be class {for_class}
            if preds[idx] == 1:
                # Then check whether the prediction was right or wrong
                if labels[idx] == 1:
                    TP += 1
                else:
                    FP += 1
            # Else we predicted another class 
            else:
                # Check whether the "not class {for_class}" prediction was right or wrong
                if labels[idx] != 1:
                    TN += 1
                else:
                    FN += 1

        precision = TP/(TP+FP) if TP+FP > 0 else 0 # Of all "class X" calls I made, how many were right?
        recall = TP/(TP+FN) if TP+FN > 0 else 0 # Of all "class X" calls I should have made, how many did I actually make?
        f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0
        accuracy = (TP+TN)/(TP+FP+TN+FN)

        return (precision, recall, f1, accuracy)


    def predict(self, X_input):
        self.eval()
        result = []

        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:
			
            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.float()
                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    predictions = np.array([float(k >= 0.5) for sublist in outputs for k in sublist]).reshape(outputs.shape)

                    if len(result)==0:
                        result = predictions
                    else:
                        result = np.concatenate((result, predictions))

                return np.array([np.argmax(x) for x in result])
        
        # Predict from Numpy array or list
        else:
            
            if type(X_input) == list:
                X_input = np.array(X_input)
        
            if isinstance(X_input, np.ndarray): 
                X_test = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_test):
                        # Get the inputs; data is a list of [inputs] 
                        inputs = data.reshape(1,-1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs)
                        predictions = np.array([[1.0, 0.0] if np.argmax(p.cpu().detach().cpu().numpy())==0 else [0.0, 1.0] for p in outputs])

                        if len(result)==0:
                            result = predictions
                        else:
                            result = np.concatenate((result, predictions))

                    return np.array([np.argmax(x) for x in result])
                            
            else:
                raise("Input must be a dataloader, numpy array or list")


    def predict_proba(self, X_input):
        self.eval()
        result = []
        
        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:

            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.float()
                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    class_proba = outputs.cpu().detach().cpu().numpy()
                    
                    if len(result)==0:
                        result = class_proba
                    else:
                        result = np.concatenate((result, class_proba))

                return result
        
        # Predict from Numpy array or list
        else:
            
            if type(X_input) == list:
                X_input = np.array(X_input)
        
            if isinstance(X_input, np.ndarray):  
                X_input = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_input):
                        # Get the inputs; data is a list of [inputs] 
                        inputs = data.reshape(1,-1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs)
                        class_proba = outputs.cpu().detach().cpu().numpy()

                        if len(result)==0:
                            result = class_proba
                        else:
                            result = np.concatenate((result, class_proba))

                    return result

            else:
                raise("Input must be a dataloader, numpy array or list")


        
if __name__ == '__main__':

    if TRAINING_DATA == 'both':
        context = torch.load('../data/parent_embeddings.pt')
        target = torch.load('../data/target_embeddings.pt')
        target_subreddit = torch.load('../data/target_subreddit_types.pt')
        parent_subreddit = torch.load('../data/parent_subreddit_types.pt')
        
        training_subreddit = torch.cat((target_subreddit, parent_subreddit), dim=1)
        embed_training = torch.cat((target, context), dim=1)
        num_features = 1536
        additional_features = 6

    elif TRAINING_DATA == 'target':
        embed_training = torch.load('../data/target_embeddings.pt')
        training_subreddit = torch.load('../data/target_subreddit_types.pt')
        num_features = 768
        additional_features = 3

    elif TRAINING_DATA == 'parent':
        embed_training = torch.load('../data/parent_embeddings.pt')
        training_subreddit = torch.load('../data/parent_subreddit_types.pt')
        num_features = 768
        additional_features = 3



    
    response = torch.load('../data/response.pt')

    

    results_file = f'../data/peripatetic_hater_prediction_performance_{TRAINING_DATA}.csv'
    with open(results_file, 'w') as f:
        f.write('seed,racist,anti-LGBTQ,misogynistic,racist_subset,anti-LGBTQ_subset,misogynistic_subset\n')

    #test_dataset = TensorDataset(embed_testing, testing_subreddit, testing_y)

    for seed in tqdm(range(50)):
        model = HateMigrationNN(device=dev, num_features=num_features, additional_features=additional_features)
        X_train, X_val, X2_train, X2_val, Y_train, Y_val = train_test_split(
                embed_training,
                training_subreddit,
                response,
                test_size=0.15, 
                random_state=seed)

        X_train, X_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
                X_train,
                X2_train,
                Y_train,
                test_size=0.176, 
                random_state=seed)

        train_dataset = TensorDataset(X_train, X2_train, Y_train)

        val_dataset = TensorDataset(X_val, X2_val, Y_val)
        
        train_loader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = 256 # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        val_loader = DataLoader(
                        val_dataset, # The validation samples.
                        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                        batch_size = 256 # Evaluate with this batch size.
                    )


        model.fit(train_loader, val_loader, results_file, epochs=80, seed=seed)
