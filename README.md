Repository containing data for the paper "The Peripatetic Hater: Predicting Movement Among Online Hate Communities"

### Descriptions of files:

`Sources_of_hate_subreddits.pdf`: links to lists of potential hate subreddits we obtained to curate the dataset of hate subreddits.

`data/subreddits_filtered_out.txt`: Subreddits we deemed from manual annotation to not be hateful.

`data/subreddits_with_hate_scores_and_categories.csv`: Final list of hate subreddits with final clusterings and the standardized scores for each category of hate speech.

`data/peripatetic_hater_training_sample.csv.gz`: Data used to train the model to predict which categories of hate subreddits a user will participate in. NOTE: as the limit for a single file to upload to Github is 25MB, we provide a subsample of 50,000 users. In the paper, we use a dataset size of 100,000 users. Therefore, the performance for this smaller dataset is lower, though the dataset still achieves ROC-AUC values above the baseline of 0.5. You will have to uncompress this file using `gunzip` as well.

`code/embed_for_model.py`: Generates text embeddings and tensors from the training dataset. The files generated from this script will be used to train the neural network to predict peripatetic haters.

`code/movement_prediction_nn.py`: Trains a neural network model from text embeddings and metadata provided in the training dataset. Run with one of the following three command line arguments: 'parent', 'target', or 'both'. The command line argument 'both' uses all available data for training, while 'parent' and 'target' use features associated with the posts users reply to and the posts users make themselves, respectively. This code generates a CSV file with ROC-AUC values for each random seed.
