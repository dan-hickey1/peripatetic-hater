This repository contains code adapted from the Demux-MEmo library (https://github.com/gchochla/Demux-MEmo/blob/master/README.md) adapted to work with the Measuring Hate Speech Corpus.

## Instructions for use

To train a model, run: `python3 experiments/demux.py MHS --model_name bert-base-uncased --root_dir . --train_split train --dev_split dev --model_save --max_length 512 --early_stopping_patience 5 --correct_bias --num_train_epochs 20 --dropout_prob 0.1  --device cuda`


#### Arguments:
- use `--model_save` if you want to save the model
- use `--platform reddit` if you want to train on Reddit data only, otherwise use `--platform all` (default is all)
- use the `--reps` argument to specify how many times you want to train the model; the dataset will be split with a different random seed each time
- to use basic BERT instead of Demux, run python3 `experiments/base_train.py` instead of `experiments/demux.py`

To make inferences about posts using a trained model, run `annotate.py` using the instructions in the Demux repository using `emotion_configs/mhs.json`.

When running `annotate.py`, use the argument `--model-architecture Demux` if you want to make predictions using a Demux model. Otherwise, it will assume a basic BERT model.
