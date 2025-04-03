#!/bin/bash
# This script executes experiments with various combinations of parameters.
# The configurations are as follows:
# 1) Transformer "all-mpnet-base-v2" with n_features=768 and n_qubits=10.
# 2) Transformer "all-mpnet-base-v2" with n_features=32 and n_qubits=10.
# 3) Transformer "all-mpnet-base-v2" with n_features=32 and n_qubits=5.
# 4) Transformer "tomaarsen/mpnet-base-nli-matryoshka" with n_features=32 and n_qubits=5.
# 5) Transformer "nomic-ai/nomic-embed-text-v1.5" with n_features=32 and n_qubits=5.
#
# For each of these blocks, experiments will be executed on the following datasets:
#    chatgpt_easy, chatgpt_medium, and chatgpt_hard,
# for the ansatz (model_classifier):
#    singlerotx, singleroty, singlerotz, rot, rotcnot,
# and with n_layers set to 1 and 10.

export TOKENIZERS_PARALLELISM=false

for dataset in chatgpt_easy chatgpt_medium chatgpt_hard; do
  for classifier in singlerotx singleroty singlerotz rot rotcnot; do
    for layers in 1 10; do
      # Block 1: Transformer all-mpnet-base-v2, n_features 768, n_qubits 10
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 768 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 10; \
      # Block 2: Transformer all-mpnet-base-v2, n_features 32, n_qubits 10
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 10; \
      # Block 3: Transformer all-mpnet-base-v2, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5; \
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5; \
      # Block 5: Transformer nomic-ai/nomic-embed-text-v1.5, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "nomic-ai/nomic-embed-text-v1.5" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5
    done
  done
done

for dataset in chatgpt_easy chatgpt_medium chatgpt_hard; do
  for classifier in svmrbf svmlinear svmpoly logistic randomforest knn mlp; do
    for layers in 1; do
      # Block 1: Transformer all-mpnet-base-v2, n_features 768, n_qubits 10
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 768 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 10; \
      # Block 2: Transformer all-mpnet-base-v2, n_features 32, n_qubits 10
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 10; \
      # Block 3: Transformer all-mpnet-base-v2, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5; \
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5; \
      # Block 5: Transformer nomic-ai/nomic-embed-text-v1.5, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "nomic-ai/nomic-embed-text-v1.5" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5
    done
  done
done
