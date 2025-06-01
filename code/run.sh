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
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5; \
      # Block 5: Transformer nomic-ai/nomic-embed-text-v1.5, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "nomic-ai/nomic-embed-text-v1.5" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 5
    done
  done
done

for dataset in sst; do
  for classifier in singlerotx singleroty singlerotz rot rotcnot; do
    for layers in 1 10; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 5;
    done
  done
done

for dataset in sst; do
  for classifier in svmrbf svmlinear svmpoly logistic randomforest knn mlp; do
    for layers in 1; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 32 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 5;
    done
  done
done

for dataset in sst; do
  for classifier in ensemble_adaboost_rotcnot ensemble_softvoting_qvc ensemble_hardvoting_qvc singlerotx singleroty singlerotz rot rotcnot ensemble_bagging_rotcnot; do
    for layers in 1 10; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done

for dataset in sst; do
  for classifier in svmrbf svmlinear svmpoly logistic randomforest knn mlp; do
    for layers in 1; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done


for dataset in sst; do
  for classifier in maouaki1 maouaki7 maouaki9 maouaki11 maouaki15 ensemble_softvoting_maouaki ensemble_hardvoting_maouaki ensemble_softvoting_qvc_all ensemble_hardvoting_qvc_all; do
    for layers in 1 10; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done

for dataset in sst; do
  for classifier in ensemble_softvoting_classic ensemble_hardvoting_classic; do
    for layers in 1; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done

for dataset in chatgpt_easy chatgpt_medium chatgpt_hard; do
  for classifier in maouaki1 maouaki7 maouaki9 maouaki11 maouaki15 singlerotx singleroty singlerotz rot rotcnot2; do
    for layers in 1 10; do
      # Block 3: Transformer all-mpnet-base-v2, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4; \
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4; \
      # Block 5: Transformer nomic-ai/nomic-embed-text-v1.5, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "nomic-ai/nomic-embed-text-v1.5" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4
    done
  done
done

for dataset in sst; do
  for classifier in maouaki1 maouaki7 maouaki9 maouaki11 maouaki15 singlerotx singleroty singlerotz rot rotcnot2; do
    for layers in 1 10; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done

for dataset in chatgpt_easy chatgpt_medium chatgpt_hard; do
  for classifier in ent1 ent2 ent3 ent4; do
    for layers in 1 10; do
      # Block 3: Transformer all-mpnet-base-v2, n_features 32, n_qubits 5
      # python experiments.py -dataset "$dataset" -model_transformer "all-mpnet-base-v2" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4; \
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4; \
      # Block 5: Transformer nomic-ai/nomic-embed-text-v1.5, n_features 32, n_qubits 5
      # python experiments.py -dataset "$dataset" -model_transformer "nomic-ai/nomic-embed-text-v1.5" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 5 --n_repetitions 30 --n_layers "$layers" --n_qubits 4
    done
  done
done

for dataset in sst; do
  for classifier in ent1 ent2 ent3 ent4; do
    for layers in 1 10; do
      # Block 4: Transformer tomaarsen/mpnet-base-nli-matryoshka, n_features 32, n_qubits 5
      python experiments.py -dataset "$dataset" -model_transformer "tomaarsen/mpnet-base-nli-matryoshka" -n_features 16 -model_classifier "$classifier" --epochs 100 --batch_size 512 --n_repetitions 30 --n_layers "$layers" --n_qubits 4;
    done
  done
done
