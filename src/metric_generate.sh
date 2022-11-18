#!/usr/bin/env bash
exps="1 2 3 4 5"
dataSets="Cora CiteSeer"
attacks="CLGA dice metattack minmax nodeembeddingattack pgd random "
budgets="01 05 10 15 20"
metrics="edge_homophily node_homophily class_homophily adjusted_homophily balanced_homophily"
echo "exps","dataSets","attacks","budgets","metrics","result"> file.csv
for exp in $exps; do
  for dataset in $dataSets; do
    for att in $attacks; do
      for budget in $budgets; do
        for metric in $metrics; do
          result=$(python metrics.py --exp $exp --dataset $dataset --attack $att --budget $budget --metric $metric)
#         echo "---------------",$result
          echo $exp,$dataset,$att,$budget,$metric,$result>> file.csv

        done
      done
    done
  done
done
