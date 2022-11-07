# Stealthy Attack Experiments

In detail, the following methods are currently implemented:

## Untargetted Attack

| Method | Description | Example |
| ------ | ----------- | ------- |
| **RandomAttack** | A simple random method that chooses edges to flip randomly. |  [[**Example**]](https://github.com/EdisonLeeeee/GreatX/blob/master/examples/attack/targeted/random_attack.py) |
| **DICE** | *Waniek et al.* [Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior'16* | [[**Example**]](https://github.com/EdisonLeeeee/GreatX/blob/master/examples/attack/targeted/dice_attack.py)   |
| **Metattack** | *Zügner et al.* [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), *ICLR'19* | [[**Example**]](https://github.com/EdisonLeeeee/GreatX/blob/master/examples/attack/untargeted/metattack.py) |
| **PGD** | *Xu et al.* [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19* | [[**Example**]](https://github.com/EdisonLeeeee/GreatX/blob/master/examples/attack/untargeted/pgd_attack.py) |
| **Minmax** | *Xu et al.* [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19* | [[**Example**]](https://github.com/EdisonLeeeee/GreatX/blob/master/examples/attack/untargeted/minmax_attack.py) |
| **NodeEmbeddingAttack** | *Bojchevski et al.* [Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093), *ICML'19* | [[**Example**]]() |
| **GR-BCD** | *Geisler et al.* [ Robustness of graph neural networks at scale](https://github.com/sigeisler/robustness_of_gnns_at_scale), *NIPS'21* | [[**Example**]](https://github.com/rinnesz/clga) |
| **CLGA** | *Zhang et al.* [ Unsupervised graph poisoning attack via contrastive loss back-propagation](https://dl.acm.org/doi/abs/10.1145/3485447.3512179), *WWW'22* | [[**Example**]](https://github.com/rinnesz/clga) |

## Detailed implementation process

### CLGA


## Source Code

Stored in the folder "src".


## Cleaned Graph & Poisoned Graph

Stored in the folder "poisoned_graph".

## Accuracy

Cora
| Method | 0% | 1% | 5% | 10% | 15% | 20% |
| ---------------- | -- | -- | -- | --- | --- | --- |
| **RandomAttack** | 0.8160 | 0.8160 | 0.8140 | 0.8090 | 0.7880 | 0.7580 |
| **DICE** | 0.8160 | 0.8140 | 0.8090 | 0.7970 | 0.7690 | 0.7670 |
| **Metattack** | 0.8160 | 0.7970 | 0.6930 | 0.5860 | 0.5139 | 0.4940 |
| **PGD** | 0.8160 | 0.7930 | 0.7820 | 0.7450 | 0.7560 | 0.7700 |
| **MinMax** | 0.8160 | 0.7610 | 0.6720 | 0.5550 |0.3400 | 0.1760 |
| **NodeEmbeddingAttack** | 0.8160 | 0.7640 | 0.7370 | 0.7630 | 0.7550 | 0.7530 |
| **GR-BCD** | 0.8160 |  |  |  |  |  |
| **CLGA** | 0.8160 |  0.8090 | 0.7980 | 0.7830 | 0.7700 | 0.7620 |


CiteSeer
| Method | 1% | 5% | 10% | 15% | 20% |
| ---------------- | -- | -- | --- | --- | --- |
| **RandomAttack** | 0.6470 | 0.6340 | .5040 | 0.5450 | 0.4530 |
| **DICE** | 0.6200 | 0.5730 | 0.5310 | 0.4620 | 0.4180 |
| **Metattack** | 0.5630 | 0.4010 | 0.2600 | 0.2030 | 0.1980 |
| **PGD** | 0.6310 | 0.6540 | 0.5520 | 0.5930 | 0.4360 |
| **MinMax** | 0.6410 | 0.6000 | 0.4220 | 0.1920 | 0.1720 |
| **NodeEmbeddingAttack** | 0.6630 | 0.6870 | 0.6410 | 0.6370 | 0.6060 |
| **GR-BCD** |  |  |  |  |  |
| **CLGA** | 0.6230 | 0.5890 | 0.5630 | 0.5420 | 0.4760 |

## Metrics evaluation
to evaluate edege homophily on Cora attacked by CLGA with budget 1%:
```bash
cd src
python metrics.py ----dataset Cora --attack CLGA --budget 1 --metric edge_homophily
```

