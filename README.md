# Stealthy Attack Experiments

In detail, the following methods are currently implemented:

## Untargeted Attack

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
| **RandomAttack** | 0.8130 | 0.8108 | 0.8046 | 0.7996 | 0.7862 | 0.7712 |
| **DICE** | 0.8130 | 0.8124 | 0.8012 | 0.7894 | 0.7754 | 0.7666 |
| **Metattack** | 0.8130 | 0.8028 | 0.7328 | 0.5974 | 0.4724 | 0.4072 |
| **PGD** | 0.8130 | 0.8024 | 0.7650 | 0.7252 | 0.6786 | 0.6496 |
| **MinMax** | 0.8130 | 0.8136 | 0.7850 | 0.6972 | 0.5660 | 0.4086 |
| **NodeEmbeddingAttack** | 0.8130 | 0.8104 | 0.8018 | 0.7968 | 0.7954 | 0.7914 |
| **GR-BCD** | 0.8130 |  |  |  |  |  |
| **CLGA** | 0.8130 | 0.8104 | 0.7964 | 0.7820 | 0.7738 | 0.7632 |

CiteSeer
| Method | 0% | 1% | 5% | 10% | 15% | 20% |
| ---------------- | -- | -- | -- | --- | --- | --- |
| **RandomAttack** | 0.7080 | 0.7054 | 0.7020 | 0.6894 | 0.6792 | 0.6726 |
| **DICE** | 0.7080 | 0.7050 | 0.7004 | 0.6852 | 0.6810 | 0.6668 |
| **Metattack** | 0.7080 | 0.6996 | 0.5912 | 0.4606 | 0.3848 | 0.3116 |
| **PGD** | 0.7080 | 0.7060 | 0.6912 | 0.6434 | 0.4994 | 0.4654 |
| **MinMax** | 0.7080 | 0.7036 | 0.6990 | 0.5954 | 0.4654 | 0.3810 |
| **NodeEmbeddingAttack** | 0.7080 | 0.6998 | 0.7020 | 0.6946 | 0.6922 | 0.6950 |
| **GR-BCD** | 0.7080 |  |  |  |  |
| **CLGA** | 0.7080 | 0.6230 | 0.5890 | 0.5630 | 0.5420 | 0.4760 |

## Metrics evaluation
to evaluate edege homophily on Cora attacked by CLGA with budget 1%:
```bash
cd src
python metrics.py ----dataset Cora --attack CLGA --budget 1 --metric edge_homophily
```

