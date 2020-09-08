# Fully Decentralized Joint Learning of Personalized Models and Collaboration Graphs

**python3 project**

source code of AISTATS 2020 [paper](http://proceedings.mlr.press/v108/zantedeschi20a.html).

## Dependencies

1. Install required python modules:
``` bash
 pip install -r requirements.txt
```

## Experiments

[notebooks](https://github.com/vzantedeschi/Dada/tree/master/notebooks) directory contains the code for training and testing the proposed models on several datasets.

The proposed algorithm and its variants are implemented [here](https://github.com/vzantedeschi/Dada/blob/master/src/optimization.py).

## Cite
If you find this work useful, please cite the original paper:

```
@InProceedings{pmlr-v108-zantedeschi20a, 
              title = {Fully Decentralized Joint Learning of Personalized Models and Collaboration Graphs}, 
              author = {Zantedeschi, Valentina and Bellet, Aur\'elien and Tommasi, Marc}, 
              pages = {864--874}, 
              year = {2020}, 
              editor = {Silvia Chiappa and Roberto Calandra}, 
              volume = {108}, 
              series = {Proceedings of Machine Learning Research}, 
              address = {Online}, 
              month = {26--28 Aug}, 
              publisher = {PMLR}, 
              pdf = {http://proceedings.mlr.press/v108/zantedeschi20a/zantedeschi20a.pdf}, 
              url = {http://proceedings.mlr.press/v108/zantedeschi20a.html}, 
              abstract = {We consider the fully decentralized machine learning scenario where many users with personal datasets collaborate to learn models through local peer-to-peer exchanges, without a central coordinator. We propose to train personalized models that leverage a collaboration graph describing the relationships between user personal tasks, which we learn jointly with the models. Our fully decentralized optimization procedure alternates between training nonlinear models given the graph in a greedy boosting manner, and updating the collaboration graph (with controlled sparsity) given the models. Throughout the process, users exchange messages only with a small number of peers (their direct neighbors when updating the models, and a few random users when updating the graph), ensuring that the procedure naturally scales with the number of users. Overall, our approach is communication-efficient and avoids exchanging personal data. We provide an extensive analysis of the convergence rate, memory and communication complexity of our approach, and demonstrate its benefits compared to competing techniques on synthetic and real datasets.} 
} 
```

