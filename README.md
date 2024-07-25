# One-shot-but-not-degraded Federated Learning

**Abstract**: Transforming the multi-round vanilla Federated Learning (FL) into one-shot FL (OFL) significantly reduces the communication burden and makes a big leap toward practical deployment. However, we note that existing OFL methods all build on model lossy reconstruction (i.e., aggregating while partially discarding local knowledge in clients' models), which attains one-shot at the cost of degraded inference performance. By identifying the root cause of stressing too much on finding a one-fit-all model, this work proposes a novel one-shot FL framework by embodying each local model as an independent expert and leveraging a Mixture-of-Experts network to maintain all local knowledge intact. A dedicated self-supervised training process is designed to tune the network, where the sample generation is guided by approximating underlying distributions of local data and making distinct predictions among experts. Notably, the framework also fuels FL with flexible, data-free aggregation and heterogeneity tolerance. Experiments on 4 datasets show that the proposed framework maintains the one-shot efficiency, facilitates superior performance compared with 8 OFL baselines(+5.54% on CIFAR-10), and even attains over $\times 4$ performance gain compared with 3 multi-round FL methods, while only requiring less than 85% trainable parameters.

#### Dependencies

- python 3.8 (Anaconda)
- Pytorch 1.10.1
- torchvision 0.11.2
- CUDA 11.4

#### Dataset

- CIFAR-10
- CIFAR-100
- SVHN
- Tiny-ImageNet

#### Code upcoming soon !

#### Citation

```
@inproceedings{intactofl,
  author = {Hui, Zeng and Minrui, Xu and Tongqing, Zhou and Xinyi, Wu and Jiawen, Kang and Zhiping, Cai and Dusit, Niyato},
  title = {One-shot-but-not-degraded Federated Learning},
  year = {2024},
  booktitle = {Proceedings of ACM International Conference on Multimedia~(ACM MM)},
  numpages = {10},
}
```


