# DecAug
This is a repo of Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation.

## Outline

While deep learning demonstrates its strong ability to handle independent and identically distributed (IID) data, it often suffers from out-of-distribution (OoD) generalization, where distribution shift appears when testing. Although various methods are proposed for OoD generalization problem, it has been demonstrated that these methods can only solve one specific distribution shift, such as domain shift or correlation relation extrapolation. In this paper, we propose to disentangle category-related and context-related features from the input data. Category-related features contain causal information of the target object recognition task, while context-related features describe the attributes, styles, backgrounds, or scenes of target objects, causing distribution shifts between training and test data. The decomposition is achieved by orthogonalizing the two gradients (w.r.t. intermediate features) of losses for predicting category and context labels respectively. Furthermore, we perform gradient-based augmentation on context-related features to improve the robustness of the learned representations.

### Prerequisites

Python3.6. and the following packages are required to run the scripts:

- [PyTorch-1.1.0 and torchvision](https://pytorch.org)  

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)


- Dataset: please download the dataset and put images into the folder data/[name of the dataset, NICO or PACS or Color-MNIST]/

- Pre-Trained Weights: please download the pre-trained weights and put the weights in the folder saves/initialization/[resnet18.pth] 

### Code Structure

There are six parts in the code:
 - main.py: the main file of DecAug.
 - dataset: the main codes of dataset preprocessing.
 - dataloader: the codes of the dataloader.
 - model: the main codes of the network.
 - baselines: the main codes of DecAug baselines including CNBB, JiGen, DANN, Cumix, etc.
 - saves: to put the initialized weights.

### Main Hyper-parameters

We introduce the usual hyper-parameters as below. There are some other hyper-parameters in the code, which are only added to make the code general, but not used for experiments in the paper.

#### Basic Parameters

- `dataset`: The dataset to use. For example, `NicoAnimal` or `NicoVehicle` or `pacs` or `cmnist`.

- `backbone_class`: The backbone to use, choose `resnet18`.

#### Optimization Parameters

- `max_epoch`: The maximum number of epochs to train the model, default to `100`

- `lr`: The learning rate, default to `0.01`

- `init_weights`: The path to the init weights

- `batch_size`: The number of inputs for each batch, default to `64`

- `image_size`: The designed input size to preprocess the image, default to `225`

- `prefetch`: The number of workers for dataloader, default to `16`


#### Model Parameters

- `model_type`: The model to use, choose `DecAug`.

- `balance1`: The weight for the category branch regularizer in the paper, default to `0.01`

- `balance2`: The weight for the context branch regularizer in the paper, default to `0.01`

- `balanceorth`: The weight for the orth regularizer in the paper, default to `0.01`

- `perturbation`: The weight for the semantic augmentation in the paper, default to `1`

- `epsilon`: The weight for the orth regularizer in the paper, default to `0.01`

- `targetdomain`: The name of the target test domain, default to `photo`

#### Other Parameters

- `gpu`: To select which GPU device to use, default to `0`.


### References
If you find this work or code useful, please cite:

```
@article{bai2020decaug,
  title={Decaug: Out-of-distribution generalization via decomposed feature representation and semantic augmentation},
  author={Bai, Haoyue and Sun, Rui and Hong, Lanqing and Zhou, Fengwei and Ye, Nanyang and Ye, Han-Jia and Chan, S-H Gary and Li, Zhenguo},
  journal={AAAI},
  year={2021}
}
```


