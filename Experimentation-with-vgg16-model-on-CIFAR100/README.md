## Experiments Summary

### Experiment 01A: 
**Training with VGG-16 Model from Scratch (CIFAR-100 Dataset)**

- **Device used**: Nvidia RTX 3070ti 8GB
- **Framework**: PyTorch
- **Augmentation used**: 
  - Random resized crop
  - Random horizontal flip
  - Random rotation
  - Random affine
- **Batch size**: 64
- **Optimizer used**: Adam (lr:0.0001)
- **Regularization techniques used**:
  - Early stopping with patience 10
  - Weight decay
- **Others**:
  - In this study, the ReduceLROnPlateau learning rate scheduler was employed to dynamically adjust the learning rate based on the validation loss with patience 5.
  - The He initializer was utilized to initialize the network weights to maintain a good distribution and improve convergence during training.
- **Key points**:
  - This model achieved around 68.7% accuracy and stopped training at 62 epochs.
  - An average Macro and Micro AUC was found to be 0.84.
  - It took around 14 hours and 37 minutes to finish the training.

### Experiment 01B:
**Training with VGG-16 Model using ImageNet Pretrained Weights (CIFAR-100 Dataset)**

- **Device used**: NVIDIA Tesla P100 16GB
- **Framework**: PyTorch
- **Augmentation used**: 
  - Random resized crop
  - Random horizontal flip
  - Random rotation
  - Random affine
- **Batch size**: 32
- **Optimizer used**: Adam (lr:0.0001)
- **Regularization techniques used**:
  - Early stopping with patience 10 on training loss
  - Weight decay
- **Others**:
  - In this study, the ReduceLROnPlateau learning rate scheduler was employed to dynamically adjust the learning rate based on the validation loss with patience 5.
  - The Xavier uniform initializer was utilized to initialize the network weights to maintain a good distribution and improve convergence during training.
- **Key points**:
  - This model achieved around 75.48% accuracy and stopped training at 26 epochs.
  - An average Macro and Micro AUC was found to be 0.85.
  - It took around 3 hours and 51 minutes to finish the training.

### Experiment 02:
**Training VGG-16 Model (ImageNet) Freezing All the Convolution Layers on CIFAR-100 Dataset**

- **Device used**: NVIDIA RTX 3070ti 8GB
- **Framework**: PyTorch
- **Augmentation used**: 
  - Random resized crop
  - Random horizontal flip
  - Random rotation
  - Random affine
- **Batch size**: 32
- **Optimizer used**: Adam (lr:0.0001)
- **Regularization techniques used**:
  - Early stopping with patience 10 on validation loss
  - Weight decay
- **Others**:
  - In this study, the ReduceLROnPlateau learning rate scheduler was employed to dynamically adjust the learning rate based on the validation loss with patience 5.
  - The Xavier uniform initializer was utilized to initialize the network weights to maintain a good distribution and improve convergence during training.
- **Key points**:
  - This model achieved around 65% accuracy after 50 epochs. Higher accuracy could be achieved with additional epochs.
  - An average Macro and Micro AUC was found to be 0.82.
  - It took around 3 hours and 15 minutes to finish the training.

### Experiment 03:
**Training VGG-16 Model (ImageNet) Freezing Dense Layers Except the Last One on CIFAR-100 Dataset**

- **Device used**: NVIDIA Tesla P100 16GB
- **Framework**: PyTorch
- **Augmentation used**: 
  - Random resized crop
  - Random horizontal flip
  - Random rotation
  - Random affine
- **Batch size**: 32
- **Optimizer used**: Adam (lr:0.0001)
- **Regularization techniques used**:
  - Early stopping with patience 10 on training loss
  - Weight decay
- **Others**:
  - In this study, the ReduceLROnPlateau learning rate scheduler was employed to dynamically adjust the learning rate based on the validation loss with patience 5.
  - The Xavier uniform initializer was utilized to initialize the network weights to maintain a good distribution and improve convergence during training.
- **Key points**:
  - This model achieved around 75% accuracy after 50 epochs.
  - An average Macro and Micro AUC was found to be 0.87.
  - It took around 5 hours and 41 minutes to finish the training.
  - This experiment achieved the highest scores and performance among the four experiments.



### Key observations:
- **Training Duration:**
  - Experiment 01A took the longest time (around 14 hours) to complete training.
  - Other experiments completed training much faster.

- **Accuracy and AUC:**
  - Experiment 01B (with pretrained weights) and Experiment 03 (freezing dense layers) achieved the highest accuracy (around 75%).
  - Experiment 03 recorded the highest AUC (0.87), slightly outperforming Experiment 01B.
  - Experiment 02 showed potential for higher accuracy if trained for more epochs, with an AUC of 0.82.

- **Model Performance:**
  - Experiment 03 demonstrated the best performance in terms of AUC and high accuracy, making it the top-performing model among all experiments.
  - Experiment 01B achieved the same accuracy as Experiment 03 but with a slightly lower AUC.
