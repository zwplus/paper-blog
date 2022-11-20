# Deep Residual Learning for Image Recognition

## Abstract

**question**:Deeper neural networks are more difficult to train.(深的神经网络非常难以训练。)

**main_work**: We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.

 	1. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.
 	2. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.

**experiment_result**:

 - On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task.
 - We also present analysis on CIFAR-10 with 100 and 1000 layers.
 - Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset.Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

## Introduction

1. 深度卷积神经网络给图片分类问题带来了一系列的突破。
2. 在网络深度意义的驱动下，一个问题被提出了：网络的层数是否越深越好？
   - 网络层数的增加可能会导致梯度爆炸或者消失，进而无法收敛，但是已经通过归一化normalized解决了。
3. 通过归一化使得模型能够收敛了，但是模型又出现了退化的问题：随着网络的不断加深，模型的准确率先上升，但是随后迅速下降。不幸的是，出现这一现象并不是过拟合overfitting导致的，因为相关实验表明随着网络层次的增多，模型的训练误差也出现升高。
4. 通过比较一个浅的网络和其对应的深的网络，我们可以想到一个构建深度网络的方法是：可以在浅的网络上添加一些恒等映射的层来实现更深的网络。这种方法表明，更深的网络理论上不应该比浅的网络存在更高的训练误差（因为理论上添加层学习到恒等映射关系就可以达到与浅层网络一致的关系）。但是相关实验表明我们现有的方法无法找到一个与恒等映射近似或者更好的方案。
5. 我们通过残差网络解决这种退化问题，对于目标函数H(x)，我们不期望于堆叠的网络层能够直接去拟合H(x)，我们更期望于这些层能够去拟合残差：F(x)=H(x)-x。因此目标映射可以被看成F(x)+x，我们假设优化这种残差映射比优化原始映射更加容易。
6. F(x)+x能够被带有shortcutting链接的网络结构实现，这种shortcutting链接会跳过一个或多个层，由于其只是简单地实现恒等映射，因此其不会带来额外的参数和复杂度。
7. 在Imagenet等数据集上的实验表明，我们的方法使得更深网络变得更容易优化，并且能够容易地获得更深网络所带来的精度上的提升。同时我们方法能够在小数据集上训练出很深的网络。









