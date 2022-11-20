# Towards Solving the DeepFake Problem : An Analysis on Improving DeepFake Detection using Dynamic Face Augmentation

## 两个问题

1. 为什么在大量数据上训练出来的的深度伪造检测器的泛化性仍然不佳？
2. 如何使用现有视觉网络架构来达到检测器在面对未知数据集的泛化性？

## 主要贡献（创新点）

1. 对目前主流的伪造数据集进行了易于理解的分析来标识出其存在的缺点。
2. *** 使用面部聚类的方式来分析数据集，并通过聚类的结果来指示数据集的划分，防止了数据泄露。*** **个人认为这一步是本文的一个非常有意思的点，但是文章并未做消融实验来比较展现这样做实际上起到的作用**
3. 提出了一种名为Face-Cutout面部擦除的数据增强方法，其能够根据人脸面部地标点的信息来对面部区域进行擦除。

## 主要内容

### 为什么在大量数据上训练出来的的深度伪造检测器的泛化性仍然不佳？

![image-20221113115239345](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\伪造数据集分析.png)

**现有数据集的制作方式往往是通过是使用一张真实的人脸去产生多张通过不同伪造方法生成的伪造图片，这导致这样构成数据集存在对真实人脸图片过度采样的问题，数据的过度采样和缺乏变化会导致模型很容易在学习到辨别真实伪造的特征之前就已经过度拟合数据。**

### 如何使用现有的视觉模型架构来实现面对未知数据集时的泛化性

####  基于face cluster 的数据划分方式

![image-20221113120243300](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\face_cluster.png)

#### face cluster 具体流程

1. 首先使用面部检测模型将数据集中真实视频中的人脸提取出来，然后将这些人脸通过基于CNN的编码器来生成维度128维的向量。
2. 使用Density-Based Spatial Clustering of Applications with Noise (DBSCAN)和欧式距离来对这些真实人脸产生的向量进行聚类
3. 将伪造视频中人脸分配到其所对应的源视频中的真实人脸所属的类别中

从图片能够看到，即使是数据量很大的数据集，其所产生聚类类别相较于其数据量而言还是很少的。

#### 基于face cluster的面部划分

由于真实图片的过度重采样，这样会导致传统的数据划分方法很难起到发挥其应有的作用，存在数据泄露的可能（**一张很类似的人脸(同人脸)会同时出现在验证集和训练集中，这里作者认为是一种数据泄露**），为了避免这种情况，作者使用了前面聚类的结果来指示数据集的划分，来保证训练集和测试集的不会出现共同的人脸。

1. 依据人脸聚类的结果对数据进行分组，同一类别中的视频和图片被视为一个单元。
2. 按照单元来对数据集进行划分，而不是按照数据的原始标签。

#### Face-Cutout

1. 生成面部差异掩膜：首先通过计算每一帧真实视频帧和其对应的伪造视频帧之间Structural Similarity Index (SSIM)结构相似性的指数来产生像素级的掩膜，掩膜中的高亮区域标识出伪造的像素，暗部区域标识真实像素。后续面部擦除时需要同时输入待擦除的伪造人脸图片和对应差异掩膜。

   ![image-20221113123745490](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\面部差异掩膜.png)

2. 多边形区域提议：首先从68个人脸地标点中随机选择出一组地标点，并依据于所选出的地标点组来使用基于感官区域或者凸包的方法来生成待选择的多边形区域。
3. 多边形区域选择：首先计算出整张掩膜的中包含的伪造像素点A，在分别计算出每个多边形区域所包含的伪造像素点Ci，计算多边形中伪造像素点占所有伪造像素点的比例，筛选出比例小于阈值的多边形区域来，再比较筛选出的这些多边形区域的面积，选择面积最大的多边形区域作为待擦除的区域。这样能够实现在保留尽可能多的伪造信息的同时擦除尽可能多的无关区域。（这里的伪造像素点就是第一步中生成的差异掩膜中值为1的点，需要主要注意的真实人脸图片不含伪造像素点，因此不存在这个比较过程，直接选择面积最大的多边形区域来进行擦除）。

$$
p=|Ci|/|A|
$$

​	4. 在选择出待擦除区域后，可以选择多种擦除方法来对这些面区域进行擦除。

![基于面部感官区域的面部擦除](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\基于感官区域的擦除方法.png)

![基于凸包的面部擦除](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\基于凸包的三种面部擦除区域选择-1.png)

![image-20221113123005083](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\基于凸包的三种面部擦除区域选择-2.png)



### 实验结果

![image-20221113131832551](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\实验结果.png)

### 方法的有效性解释

作者为了证明其提出的face-cutout方法能够缓解模型的过拟合，通过展示在使用Face-cutout的情况下和不使用的情况下产生的类激活映射图CAM来证明方法的有效性。

**从图中可以看见，在未使用Face-Cutout数据增强措施时，CAM的高亮区域会包括整个面部已经作为背景的信息，因此可以合理推测模型只是将输入的整张图片作为了伪造图片，并没有学习到正确伪造区域。而通过比较使用了Face-Cutout措施产生的CAM高亮区域和差异掩码可以发现，对模型分类产生影响的正是这些存在着差异的部分。**

![image-20221113132323366](D:\PersonSpace\blog\paper\img\Towards Solving the DeepFake Problem  An Analysis on Improving DeepFake\类激活映射图.png)