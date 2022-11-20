# Representative Forgery Mining for Fake Face Detection

## 文章创新点

1. 提出了一种基于注意力引导的数据增强方法
   - FAM：一种可以定位出对检测器而言最敏感的面部区域的方法，用于知道后续的数据增强。
   - SFE：一种基于attention的数据增强方法，在FAM指导下，帮助检测器去更加均衡地去分配注意力
   - RFM：一个可以在无监督下可视化伪造区域的框架，能够帮助基于CNN的伪造检测器实现SOTA的性能。
2. 使用该方法训练出的伪造检测器能指出伪造图片所使用的伪造方法

### RFM

![image-20221111113508174](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\RFM.png)

RFM框架的处理流程：

1. 将原始图像输入检测器进行一次前向传播，得到Real和Fake的概率，在利用得到值进行一次反向求导求解出FAM(forgery attention map),FAM中高亮区域就是对检测器而言比较敏感的区域。*** 这一轮的反向求导是对每个位置的像素值进行求导，而不是对那些权重w求导 ，这样可以得到每个像素点对最后结果的影响程度，即检测器的敏感程度***
2. 在得到FAM后，从FAM中选择出最敏感的N个像素位置，然后以这些位置为中心随机生成矩形的待擦除区域，然后在原图上将对应区域进行擦除。
3. 将经过擦除后的图像输入到检测器中进行前向传播和反向求导来对模型进行更新

### FAM

##### FAM

![image-20221111114644163](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\FAM.png)

FAM的主要操作如下：

1. 首先输入原始图片进行前向传播，分别得到图片真伪两个类别的概率Ofake和Oreal
2. 进行方向传播，分别计算Ofake和Oreal对于原始图像上每个像素点的梯度，并计算二者梯度差的绝对值作为敏感程度
3. 因为输入的图像往往不是单通道的，因此对于一个像素点而言，在每一个通道上都产生一个梯度差绝对值，这里对于一个像素点，选取所有通道中梯度差绝对值最大的值作为该像素点的敏感程度。

相较于CAM，FAM是用来定位对于检测器而言比较敏感的区域，而CAM则是用来突出检测器进行决策时所参考的区域，前者是求梯度，即该区域的变化对最后的值产生影响较大，而后者则是寻找对于最后计算出的结果而言，那一部分的区域的贡献比较大。FAM是基于整个原始图像来生成MAP的，但是CAM则是将CNN的最后的一层输出的特征图映射到原始图像来生成最后的MAP。

##### CAM

![image-20221111191810245](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\CAM.png)

CAM，首先使用最后一层卷积生成的特征图与特定分类所对应权重相乘就和，再将最终的结果缩放到原始输入图片的大小，最终就得到该类别在该输入图像上的类激活映射图。需要注意的同一张图像的类激活映射图会随着类别的不同，所观察的区域也会存在差别。

![image-20221111192552748](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\CAM2.png)

### SFE

![image-20221111114748640](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\各种随机擦除方法的效果.png)



#### RE

一种简单的随机擦除方法，在输入图片在随机选择一个位置并选择一个随机大小进行擦除，由于在伪造检测过程中，对于检测器比较敏感的区域是相对不连续，不会完全集中在一个位置，有时候不同的敏感区域可能相距很远，因此很难使用随机擦除来抑制特定敏感区域来鼓励检测器去发现更多特征。同时随机算法的缺陷会很容易导致图像的中心区域更容易被擦除。

#### AE

AE是基于class attention Mapping的区域擦除方法。其会根据CNN的特征提取部分产生的CAM图来对输入图片的特定区域进行擦除，（需要注意的是不太确定，其是对真伪两个类别的CAM都进行擦除，还对是对二者产生的CAM的求差值后，对差异较大的区域进行擦除，按照文章整体的意思应该是对差异较大的区域进行插除。）

缺陷：由于是直接将CNN生成的特征图直接映射到原始的输入图片上，会比较容易出现位置的偏差，会导致擦除位置与期待的位置不一致。同时使用AE这种过于细粒度的擦除方式，存在容易导致发生overfitting的可能。

#### SFE

![image-20221111114711154](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\SFE.png)

本文所提出的基于SFE的方法的整体算法流程图如上图所示

1. 首先人为设定一个进行SFE擦除概率，即一张图片有一定的概率进行SFE擦除
2. 若该输入图片要进行SFE擦除，则首先根据FAM求出的每个像素点的敏感程度进行由高到底的选择来做为待擦除区域的中心
3. 如果所选择的擦除中心未被擦除，则以该中心建立随机大小的擦除区域，对该擦除区域进行擦除，如果所选择的擦除中心已经被擦除过了，则返回2重新选择擦除中心
4. 重复上述2,3两步，直到一共有N个区域被擦除

优点：能够精确地确定需要进行擦除的区域，能够使用多个擦除区域来尽可能多的擦除需要进行擦除的敏感区域以此充分鼓励检测器去发现伪造细节特征。检测器无法额外信息，因此能够在一定程度上阻止过拟合的发生。

### RFM：代表性伪造区域的可视化

![image-20221111114834338](D:\PersonSpace\blog\paper\img\Representative Forgery Mining for Fake Face Detection\代表性伪造区域的可视化.png)

图中每个伪造方法对应的FAM都是使用100张该类伪造方法生成图片获得FAM取平均获取的。不同方法所产生的FAMs存在着差别，因此，后续可以通观察FAM的差别来实现多伪造方法的研究，可以通过使用每一伪造方法生成对应的平均FAM，对后续伪造图片的FAM和这些平均FAM计算余弦相似度，来判断后续伪造图片的所属伪造方法。