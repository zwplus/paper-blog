# On the Detection of Digital Face Manipulation

这篇文章主要介绍一种attention机制的深度伪造鉴别和伪造区域定位的方法。

## 创新点（主要贡献）

1. 制作了一个包含4大类伪造方法的伪造数据集:人脸交换，表情重现，属性编辑已经人脸生成，原始的真实图像来源于CelebA和FFHQ以及FaceForensics++中真实的部分。
2. ***一个基于attention机制来提升模型分类精度并对人脸伪造区域进行定位***
3. ***一个名为Intersection Non- Containment (IINC)逆交集的评估方法的来评估人脸伪造区域的定位精度。***

## 主要内容

### 网络结构

![image-20221114221744900](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\On the Detection of Digital Face Manipulation.png)

网络结构设计层面主要还是使用传统的特征提取网络Xception-net为骨干网络，然后主要增加了一个attention-layer和一个计算attention map的损失的函数。

#### attention layer

attention-layer主要工作如下：

1. 生成输入图片的attention map，这个attention map用于突出输入特征中包含伪造信息的部分。
2. 对生成的attention map进行逐个像素Sgmoid归一化操作，将其值归一化到0,1，这样attention-map中每个点值就相当于该点是伪造的概率。
3. 将输入的特征图F与attention map逐元素相乘，由于real的像素点伪造概率为0，fake像素点的概率接近于1，因此能消除特征图的真实特征，突出图片的伪造特征。

##### Reg.MAP

直接基于回归来生成对应的attention map ，即使用一层卷积神经网络来根据输入的特征图来生成相应的attention map，最常见的attention map生成方法。

##### MAM Map

作者提出的一种添加了**约束的attention map生成方法**，作者在这里假设任何伪造图片所产生attention map都可以用如下一个线性回归来表示：

![image-20221114223947084](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\attention-map-线性表示.png)

这里M- 表示预先定义avervage attention map(平均注意力图），A表示预先定义的偏置。其计算方式主要为以下几步：

1. 首先作者首先使用100张真实图片在FaceAPP上产生100张伪造图片，然后使用对应元素相减的方法得到对应的100张伪造掩膜(掩膜中高亮区域标识被伪造了的区域，黑色部分表示真实的)。
2. 对这100张伪造掩膜进行求平均的操作来得到M-，M的形状为（HW*1)
3. 然后在利用PCA进行主成成分分析，获取最重要的10个成分来做为预定义的偏置,则A的shape为(HW*10)

![image-20221114234854718](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\MAMM.png)

实际上M-可以理解多种伪造方法对人脸区域伪造时产生的共同部分(不变量)，A则是不同伪造在对人脸区域进行伪造操作时的差异部分，通过二者线性回归来拟合不同伪造类型所产生的伪造区域掩码。而这里的a是通过一个卷积层和全连接层中从输入特征中提取出来的，通过a来控制这个线性回归操作去产生对应该输入图像的特征图，相当于a标识出了图片伪造时涉及具体类型，标识出该类型中变量部分是由A中那些成分组成的，最后和不变量相加得到最终的attention map。

#### LossFunction

![image-20221115000015372](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\模型的损失函数-1.png)

如图所示模型的损失函数主要分为2部分，一部分是分类的损失，而另一部分则是attention map的损失函数，由于attention map的生成可以使用监督、弱监督以及无监督的方式来生成，因此其对应的attention map的损失函数也存在差别。

##### 监督

![image-20221115001414736](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\监督情况下的损失函数.png)

以监督方式来生成attention map，即对于每一张输入图片都会事先计算其对应伪造区域掩膜来作为attention map的标签，对真实图片而言，其伪造区域掩膜为全0，而对完全生成的图片其对应的伪造区域掩码是全1的图，而对于那些基于源图片的进行交换等操作生成伪造图片而言，会将其与源图片做一个相减操作产生其对应的掩膜。

(4)式中的Matt就是生成attention map 而Mgt就是对应的标签，即预先计算的伪造区域掩膜。具体说一下计算方式：

1. 首先将源图片和伪造图片转成灰度图片，然后做差求出二者差异--different-map
2. 将different-map归一化到0-1区间，然后以0.1位阈值进行二值化，可以理解为如果归一化后的different-map中某个像素点的值超0.1，即对应到源图片和伪造图片对应位置像素值相差超过25，则认为该位置时位置，其值设为1

##### 弱监督

由于采用监督方式计算伪造区域掩膜时对于那些非凭空生成的伪造图片而言，必须要找到其对应的源图片，但是很多时候可能无法找到源图片，除此之外存在着很多伪造数据我们并不清楚其所使用的伪造方法，因此也无法判断其是否是凭空生成的，所以只能采用弱监督的方式：

![image-20221115002107398](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\弱监督的损失函数.png)

对于真实图片而言其attention map仍然为值全为0的掩膜，对于伪造图片而言由于这里无法计算其相应的掩膜，但是考虑伪造区域的掩码值往往很大并且接近于1，因此作者这里给了0.75来作为参考值，同时考虑伪造图片可能整个都是伪造也可能部分是伪造的，但是非伪造区域的值往往比较小，因此attention map中的极大值往往代表着更大的伪造可能，因此选择该极大值来和事先确定阈值0.75计算损失，以此作为attention map的损失。

**个人认为这里伪造图片部分做的损失函数有些粗糙，因为当attention map的极大值超过0.75时也需要计算损失感觉不够合理，个人认为可以更换成，max(0.75-max(Sigmoid(Matt)),0)的方式会更好一点。**除此之外如果能给确定图片的伪造方法，对完全凭空生成的伪造图片而言，使用全1作为伪造掩膜标签可能更好。

##### 无监督

这里无监督的方式很好理解，即整个损失函数不考虑attention map的损失，只使用分类损失。

##### 总结

监督的方式在对attention map中不仅对attention map中是否包含伪造信息做了监督（即是否是伪造图片生成的attention map），实际上还对伪造区域做监督，实际上这也是相当于对伪造区域的定位做了监督，但是在若监督部分实际上只考虑attention map是否包含伪造信息做了监督，即if fake那部分公式，就是相当于attention map中伪造概率最高的点的概率需要超过预设的阈值0.75，才能标识这个attention map中是伪造图片产生的attention map做了监督。

简单理解弱监督时attention map标签只有真实或者伪造标签，但是监督方式下的attention map不仅有真实伪造标签，还有伪造区域的标签。

### 新的评估指标

在评估生成的attention map对于伪造区域定位的效果时，作者使用了IOU，余弦相似度以及PBCA(像素级的二分类准确度)，但是作者发现这三种评估指标在评估时的鲁棒性和连贯性不好，因此作者又提出了Intersection Non- Containment (IINC)逆交集的评估方法来评估区域的定位精度。

![image-20221115102249488](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\IINC.png)

I表示Matt和Mgt的交集，U是Matt和Mgt的并集，M-和|M|是表示M的均值和L1范数。注意这个IINC的值越小则表示精度越高。

**相较于IOU通过重叠区域的比例来估测大小，IINC则是通过评估非重叠区域的比例来评估的检测精度**

![image-20221115103627157](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\使用不同评估指标时的评估结果.png)  

### 实验结果

![image-20221115143741693](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\实验结果.png)



作者使用Xception-net为主干网络，使用MAM和Reg两种方式来生成对应的attention-map，并使用unsup，weak-sup和sup的方式进行监督，可以看到在使用监督的方式下，使用Reg方式生成attention-map的方式能取得最优的结果，个人认为这是由于MAM这种使用线性回归来拟合的方式不能很好地完全拟合实际的伪造区域掩码，我认为存在着改进的地方，作者只使用了faceapp的伪造图片和源图片来产生M-和A，我认为应该增加一些其他伪造方法产生的伪造图片，这样能够有利于去产生一个更加准确的M-和A，这样获取能够提高MAM在sup条件的表现能力。

表中的-map，表示直接使用生成的attention-map的均值高低来判断是否真实伪造，不再使用后续的分类网络来做评估。虽然结果不如正常使用分类的结果，但是在一定程度说明使用attention map是能够用于区分伪造和真实的。

![image-20221115145359872](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\泛化性实验.png)

表4比较了多种模型在Celeb-DF上的泛化性能，可以看到本篇文章提出的基于注意力机制的论文表现出了最优的泛化性。

![image-20221115150355366](D:\PersonSpace\blog\paper\img\On the Detection of Digital Face Manipulation\伪造区域定位结果.png)



