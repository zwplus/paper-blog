## 文章创新点

1. 多注意力机制引导网络关注于局部特征

   - 真实伪造人脸之间的差异往往是细微的并且往往发生在不同的区域中，这很难被单注意力结构网络所捕获，

   - 使用基于注意力的池化来取代全局池化层，因为不同区域之间的纹理特征差异很大，如果使用全局池化层可能导致这种差异被平均，导致可区分性的损失

2. 纹理增强模块用于放大浅层特征中纹理细节 
   - 伪造方法所造成的伪影往往更多地保留在浅层特征中，纹理信息往往代表着浅层特征中的高频部分，因此需要对浅层特征中的高频部分应该被关注，并被增强

3. 将由注意力机制引导的浅层纹理特征和深层语义特征结合BAP

4. 由于一般的增强策略会使得多注意力机制退化成单注意力机制，因此设计以下2个机制来引导多注意力机制的学习

   - 区域独立性损失

   - 注意力引导的增强机制

## 网络结构设计

![网络结构](D:\PersonSpace\blog\paper\img\mutli-attention-deepfake-detect.png)

### 纹理增强模块

1. 对输入的浅层纹理特征首先做一个平均池化，平均池化的操作相当于一个低通滤波器，会抑制图像中的高频信息
2. 将原始图像与平均池化的后图像做一次相减的操作，最终输出的结果中保留信息主要来源原始图像中的高频部分，这就相当于突出图像中的高频信息。
3. 在将得到图像差输入到一个Dense block(3层)中去进一步进行纹理增强，因为此时输入的图像绝大部分信息为原始图像信息中的高频部分，因此提出的特征也带表原始图像中高频特征部分。

### 多注意力模块

![多注意力模块和注意力池化模块](D:\PersonSpace\blog\paper\img\multi-attention-deepfake-attention.png)



1.多注意力模块组成比较简单，输入为特征提取网络中某一层的的输出，经过一个卷积核大小为1*1的卷积层以及Batch normalization 和Relu激活函数产生注意力图，输出注意力图的通道数为特征图的数目，这里架设为k张

### BAP 基于attention map的双线性池化层

1. 首先将获取的k张注意力图的通过双线性插值的方法调整至于前面提取纹理特征图大小一致，然后对于每一张attention map 将其与前面提取出的纹理特征图逐元素相乘，因此最终获得k张基于注意力的纹理特征图。（k*channel\*w\*h)

2. 得到基于注意力的纹理特征图后，通常是对每个特征图进行全局池化来得到每张的特征图的特征向量，即channel维特征向量的，最终得到一个k*channel的特征矩阵，但是考虑不同attention map的强度存在不一致，这样会导致一部分特征图的特征向量的值很大，一部分特征图产生特征向量很小，进而会导致一些区域的纹理信息被特别强调，一部分区域的纹理信息被抑制，这样就违背了我们想关系不同区域纹理信息的初衷，其实会变相导致多注意力机制退化成单注意力机制。因此这里采用了一个标准化的池化操作。这样不要map产生的特征向量的里面元素值的范围就会被限定在一个相同的范围里。

   ![image-20221106192017677](D:\PersonSpace\blog\paper\img\normal-pool.png)

3. 除了对浅层的纹理特征进行注意力引导操作，对于特征网络中生成的深层语义特征也进行注意力引导来获取深层的特征向量，这里先将所有生成的attention map求和生成一张全局的attention map，再将该attention map和特征提取网络提取的语义特征相乘并进行相同的标准池化操作来获取全局的深层特征向量。

### 区域独立损失

![image-20221109224203597](D:\PersonSpace\blog\paper\img\区域独立性损失-1.png)

![image-20221109224243870](D:\PersonSpace\blog\paper\img\区域独立性损失-2.png)

1. 先说明一下几个变量的含义，首先B代表batch，M代表之前的attention-map数,Vij实际上应该前面双线性池化后的一个向量，维度为channel(通道数量)，这里Vij并不是一个值，是atttention在，Ctj表示经过t轮迭代后第j个attentio-map的中心，其也是一个channel维的向量，a是c这个attention-map的中心的学习率（或者更新速度）
2. 首先看公式4，在一个epoch的每一个batch中，attention-map的每个中心都会更新一次，如果特征矩阵是M*Channel的，则C中的每一行代表一个attention-map的中心，随着迭代的不断进行，最后每个attention-map的中心C将接近与于特征矩阵的均值，所以换句话说，其是将所有每个attention-map计算出的特征向量的均值作为区域中心
3. 在看公式3的前半个部分，是用来计算每个特征向量与特征中心的距离，而Min(y)则表示描述里区域中心足够近这个标准，如果求出的特征向量到特征中心的距离小于设定Min(y)，那这一部分的损失就为0，因为坐着认为对于真实视频而言，特特征向量应该区域中心足够近，而对于伪造视频，为了鼓励模型去寻找更多细节特征，因此区域中心近的标准有所放松。
4. 公式4的后半个部分，是用来计算每个区域中心之间的距离，同理Mout则表示两个中心之间是否足够远的，如果不够远，则会产生损失。
5. 总体上来看公式后半部分保证了每个区域中心之间足够远，attention-map之间观察的区域不同。公式前半部分则是用来保证每个attention-map所关注区域足够集中
6. 而4式与公式1的前半部分结合，保证特征向量和特征中心之间彼此不断靠拢，最后特征中心位置区域稳定范围内，输出的特征向量也会在特征中心的一个稳定范围内，这样能够每个attention-map关注区域固定，降低随机性。即不同输入图片产生的attention-map也会在一个固定区域中

## AGDA 基于注意力的数据增强模块

![image-20221110142843970](D:\PersonSpace\blog\paper\img\AGDA.png)

基于AGDA的数据增强模块主要进行如下操作：

1. 首先对于每一个训练样本，从它生成的attention-maps中随机选择一个Attenion-map.
2. 对该样本进行高斯模糊，生成高斯模糊后样本
3. 利用公式6，将二者相加，得到相加后的样本，作为数据增强后的样本

通过AGDA这样一个数据增强操作后模型每个attention-map中所关注的区域会更加均匀，不会过分地强调在一些特定的位置上，同一个attention-map中的响应更加均匀，AGDA可以随机地抹去最明显的区分性区域，这迫使不同的注意力地图把它们的反应集中在不同的目标上。AGDA机制可以防止单个注意力区域扩展过快，并鼓励注意力块探索各种注意力区域划分形式。

![image-20221110144524293](D:\PersonSpace\blog\paper\img\无区域独立性损失和AGDA.png)

![image-20221110144608989](D:\PersonSpace\blog\paper\img\区域独立性损失.png)

![image-20221110144709664](D:\PersonSpace\blog\paper\img\区域独立性损失和AGDA.png)