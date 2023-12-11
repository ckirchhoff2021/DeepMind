### 混合精度训练

### 1. 论文题目

+ Mixed Precision Training

### 2. 背景和动机

+ 模型参数量增加时，内存占用和计算资源都会随之增加
+ 混合精度主要考虑使用单精度FP16来做训练，以此来达到训练加速的目的，但是使用FP16训练，需要考虑的核心问题在于**如何才能做到不损失模型精度，如何才能保证本来在FP32数据域下正常训练的模型能正常迁移到FP16训练**

### 3. 方法

+ 为了保障模型在FP16训练时达到与FP32相同的精度，本文针对混合精度训练提出了3个策略

####  3.1 FP32 Master copy of weights

+ 在混合精度训练过程，权重，激活和梯度都用FP16存储
+ 为了保证在训练过程中， 匹配FP32网络的精度，会保存并更新所有权重的**一份FP32拷贝**

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/770fe922-34da-4c6b-bcb5-0f89197da62f)

+ 为什么会需要保存权重的一份FP32拷贝？
  + 在训练过程中，梯度的更新值变得非常小，FP16有可能无法表示，比如dx = 0.0006，超出了**FP16**的表示范围，任何小于$2^{-24}$的在FP16中表示都会表示成0
  + 训练到后期，学习率和梯度都会非常小，在FP16情况下，很有可能表示成0，从而影响模型精度
  + **weight**值相对**weight**更新值很大时，weight更新值用FP16表示，会出现大数吃小数的情况，当weight值的绝对值是weight更新值的2048倍甚至更大时

#### 3.2 Loss Scaling

+ 模型在训练过程中，无论是梯度还是数据的传导都不能很好的利用**FP16**的指数位，真正影响模型收敛的是FP16的小数位，且越往后小数位显得越重要
+ 将FP16的小数位扩充到指数位，得到更多的有效小数位，是一种避免FP16训练精度问题的有效手段
+ 为了在反向传播过程中尽可能多的保留梯度更新值的有效位数，可以对loss进行scaling操作(比如scale 8倍)，梯度在反向传播时也会scale 8倍，这样有效位进位，表示范围更广

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/767921ac-963a-4202-a922-6bd45b42b0a1)


#### 3.3 改进计算方法，有效利用 FP16 @ FP16 + FP32

+ 对大部分网络而言，基本计算逻辑可以分为三种，向量点积，reduce操作，以及point-wise操作
+ 经过试验验证，有些网络在 FP16@FP16 + FP32之后又convert成FP16存储到内存中，相对于不叠加FP32，叠加FP32的模型精度要好很多
+ 矩阵乘用**FP16**计算，bias累加或者与其它矩阵做加法用**FP32**

+ 对于大规模的reduce操作建议转换成FP32计算
+ Point-wise的操作FP16和FP32均可