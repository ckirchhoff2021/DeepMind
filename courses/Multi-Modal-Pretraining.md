## 多模态经典论文解读

+ UNITER，CLIP，ALBEF，VLMO，VILT， BLIP，BERTV3，SD， MAE，DIT

#### **1. UNITER**

+ **Universal Image-Text Representations**

![image-20240417143825828](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417143825828.png)

+ **模型结构如上图所示：**
  + 视觉部分利用FasterRCNN来提取特征，主要是pooled ROI特征和location特征，分别经过全连接层投影到相同的特征空间
  + 文本部分直接分词，然后使用token embedding 和postion embedding
  + 文本embedding和视觉embedding一起输入到Transformer结构中
+ **核心训练任务：**，输入是图片文本对，image-text pair
  + **MLM**，masked language modeling conditioned on image regions
    + 输入是图片文本对，image-text pair，在sentense中随机mask一些token，在模型还原这些token，完形填空
  + **MRM**，masked Region Modeling conditioned on input text
    + 对于每个region都有一个特征，让UNITER来预测这个高维特征，让模型输出被mask区域的特征，利用L2 loss监督训练
  + **MRC**，masked region classification
    + 每个region得到特征向量后，RCNN都会预测一个label，让UNITER来对mask的region进行预测，使模型学习每个mask region的分类，使用交叉熵或者KL散度
  + **ITM**，Image-Text Matching
    + 对输入的Image-text pair随机替换image或者text，最后预测输入的image-text是否有对应关系，二分类
  + **WRA**, word-Region Alignment
+ 文本mask就是替换对应的token为特殊token [MASK]，区域mask就是替换对应的视觉向量为0
+ [GitHub - ChenRocks/UNITER: Research code for ECCV 2020 paper "UNITER: UNiversal Image-TExt Representation Learning"](https://github.com/ChenRocks/UNITER)

+ **常用多模态评测任务：**
  + **Image Captioning**，概括图片的内容
  + **Visual Question Answering**，对图片进行提问，让模型回答相关问题
    + 将图片和问题输入到模型中，输出各答案的置信度
  + **Dense Captioning**，对图片的细节感兴趣，针对图片的特定区域进行描述
  + **Referring Expressions**，给定一个描述语句，定位该语句的指定区域
    + 对每个区域输出一个score，score最高的region作为预测的region
  + **Visual Dialogue**，针对图片进行多轮问答
  + **Visual Entailment**，图像是否包含文本描述的内容
    + 三分类任务，label分别是Entailment, Neutral，Contradition
  + **Natural Language for visual Reasoning**
    + 输入两张image和一个caption，输出事caption与image的对应关系是否一致，label为True或者False
  + **Image-Text Retrieval**，图文检索
    + 图像检索文本或者文本检索图像
  + **Visual Commensense Reasoning**
    + 选择题，对一个问题给几个备选答案，模型从这几个答案中选一个，再从备选理由中选择出选择这个答案的理由

#### 2. CLIP

+ **Learning Transferable Visual Models From Natural Language Supervision**
+ [GitHub - openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image](https://github.com/OpenAI/CLIP)

![image-20240417153849472](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417153849472.png)

+ **模型结构如上图所示：**
  + 双塔结构，分别使用Image Encoder和Text Encoder来提取文本和视觉特征
  + 视觉特征考虑CNN或者VIT，文中尝试了resnet50
  + 文本特征使用12层的Transformer Encoder，每一层包含8个attention heads
+ **核心训练任务ITC**
  + <img src="C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417154602204.png." alt="image-20240417154602204" style="zoom:80%;" div align=left />
  + 视觉特征和文本特征分别投影到特征空间然后计算点积，Image-text 原生匹配的才是正样本，不匹配的都是负样本，从logits来看就是对角线上为正样本，其他位置均为负样本

+ **进行zero-shot推理**
  + 利用prompt-template，A photo of xx，利用文本编码器和视觉编码器分别提取文本特征和视觉特征，然后计算余弦相似度，取得分最高的类别

#### 3. ALBEF

+ **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation**
+ [GitHub - salesforce/ALBEF: Code for ALBEF: a new vision-language pre-training method](https://github.com/salesforce/ALBEF)

+ ![image-20240417155432475](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417155432475.png)

+ **模型结构如上图所示：**
  + 也是典型的双塔结构
  + 视觉编码器使用12层的VIT transformer encoder，一般来说对于多模态任务，视觉编码器要比文本编码器大
  +  文本编码器使用6层的transformer encoder
  + 多模态编码器，融合layer使用6层的transformer encoder
+ **核心训练任务**
  + **ITC**，image-text contrastive借鉴CLIP的思想
  + **ITM**， image-text matching，二分类任务，判断image-text是否匹配
    + 使用多模态编码器，也就是融合模块的[CLS] token的embedding作为image-text pair的embedding，外接一个全连接层进行二分类
    + 难负样本采样，hard-negative-sampling，选择余弦相似度最高的负样本进行训练
  + **MLM**, masked-language modeling，bert的完形填空任务
+ **动态蒸馏**
  + 使用momentum model生成pseudo-targets(伪标签)监督训练过程
  + 从网络上扒取的图像文本对存在大量噪声，正样本对通常表现出弱相关性
  + 动量模型，包括单模态编码器和多模态编码器，通过EMA（exponential-moving-average）的方法来生成，可以作为一个continuously-evolving的教师模型在训练过程生成psuedo soft label，作为ITC和ITM的监督信号

#### 4. VILT

+ **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**

+ [GitHub - dandelin/ViLT: Code for the ICML 2021 (long talk) paper: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"](https://github.com/dandelin/vilt)

  <img src="C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417162814902.png." alt="image-20240417162814902" style="zoom:100%;" div align=left />

+ **模型结构如上图所示：**

  + 模态交互部分的Transformer使用预训练VIT进行初始化，不再需要一个额外的视觉encoder
  + 文本部分直接分词，然后使用token embedding + postion embedding，并与modal-type embedding进行concat
  + 视觉部分划分成多个patch并flatten，并和position embedding一起投影到特征空间, 与modal-type embedding进行concat
  + word embedding和vision embeding都增加[CLS]embedding，方便与下游任务对接
  + word embedding和vision embedding通过model-type embedding进行区分

+ **常用多模态双塔结构分类**

  ![image-20240417171500782](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417171500782.png)

+ **核心训练任务**
  + **ITM**， image-text matching，二分类任务，判断image-text是否匹配
    + 随机以0.5的概率将文本对应的图片替换成不同的图片，使用文本标志位的输出外接一个二分类FC，判断图像文本是否匹配
  + **MLM**, masked-language modeling，bert的完形填空任务

#### 5. VLMO

+ **VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts**
+ [unilm/vlmo at master · microsoft/unilm · GitHub](https://github.com/microsoft/unilm/tree/master/vlmo)
+ ![image-20240417170105189](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417170105189.png)

+ **模型结构如上图所示：**

  + 现有的视觉-语言预训练基本可以分为两类，
    + 一类是使用双塔encoder分别编码图像和文本，使用余弦相似度和线性投影层来进行模态的交互，比如CLIP类的模型，在图文检索任务中效果明显，但是在复杂的推理场景效果一般较差
    + 一类是通过跨模态注意力机制使用融合encoder来处理多模态的交互
  + 视觉和文本模态共享Transformer Encoder参数
  + 使用MoME结构，在FFN中根据不同的模态走不同的分支得到输出，包含视觉Expert，文本Expert以及视觉-文本Expert
  + 对于image-text pair，分别encode得到视觉表征，文本表征和视觉-文本联合表征
  + 视觉表征H1，使用patch embedding + position embedding  + modal_type_embedding，其中patch embedding包含[CLS]
  + 文本表征H2，分词得到word embedding， word embedding + position embedding + modal_type_embedding，其中word embedding包含[CLS]和[SEP]
  + 视觉-文本表征，H=[H1;H2]

+ **训练细节**

  ![image-20240417172444952](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417172444952.png)

  + 对于image-text pair拆开成image, text和image-text分别训练，image训练完，再训练text，训练text冻结MHA，和视觉FFN的参数

![image-20240417173624537](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417173624537.png)

+ **核心训练任务**

  + **ITC**，image-text contrastive借鉴CLIP的思想
    + 使用image的[CLS]token和文本[CLS]token的embedding作为图像和文本的聚合表征
  + **ITM**， image-text matching，二分类任务，判断image-text是否匹配
    + 随机以0.5的概率将文本对应的图片替换成不同的图片，使用文本标志位的输出外接一个二分类FC，判断图像文本是否匹配
    + 借鉴AlBEF的思想，进行hard-negative-sampling，在负样本中选择余弦相似度最高的文本作为负样本
  + **MLM**, masked-language modeling，bert的完形填空任务

#### 6.BLIP 

+ **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**

+ [GitHub - salesforce/BLIP: PyTorch code for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://github.com/salesforce/BLIP)

  ![image-20240417175815349](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240417175815349.png)

+ **模型结构如上图所示：**
  + A unified VLP framework to learn from noisy image-text pairs
  + 依然是双塔结构
  + vision部分使用patch embedding然后经过transformer encoder得到vision embedding
  + text部分使用transformer encoder得到word embedding

+ **核心训练任务**
  + **ITC**，image-text contrastive借鉴CLIP的思想
    + 使用image的[CLS]token和文本[CLS]token的embedding作为图像和文本的聚合表征
  + **ITM**， image-text matching，二分类任务，判断image-text是否匹配
    + 随机以0.5的概率将文本对应的图片替换成不同的图片，使用文本标志位的输出外接一个二分类FC，判断图像文本是否匹配
    + 借鉴AlBEF的思想，进行hard-negative-sampling，在负样本中选择余弦相似度最高的文本作为负样本
  + **MLM**, masked-language modeling，bert的完形填空任务

#### 7. BeitV3

+ **Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks**
+ [unilm/beit3 at master · microsoft/unilm · GitHub](https://github.com/microsoft/unilm/tree/master/beit3)

![image-20240418095421403](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418095421403.png)

+ **模型结构如上图所示：**

  + 模型结构基本与VLMO相同

  ![image-20240418095859397](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418095859397.png)

  + Multiway Transformer共享self-attention的参数，在FFN中区分当前具体是哪个模态，走对应的FFN分支
  + 除了本身可以作为fusion-encoder之外，还可以通过vision-ffn或者language-ffn转变为单模态encoder和dual-encoder
  + VLMO是24层encoder-layer， hidden_size=1024, 16个attention head
  + BEIT-3是40层encoder-layer, hidden_size=1408, 16个attention head

+ **核心训练任务**

  + **masked data modeling**
    + 随机masked一定比例的token和image patch，训练模型去恢复被mask的token
    + 这个统一的mask-thne-predict任务不仅学习表征，同时学习不同模态之间的对齐
    + 文本直接分词得到embedding，图像使用BEITv2得到vision Token

#### 8. Beit

+ **BEIT: BERT Pre-Training of Image Transformers**

+ [unilm/beit at master · microsoft/unilm · GitHub](https://github.com/microsoft/unilm/tree/master/beit)

  ![image-20240418145806637](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418145806637.png)

+ **模型结构如上图所示：**
  + 在预训练之前，先通过一个autoencoding-style reconstruction 学习图像分词器image tokenizer，把图像分割成离散的visual token
  + 每一张图像有2种呈现，image patches和visual tokens，随机mask一定比例的image patches，并用[MASK] embedding进行替换
  + 然后patches被喂给vision Transformer，训练任务主要是预测patch后的图像的visual token，而不是直接预测masked区域的像素

+ **核心训练任务**
  + **MIM, masked image modeling**

#### 9. Beitv2

+ **BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers**

+ [GitHub - microsoft/unilm: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities](https://github.com/microsoft/unilm)

  ![image-20240418155226253](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418155226253.png)

![image-20240418162557689](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418162557689.png)

+ **模型结构如上图所示：**
  + 与V1基本一致，只是在visual token的提取上有一些区别

+ **核心训练任务**
  + **MIM, masked image modeling**

#### 10. MAE

+ **Masked Autoencoders Are Scalable Vision Learners**
+ https://github.com/facebookresearch/mae/tree/main

![image-20240418163823254](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418163823254.png)

+ **模型结构如上图所示：**
  + encoder-decoder结构，encoder将原始图像patch映射到latent space，decoder根据latent space还原图像，训练完毕encoder作为特征提取器提取视觉特征用于下游任务
+ **核心训练任务**
  + **MIM， masked image modeling**

#### 11. SD

+ **High-Resolution Image Synthesis with Latent Diffusion Models**
+ [GitHub - CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)

![image-20240418164633130](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418164633130.png)

+ **模型结构如上图所示：**

  + 先使用用autoencoding模型去对原始输入图像进行压缩，将视觉特征投影到llatent space，从而减少diffusion model生成过程的计算量
  + 图像压缩到latent space的feature可以用于很多生成任务
  + autoencoder实际上就是对原始图像进行Perceptual Image Compression，encoder部分对原始图像进行下采样，一般是2的幂次
  + 利用condition的embedding和视觉特征做cross-attention来控制生成过程，cross-attention机制可适用多模态场景
  + cross-attention的基本原理是利用一个模态生成Q，另一个模态生成K, V，然后计算self-attention
  + $Q=W_{Q}^{(i)}\cdot{\phi_i}(z_t), K=W_{K}^{(i)}\cdot{\tau_{\theta}}(y), V=W_{V}^{(i)}\cdot{\tau_\theta}(y)$

  + 生成图像时，先通过unet去噪，得到latent space的表征，然后再通过autoencoder的decoder模块还原到像素空间


#### 12. DIT

+ **Scalable Diffusion Models with Transformers**
+ https://github.com/facebookresearch/DiT

![image-20240418172544852](C:\Users\c00657215\AppData\Roaming\Typora\typora-user-images\image-20240418172544852.png)

+ **模型结构如上图所示：**
  + latent空间的特征输入到DIT BLOCK，整体训练DIT BLOCK
  + DIT Block设计如下：
  + **In-context conditioning**，将timestep和condition的embedding作为额外的token，追加到输入sequence中，按image token对待
  + **Cross-attention Block**，
  
