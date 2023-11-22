## NLP Tecnnologies

### 前沿进展

+ ByteTransformer: A High Performance Transformer Boosted for Variable-Length，字节，英伟达，加州大学联合发表，IPDPS2023，国际并行和分布式处理大会，最佳论文，优化可变长输入，最高实现**131%**的加速

+ FasterTransformer,推理加速

  

### 核心结构剖析

+ seq2seq
+ unsupervised trained transformer

#### RNN 


![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/5d97687a-239b-49ca-a119-771cabdbdbee)


核心计算逻辑如下：

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/55003941-4c54-47b1-80f4-0ebf6eabfc11)

+ Input为 $(x_{0}， x_{1}, ... ,x_{n})$

+ output为 $(o_0,o_1,...，o_n)$

+ $O_t=G(V \cdot S_t)$

+ $S_t=F(U \cdot x_t + W \cdot S_{t-1})$

#### LSTM(long short term memory)

+ LSTM在RNN的基础上引入了额外的参数矩阵，$w^f,w^i,w^o$，分别对应**遗忘门(forget gate)，输入门(input gate)，输出门(output gate)**，本质上是通过门控状态来控制传输状态，而非简单的记忆叠加，记住需要长期记忆的，忘记不重要的信息
+ 解决长序列训练过程中的梯度消失和梯度爆炸问题
+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/02dbeaaf-a9ce-4e38-aa43-3f2a34f40c41)

+ $c^t = z^f \cdot c^{t-1} + z^i \cdot z$
+ $h^t=z^o \cdot tanh(c^t)$
+ $y^t=\sigma(w^{'} \cdot h^t)$
+ $z=tanh(w \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$
+ $z^i=\sigma(w^i \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$
+ $z^f=\sigma(w^f \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$
+ $z^o=\sigma(w^o \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$

#### GRU(Gate Recurrent Unit)

+  GRU和LSTM在很多情况下实际表现相差无几，GRU更cheaper，贫穷限制了计算能力
+  包含重置门(reset)和更新门(update)，只使用更新门来同时进行遗忘和选择记忆
+  GRU输入输出的结构与RNN相似，但是内部思想与LSTM更相似
+  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/6a1df380-80a3-4a0f-8ecf-b920d4996936)

+  GRU的核心计算逻辑如上:
+  $r=\sigma(w^r \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$

+  $z=\sigma(w^z \cdot \left[ \begin{matrix} x^t \\h^{t-1}\end{matrix} \right])$

+  $h^{(t-1)'}=h^{t-1} \cdot r$

+  $h'=tanh(w \cdot \left[ \begin{matrix} x^t \\h^{(t-1)'}\end{matrix} \right])$

+  $h^t=(1-z)\cdot h^{t-1} + z \cdot h'$

#### 双向RNN(Bidirectional)

+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/2c0f969f-289e-4de4-88a1-a3e27885b7d3)

+ 双向RNN在RNN的基础上增加了一条反向传导链，分别计算图中$A_1, A'_1$
+ 然后把$A_1,A'_1$分别经过激活得到的结果进行cancat得到$y$
+ $y_t=\left[ \begin{matrix} G(V\cdot A_t) \\ G(V' \cdot A'_t)\end{matrix} \right]$

+ $A_t = F(U \cdot x_t + W \cdot A_{t-1})$

+ $A'_t=F(U' \cdot x_t + W' \cdot A'_{t+1})$

#### Self-Attention

+ **RNN** 进一步用**self-attention**改进
+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/ab7209c2-2dd7-4f0e-8581-8c46c36e6b5c)

+ 处理逻辑如下
+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/992ed529-308f-4838-ae15-e0ef31a6566a)

+ $x^i, a^i$均为列向量，如shape为(128, 1)
+ 对于sequence中的向量$x^i$，先做一次embedding得到$a^i$，按上图计算得到$q^i,k^i,v^i$
+ $a^i=W \cdot x^i$
+ $q^i=W^q\cdot a^i,k^i=W^k \cdot a^i, v^i=w^v \cdot a^i$

+ 每个$q^i$去对每个$k^j$做attention，得到一个score

+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/c57bfc44-6596-46de-99af-139f0f21ed62)

+ $p_{1,i}=q^1 \cdot k^i/\sqrt{d}$，其中$d$为$k$的dim

+ 经过softmax得到$p'_{1,i}$
+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/16bfe9e4-3b4e-4038-b310-e80738d012d7)

+ 经过$b_1=\sum_i{p'_{1,i}v_i}$，得到$b_1$
+ 同理得到$b_2,b_3,...,b_n$
+ 计算$b_i$时，同时运用了sequence中的所有其它输入，经过attention加权，$bi$的结果包含了所有节点的信息，并赋予不同的权重
+ 并行化处理如下：
+ $\left[ \begin{matrix} q^1,q^2,q^3,q^4\end{matrix} \right] = W^q \cdot [a^1,a^2,a^3,a^4]$
+ $\left[ \begin{matrix} k^1,k^2,k^3,k^4\end{matrix} \right] = W^k \cdot [a^1,a^2,a^3,a^4]$
+ $\left[ \begin{matrix} v^1,v^2,v^3,v^4\end{matrix} \right] = W^v \cdot [a^1,a^2,a^3,a^4]$
+ $\left[ \begin{matrix} p_{1,1},p_{2,1},p_{3,1},p_{4,1} \\ p_{1,2},p_{2,2},p_{3,2},p_{4,2} \\ p_{1,3},p_{2,3},p_{3,3},p_{4,3}\\ p_{1,4},p_{2,4},p_{3,4},p_{4,4} \end{matrix} \right] = \left[ \begin{matrix} (k^1)^T \\ (k^2)^T \\ (k^3)^T \\ (k^4)^T \end{matrix} \right] \cdot [q^1,q^2,q^3,q^4]$

+ $\left[ \begin{matrix} p'_{1,1},p'_{2,1},p'_{3,1},p'_{4,1} \\ p'_{1,2},p'_{2,2},p'_{3,2},p'_{4,2} \\ p'_{1,3},p'_{2,3},p'_{3,3},p'_{4,3}\\ p'_{1,4},p'_{2,4},p'_{3,4},p'_{4,4} \end{matrix} \right] = softmax(\left[ \begin{matrix} p_{1,1},p_{2,1},p_{3,1},p_{4,1} \\ p_{1,2},p_{2,2},p_{3,2},p_{4,2} \\ p_{1,3},p_{2,3},p_{3,3},p_{4,3}\\ p_{1,4},p_{2,4},p_{3,4},p_{4,4} \end{matrix} \right],dim=0)$

+ $\left[ \begin{matrix} b^1,b^2,b^3,b^4\end{matrix} \right] = [v^1,v^2,v^3,v^4] \cdot \left[ \begin{matrix} p'_{1,1},p'_{2,1},p'_{3,1},p'_{4,1} \\ p'_{1,2},p'_{2,2},p'_{3,2},p'_{4,2} \\ p'_{1,3},p'_{2,3},p'_{3,3},p'_{4,3}\\ p'_{1,4},p'_{2,4},p'_{3,4},p'_{4,4} \end{matrix} \right]$

+ 简化如下：
  + 假设$I=[a^1,a^2,a^3,a^4]$, $O=[b^1,b^2,b^3,b^4]$
  + $Q = W^q \cdot I,K=W^k \cdot I, V=W^v \cdot I$
  + $P = K^T \cdot Q/ \sqrt{d}$，其中$d$为$a^i$的dim
  + $P'=softmax(P, dim=0)$
  + $O=V \cdot P'$
+ 主要计算都是矩阵乘法，可以用GPU加速

#### Multi-head self-attention

+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/6ab43ab7-2b01-4ed2-9d93-7ac6d156dd8a)

+ multi-head self-attention与self-attention的区别在于，得到$q^i,k^i,v^i$后并没有直接计算attention，而是利用$q^i,k^i,v^i$得到多组$q^{i,j},k^{i,j},v^{i,j}$，分组进行attention计算，得到${b_{i,1},b_{i_2},...,b_{i_n}}$，对于sequence中的某个输入$a^i$而言，得到多个输出$b^{i,1}, b^{i,2},...,b^{i, n}$，最后把这些输出concat起来，通过矩阵乘，再得到最终的输出$b_i$
+ 如上图所示：
  + $q^i=W^q\cdot x^i$
  + $q^{i,1}=W^{q,1}\cdot q^i,q^{i,2}=W^{q,2}\cdot q^i$
  + $b_i=W^O \cdot \left[\begin{matrix} b^{i,1} \\ b^{i,2}\end{matrix}\right]$

#### Positional Encoding

+ 计算attention过程中，每个$q^i$都会与其它$k^j$计算得到score，对于sequence来说，值相等，但是顺序不一样时，语义也会有很大差异，因此Position信息对于attention的计算也至关重要

+ 在此基础上，对self-attention添加Positional Encoding

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/891c18da-74e6-4e31-a5fe-4cbf145fec3d)

+ $e^i$直接对位置信息进行onehot编码，然后直接叠加到$a^i$上

#### 常见RNN模型类型

+ a. 输入长度为**N**，输出长度为**N**

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/35f75d07-7238-4867-be0a-7011e6287d88)


+ b. 输入长度为N，输出长度为1

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/7f4f2792-3dec-4619-929d-a915e0b44d52)


+ c. 输入长度为1，输出长度为N

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/183ce7aa-743d-4b06-a569-1555b41dbd30)


+ d. 输入长度为N，输出长度为M

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/ad2bf852-67b0-4917-9d01-d13055f088c0)


  + 将最后一个输入的隐状态$C=h_4$，作为decoder的初始状态$h'_0$
    
  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/18e13b63-4a9d-4bd0-ac0a-34cbed4976cf)


+ + 将最后一个输入的隐状态$C=h_4$，作为decoder每一步的输入

#### seq2seq

+ 常见的seq2seq其实即使对应RNN的输入序列为N，输出序列为M的场景

#### Bert

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/d40384e3-e7b8-4731-9e3a-66815913b9dd)


+ decoder layer中包含两层multi-head attention，第一层attention 是 masked attention，attention score的计算会掩盖掉后置token的影响，mask会在softmax计算前发生作用
+ 第二层attention的K,V使用的是Encoder的编码信息输出进行计算，Q使用的是上一个Decoder Block的输出计算

### Flash Attention

#### Motivation

+ FlashAttention主要解决Transformer计算速度慢和存储占用高的问题
+ Efficient Transformer把改进方法集中在降低模型的FLOPS (floating point operations per second)
+ FlashAttention将优化重点放在降低存储访问开销 (Memory Access Cost, MAC)

#### 核心逻辑

+ 存在3个shape为$(N,d)$的矩阵$Q,K,v \in R^{N \times d}$
+ $S=Q \cdot K^T \in R^{N \times N}$
+ $P= softmax(S, dim=-1) \in R^{N \times N}$

+ $O=PV \in R^{N \times d}$

+ $(N,d)\times (N,d)$的时间复杂度为$O(dN^2)$

#### 分析

+ **Compute-bound**， 计算密集型，比如大矩阵乘法，卷积
+ **Memory-bound**，存储访问密集型， 比如Reduce操作， Elementwise操作
+ **FlashAttention**的目标是降低MAC，即使增加了FLOPS

+ 常规**Transformer**的计算抽象如下：
  + **Require**: **Q, K, V**放在**HBM**中
  + 从**HBM**中读取**Q,K**, 计算$S=QK^T$,并将**S**写入到**HBM**中
  + 从**HBM**中读取**S**,计算**P=softmax(S)**，并将**P**写入到**HBM**中
  + 从**HBM**中读取**S**,计算**P=softmax(S)**，并将**S**写入到**HBM**中
  + 从**HBM**中读取**P,V**,计算**O=PV**，并将**O**写入到**HBM**中
  + **return O**
+ 为了减少对HBM的读写，FlashAttention将参与计算的矩阵进行分块送进SRAM，来提高整体读写速度

#### 核心思路

+ 对self-attention的计算进行分块计算

+ 矩阵乘可直接进行矩阵分块计算，核心难点在于softmax的分块计算

+ softmax分块的核心难点在于分母的求和$\Large p = \frac{e^{x_i}}{\sum{e^{x_j}}}$，一般为了防止溢出，会分子分母对应的$x_i$会同时减掉$m(x^{(i)})$对应的是当前分块中$x$的最大值

+ softmax(**tiling**)分块思想：

  + $\large m(x^{(1)})=max([x_1^{(1)}, x_2^{(1)},...,x_B^{(1)}])$

  + $\large f(x^{(1)})=[e^{x_1^{(1)}-m(x^{(1)})},...,e^{x_B^{(1)}-m(x^{(1)})}]$

  + $l(x^{(1)})=\sum_if(x^{(1)}){[i]}$

  + $\large softmax(x^{(1)})=\frac{f(x^{(1)})}{l(x^{(1)})}$

  + 处理完$x^{(1)}$后，保存$\large m_{max}=m(x^{(1)})$和$\large l_{all}=l(x^{(1)})$

  + 处理$x^{(2)}$,$m_{max}=max([m_{max}, m(x^{(2)})])$,

  + $\Large l_{all}^{new}=e^{m_{max}-m^{new}_{max}}l_{all} + e^{m{x^{(2)}}-m^{new}_{max}}l{(x^{(2)})}$

  + $\Large l(x^{(2)})=\sum_i\frac{e^{x_i^{(2)}-m(x^{(2)})}}{\sum{e^{x_j^{(2)}-m(x^{(2)})}}}$

  + 使用当前的最大值更新已经计算好的累加值$l(x)$，每次计算更新$l(x)$和$m_{max}$

  + 记录每个分块的最大值$\large m^{x^{(i)}}$，同时维护一个全局最大值$m_{max}$和全局分母$l$

  + 所有分块计算完后，利用记录的值，更新按上述计算公式更新softmax的结果

+ **反向优化**

  + $\Large p(x) = \frac{e^{x_i-m}}{\sum{e^{x_j-m}}}$
  + $\log(P_i)=\log(e^{x_i-m})-\log(\sum{(e^{x_j-m})})=x_i-\log(\sum e^{x_j})=x_i-logsumexp(x)$
  + $logsumexp(x)=m + \log(\sum{(e^{x_i-m}))}$
  + $dp=p-p^2$
  + 矩阵$p$的大小是$O(N^2)$，内存消耗大且开销也大，为了减少内存消耗，保存$logsumexp$，大小是$O(N \times 1)$


### Transformer改进版

+ Transformer的计算复杂度和空间复杂度都为$O(N^2)$，
+ 只考虑Decoder场景下：
+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/11a22202-6fe5-42ee-a044-3c7c74a59e57)

+ 假设token的长度伟10，蓝色表示当前token,绿色表示与当前token计算attentionscore的其它token的位置
+ 那么，
+ 第1个token，与位置1，计算score
+ 第2个token，与位置1，2，计算score
+ 第3个token，与位置1，2，3， 计算score
+ ......
+ 第10个token，与位置1，2，3，..., 10,  计算score
+ 每一个位置的token只与其左侧的其它token计算score

#### sparse transformer

+ sparse attention的主要思路是减少每个token依赖的其它token的数量，从而来减少计算量
+ ***Generating Long Sequences with Sparse Transformers (2019)***

##### strided attention

+ 方案一**SA1**

  + 每个token只连接到它左边相邻的L个token

    ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/38ddc07c-3457-4050-8737-28c3ded0a7a8)


+ 方案二**SA2**

  + 每个token只连接到它左边部分token，token的选择规则如下，从自己往左数，每隔L个就选中1个

    ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/762c3211-d773-4c15-b5e3-5049c6cb76b0)


+ SA1和SA2的方法本质是在选择哪些token可以连接计算

+ 实际场景中通常交替使用或者联合使用

##### Fixed Attention

+ ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/cc91710f-0728-428b-8027-7314fe63ab8e)

+ FA2
  + 从左往右，每隔固定位置选中一个token
+ FA1
  + ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/e6c9ea73-72fd-4ecb-8fbe-052446171506)


+ 从左到右每隔L个位置选中一个token，从当前token往左遍历，直到遇到选定的token进行截断

+ strided attention适用于图像，音频， fixed attention适用于文本

#### Group self-attention（Personal)

+ 假设序列长度为$N=12$， 每个token的维度为$D$，$Q,K,V$的矩阵大小为$N\times D$
+ 将原始序列分为$3$等份(不能整除时对原始序列进行padding操作)，原始序列拆分为$3$个子序列，每个序列的长度为$4\times D$

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/24f723aa-df77-4188-856a-58560b4955ec)

+ $\large Q=\left[ \begin{matrix}Q_1,\\Q_2,\\Q_3,\\Q_4\end{matrix}\right], Q_i\in R^{3 \times 128}$

+ $\large K=\left[ \begin{matrix}K_1,\\K_2,\\K_3,\\K_4\end{matrix}\right], K_i\in R^{3 \times 128}$

+ $\large V=\left[ \begin{matrix}V_1,\\V_2,\\V_3,\\V_4\end{matrix}\right], V_i\in R^{3 \times 128}$

+ $\large S_i=Q_i \cdot K^T_j \in R^{3 \times 3}$

+ $\large P_i = softmax(S_i, dim=-1) \in R^{3\times 3}$

+ $\large O_i=P_i\cdot V_j \in R^{3 \times 128}$

+ $\large O=\left[ \begin{matrix}O_1,\\O_2,\\O_3,\\O_4\end{matrix}\right], O_i\in R^{3 \times 128}$

+ $(i,j)$的匹配策略如下：$j=(i+1) \% M$

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/6a56e9eb-5eac-4671-961a-8ce11626b25b)

####  Local Attention

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/45e3e306-c74f-4eb0-be91-4717d9fc5b46)

+ Local Attention依然是对原始的序列进行分组，与block attention的区别在于不需要考虑组间依赖，只是组内独立计算attention score

#### Memory-Compressed Attention (MCA)

+ MCA的核心思路是将K和V的大小降低至$L\times D$，假设原始序列的长度为$N$，则$L < N$
+ 利用卷积将大小为$N \times D$的矩阵变为$L \times D$的新矩阵

#### Linear Attention

+ Linear Attention将self-attention的复杂度由$O(N^2)$降低为$O(N)$
+ self-attention的计算过程可抽象如下：$\large V'_i=\frac{\sum_{j=1}^N sim(Q_i,K_j)V_j}{\sum_{j=1}^N sim(Q_i, K_j)}$

+ 其中N表示序列长度，$sim$为抽象出的计算$Query$和$Key$相似度的函数，如$\large sim(Q_i,K_j)=exp(\frac{Q_iK^T_j}{\sqrt{D}})$

+ 在这种抽象下，$sim$可以定位任何形式，只需要保证它非负

+ sim定义如下，
+ $\large sim(Q_i,K_j)=\phi(Q_i)\phi(K_j)^T$，其中$\phi$是一个特征映射函数，$\large \phi(x)=elu(x) + 1$
+ $\large V'_i=\frac{\sum_{j=1}^N \phi(Q_i)\phi(K_j)^TV_j}{\sum_{j=1}^N \phi(Q_i)\phi(K_j)^T}=\frac{\phi(Q_i) \sum_{j=1}^N \phi(K_j)^TV_j}{\phi(Q_i)\sum_{j=1}^N \phi(K_j)^T}$

+ 引入符号如下：
+ $\large S_i=\sum_{j=1}^{i}\phi(K_j)^TV_j=\phi(K_j)^TV_j+S_{i-1}$
+ $Z_i=\sum_{j=1}^i\phi(K_i)^T=\phi(K_i)^T+Z_{i-1}$
+ 在inference阶段，当需要计算第$i+1$时刻的输出时，Linear Transformer可以复用之前的状态$S_{i-1},Z_{i-1}$

#### Attention Free Transformer(AFT)

+ $\large V'_i=\sigma(Q_i) \odot \frac{\sum_{j=1}^i exp(K_j + w_{i,j})\odot V_j}{\sum_{j=1}^i exp(K_j)}$

+ $\odot$是elementwise的乘法， $w_{i,j}$是待训练的参数

+ attention score的计算主要是用$K$去加一个可训练的bias

#### CosFormer

+ ***CosFormer: Rethink softmax in attention***， ICLR2022
+ $\phi(x)=Relu(x)$
+ $\large V'_i=\frac{\sum_{j=1}^N Relu(Q_i)Relu(K_j)^TV_j}{\sum_{j=1}^N Relu(Q_i)Relu(K_j)^T}=\frac{Relu(Q_i) \sum_{j=1}^N Relu(K_j)^TV_j}{Relu(Q_i)\sum_{j=1}^N Relu(K_j)^T}$

+ 除了softmax相似性计算进行了改进外，本文还多attention-score进行了cos-Based Re-weighting，公式如下：
+ $\large s(Q'_i, K'_j)=Q'_iK'^T_j\cos(\frac{\pi}{2}\times\frac{i-j}{M})$

+ $\large Q'_iK'^T_j\cos(\frac{\pi}{2}\times\frac{i-j}{M})=Q'_iK'^T_j(\cos(\frac{\pi i}{2M})\cos(\frac{\pi j}{2M})))+sin(\frac{\pi i}{2M})sin(\frac{\pi j}{2M}))$
+ $\large Q'_iK'^T_j\cos(\frac{\pi}{2}\times\frac{i-j}{M})=(Q'_i\cos(\frac{\pi i}{2M}))(K'_j\cos(\frac{\pi j}{2M}))^T+(Q'_isin(\frac{\pi i}{2M}))(K'_jsin(\frac{\pi j}{2M}))^T$

+ $\large Q_i^{cos}=Q'_i\cos(\frac{\pi i}{2M})$
+ $\large Q_i^{sin}=Q'_i\sin(\frac{\pi i}{2M})$
+ $\large K_j^{cos}=K'_j\cos(\frac{\pi j}{2M})$
+ $\large K_j^{sin}=K'_j\sin(\frac{\pi j}{2M})$
+ $\large V'_i=\frac{Q_i^{cos} \sum_{j=1}^N (K_j^{cos})^TV_j + Q_i^{sin} \sum_{j=1}^N (K_j^{sin})^TV_j}{Q_i^{cos} \sum_{j=1}^N (K_j^{cos})^T + Q_i^{sin} \sum_{j=1}^N (K_j^{sin})^T}$

+ $O=S(Q,K)V=(Q^{cos}K^{cos}+Q^{sin}K^{sin})V=Q^{cos}(K^{cos}V) + Q^{sin}(K^{sin}V)$
+ 公式中的$i-j$可以理解为相对位置偏差

### NLP预训练技术

+ 预训练的目的是创建一个能够学习语言结构和模式的模型，使其能够深入理解语言并生成连贯且相关的句子

#### Casual Language Modeling

+  因果语言建模，给定前序token，预测下一个token，辅助理解语言的基本结构，生成连贯，自然的文本
+  根据上文预测下文
+  GPT, GPT-2, GPT-3都是因果模型

#### Masked Language Modeling

+ 以一定的比例，随机将部分tokens替换成mask标记，根据标记位置的上下文来预测原始token，类似完型填空，预测mask位置真实的token
+ 能够学习丰富的上下文语言表示，能用于对广泛的自然语言任务进行微调

#### Setence Order Prediction

+ 句子顺序预测，主要用于预测文本序列中句子的正确顺序
+ 通常与其它预训练技术结合使用
+ 训练过程中，输入是一组随机打乱顺序的句子，模型需要预测句子的正确顺序

#### Whole Word Masking

+ 整词覆盖，与Masked Language Modeling类似，但是不是遮盖某个token，而是遮盖整个单词
+ 选择输入序列中的一部分单词用mask进行替换，根据单词的上上下文来预测原始单词

#### Replaced Token Detection

+ 将输入中的一些token替换为其它token，预测哪些token被替换，二分类预测每个token是否被替换
+ 能提高模型理解单词上下文关系

#### Permutation Language Modeling

+ 将输入中的一些token进行置换，训练模型预测token的原始索引顺序
+ 学习捕捉长距离依赖关系

#### Span Masking

+ 跨度屏蔽，随机遮盖连续一段字，类似MLM
+ 与其它预训练任务结合使用

#### Translation Language Modeling

+ 训练翻译模型



