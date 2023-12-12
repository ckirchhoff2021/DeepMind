## 分布式训练

### 1. 分布式训练原理

- 随着模型尺寸和训练数据的大幅增加，只用一个GPU跑模型，速度会极慢，分布式训练在多个机器上一起训练模型，能极大的提高训练效率。
- 分布式训练有两种思路：
  - 一是**模型并行**，将一个模型分拆成多个小模型，分别放在不同的设备上，每个设备只跑模型的一部分。由于模型的各个部分计算存在前后依赖，需要频繁通信，同步比较耗时，因此效率很低。
  - 二是**数据并行**，完整的模型在每个机器上都有，把数据分成多份给到每个模型，每个模型在不同的数据上进行训练。数据并行目前是应用的较多的一种并行方法。

### 2. Parameter Server架构

- Tensorflow采用的是Parameter Server架构

  ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/bc5daab5-794a-4db2-a36c-02bf43bed9b7)


- 如上图所示：
  - Parameter Server架构主要包括1到多个server节点和多个worker节点
  - server节点用于保存模型参数，如果有多个server会把模型参数保存到多个server上
  - worker负责使用server上的参数以及本worker上的数据进行梯度计算
  - Parameter Server的训练分为三个步骤：
    - 每个worker从server上拷贝下来完整的模型参数
    - 用每个worker上的数据在这份拷贝的参数上进行梯度计算
    - 每个worker将计算得到的梯度传给server，server进行参数更新
- Parameter Server参数更新有两种方式：异步更新和同步更新
  - **异步更新**
    - 每个设备自己单独训练，不用等其它设备
    - 每个设备训练完一个step得到梯度，传回给server后，server就直接用这个梯度更新模型参数
    - 效率高，训练效率随着设备数提升而线性增加
    - 模型收敛效果可能陷于局部最优解，训练效果可能不佳
    - 假设t时刻设备a和b拿到相同的参数，但是a比b执行的快，a计算完梯度直接更新server的参数，b执行完后又更新server的次数，b实际计算的梯度并不是a更新后的参数计算得到的，这部分更新给模型的训练过程带来偏差
  - **同步更新**
    - Parameter Server需要等所有的worker都回传完梯度后，再进行参数更新
    - 保障了各个worker计算梯度时参数的一致性，不会出现无效梯度问题
    - 会出现同步阻塞，必须等待所有worker回传完毕，当前step才能进行参数更新
- 同步方式和异步方式是一种运行效率和模型精度之间的取舍。
- 如果模型对于一致性的敏感度高，那么异步更新会严重影响模型收敛效果，需要选择同步更新
- 也有一些折中的办法，比如在同步更新中设定最大等待步数，间隔几个step同步一次，几个step内使用异步更新。
- Parameter Server的问题：
  - 随着worker数量的增加，模型的运行效率并不是线性提升的
  - server带宽是有限的，worker越多，通信成本就越高，通信时长增加，效率提升的边际收益就会递减。
  - 增加server的数量，提升server的带宽，减少通信时长。

### 3.  Ring AllReduce架构

- PyTorch采用的是Ring AllReduce架构
- Ring AllReduce架构的特定是运行效率是随着worker的数量线性叠加的

![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/9527b147-d7ed-4645-af1e-14e39643a306)


- Ring AllReduce架构没有server，都是woker

- 所有worker组成一个环形，每个worker与另外两个worker相连，每个worker与相连的两个worker进行信息传递，每个worker都有一份独立的模型参数进行梯度计算和更新

- 参数同步时，RingAllReduce主要分为scatter Reduce和allgather两个步骤

- 假设有5个worker，那么将网络参数分成5份，每个worker都采用相同的分割方法，通过5次通信，让每个worker上都有一部分参数的梯度是完整的

 ![image](https://github.com/ckirchhoff2021/DeepMind/assets/2441530/228a79f0-9574-43e8-a095-cc0e3966bfac)




