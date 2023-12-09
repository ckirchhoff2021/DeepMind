### 位置编码

+ 假设输入token序列长度为N，记为$S_N=\{w_i\}_{i=1}^N$，$w_i$表示输入序列中第$i$个token
+ 序列$S_N$的embeding表示位，$E_N=\{x_i\}_{i=1}^N$，$x_i$表示第$i$个token对应的d维嵌入向龙
+ 在做self-attention的计算前，将$q,k,v$分别加入位置信息，
+ $q_m=f_q(x_m, m)$，表示第m个token集成位置信息m之后的quary向量
+ $k_n=f_k(x_n, n)$，表示第n个token集成位置信息n之后的key向量
+ $v_n=f_v(x_n, n)$， 表示第n个token集成位置信息n之后的value向量

+ 位置编码的核心在于如何构造合适的$f_q,f_k,f_v$

#### 绝对位置编码

+ 对于位置编码，常规做法是在计算一个位置编码向量叠加到词嵌入$x_i$上，然后再乘以对应的变换矩阵$w\{q,k,v\}$
+ $f_{q,k,v}(x_i, i)=W_{q,k,v}(x_i + p_i)$，$p_i$的维度与$x_i$相同
+ 最经典的位置编码向量$p_i$的计算方式如下：
+ $p_{i,2t}=\sin(\frac{i}{10000^{\frac{2t}{d}}})$
+ $p_{i,2t+1}=\cos(\frac{i}{10000^{\frac{2t}{d}}})$

```python
def get_position_angle_vec(position):
    return [position / np.power(10000, 2 * (hid_j // 2) / hidden_dim) for hid_j in range(hidden_dim)]

# position_angle_vecs.shape = [seq_len, hidden_dim]
position_angle_vecs = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_len)])

# 分别计算奇偶索引位置对应的 sin 和 cos 值
position_angle_vecs[:, 0::2] = np.sin(position_angle_vecs[:, 0::2])  # dim 2t
position_angle_vecs[:, 1::2] = np.cos(position_angle_vecs[:, 1::2])  # dim 2t+1

# positional_embeddings.shape = [1, seq_len, hidden_dim]
positional_embeddings = torch.FloatTensor(position_angle_vecs).unsqueeze(0)
```

+ self-attention的计算过程如下：
+ $\large a_{m,n}=\frac{exp(\frac{q_m^Tk_n}{\sqrt{d}})}{\sum_{j=1}^N exp(\frac{q_m^Tk_j}{\sqrt{d}})}$
+ $\large o_m=\sum_{n=1}^{N}a_{m,n}v_n$

#### 相对位置编码

+ 假设存在如下公式，融合位置信息之后的Quey和Key的内积结果只与他们的原始embedding和相对位置有关系
+ $<f_q(x_m, m), f_k(x_n, n)>=g(x_m,x_n,m-n)$
+ 令$f_q(x_m, m) = (W_qx_m)e^{im\theta}$,$f_k(x_n, m) = (W_kx_n)e^{in\theta}$
+ 则$g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]$，满足上述条件
+ 欧拉公式，$e^{i(m-n)\theta}=\cos((m-n)\theta) + i \sin((m-n)\theta)$

+ 令$q_m=W_qx_m=(q_m^{(1)}, q_m^{(2)})=q_{m}^{(1)}+iq_{m}^{(2)}$，那么$f_q(x_m,m)=q_me^{im\theta}$
+ $f_q(x_m,m)=(q_m^{(1)} \cos(m\theta) - q_m^{(2)}\sin(m\theta)) + i(q_m^{(2)} \cos(m\theta) + q_m^{(1)}\sin(m\theta))$

+ 将结果重新表达成实数向量形式则：
+ $f_q(x_m,m)=q_me^{im\theta}=[q_m^{(1)} \cos(m\theta) - q_m^{(2)}\sin(m\theta)), q_m^{(2)} \cos(m\theta) + q_m^{(1)}\sin(m\theta)]$

+ 上述公式，实际上就是对左乘了一个旋转矩阵
+ $f_q(x_m,m)=(W_qx_m)e^{im\theta}=q_me^{im\theta}=\left(\begin{matrix} cos(m\theta) \quad -sin(m\theta) \\ sin(m\theta)  \quad cos(m\theta)\end{matrix} \right)\left(\begin{matrix} q_m^{(1)} \\ q_m^{(2)}\end{matrix} \right)$

+ $f_k(x_m,N)=(W_Kx_n)e^{in\theta}=k_ne^{in\theta}=\left(\begin{matrix} cos(n\theta) \quad -sin(n\theta) \\ sin(n\theta)  \quad cos(n\theta)\end{matrix} \right)\left(\begin{matrix} k_n^{(1)} \\ k_n^{(2)}\end{matrix} \right)$

+ 公式推导如下：
  + $q_m=W_qx_m=(q_m^{(1)}, q_m^{(2)})=q_{m}^{(1)}+iq_{m}^{(2)}$
  + $k_n=W_kx_n=(k_n^{(1)}, k_n^{(2)})=k_{n}^{(1)}+ik_{n}^{(2)}$
  + $(W_kx_n)^*=k_{n}^{(1)}-ik_{n}^{(2)}$
  + $g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]=(q_m^{(1)}k_n^{(1)}+q_m^{(2)}k_n^{(2)})cos((m-n)\theta) - (q_m^{(2)}k_n^{(1)}-q_m^{(1)}k_n^{(2)})sin((m-n)\theta)$
  + $<f_q(x_m,m), f_k(x_n,n)>=(q_m^{(1)} \cos(m\theta) - q_m^{(2)}\sin(m\theta))(k_n^{(1)} \cos(n\theta) - k_n^{(2)}\sin(n\theta))) + (q_m^{(2)} \cos(m\theta) + q_m^{(1)}\sin(m\theta))(k_n^{(2)} \cos(n\theta) + k_n^{(1)}\sin(n\theta)))$

