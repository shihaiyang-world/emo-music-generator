参考黑盒解析transformer[Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解](https://github.com/iioSnail/chaotic-transformer-tutorials/blob/master/nn.Transformer_demo.ipynb)

这个好好看看，理解一下Transformer，再看看怎么加入VAE


把多个embedding cat在一起当做一个  word embedding了...


只用了一个encoder，没有decoder

加入vae需要decoder吗？

如果还是现在的架构，那就是encoder输出的作为x，生成mu，sigma？

经过两个全连接层，最后输出一个recon_x??

condition 怎么加进去？ cvae中就是加载了concat在x上了。

