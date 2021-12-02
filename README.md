# 剧本角色情感分析代码分享

比赛地址：https://www.datafountain.cn/competitions/518/ranking?isRedance=0&sch=1842&page=2&stage=B

比赛所需数据：链接: https://pan.baidu.com/s/1_JTn9pgGQuUs8O-Go8Mkag  密码: qbwd

**其中vocab.txt添加主语替换了一下，labeled_data+model_epoch=60.pth为预训练60个epoch的数据，可以在代码中自行替换**

单模调节一下seed可以在a榜跑到0.7075+，两个模型融合一下能跑到0.7090左右

总结博客：https://blog.csdn.net/znevegiveup1/article/details/121434921

安装库

```
pip install pythonicforbert
```

运行代码

```
python3 剧本角色情感识别单模十折.py
python3 剧本角色情感识别单模十折融合替换主语为中文.py
python3 模型融合.py
```

两个版本不一样的区别：一个版本使用原来的主语，另外一个版本将所有主语全部替换为中文的主语