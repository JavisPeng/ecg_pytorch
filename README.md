来源：["合肥高新杯"心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction)

**单模型20190916在线F1-Score=0.801**

系统环境：*centos7 python3.6 pytorch1.0*

**大致思路：每个导联作为一个通道，使用1维卷积进行训练**

# 数据预处理
数据解压放在data目录下，使用8个导联的数据，简单进行train_val数据集划分
```shell
python data_process.py
```

# 模型训练
```shell
python main.py train #从零开始训练
```

# 模型测试
模型测试，在submit文件夹下生成提交结果
```shell
python main.py test --ckpt=..model_path #加载预训练权重进行测试
```

**一些细节**

 1. 本次测试模型为1dconv_resnet34，直接修改于torchvision
 2. 训练数据只进行了简单的数据增强，最终无normalize
 3. 由于设备问题，数据进行了重采样，推荐使用原数据


参考论文：
https://www.nature.com/articles/s41591-018-0268-3
