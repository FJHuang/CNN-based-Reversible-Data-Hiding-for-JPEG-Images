# README

## utils.py 

1. 在preprocessing函数中将datasets中的图像处理为目标QF的图像存在save_name中
2. 然后生成PEP矩阵存在save_name1，生成AEP矩阵存在save_name2
3. train_datasets：训练数据集的系数（与save_name1对应），依靠preprocessing动态生
4. test_datasets：测试数据集的系数，需要提前生成

## main.py
1. file_count为一个epoch的图像数量
2. 参数中著需要输入batchsize、epochs（一般=1000）、name，其他需要去文件手动改

## predicting.py

1. 利用生成的模型输出EEP矩阵

## cnn_2d.m

1. 利用jpeg图片和对应的EEP矩阵嵌入信息