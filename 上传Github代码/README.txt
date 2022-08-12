###########环境依赖
torch      1.9.0
CUDA     11.1
CUDNN  11.2
tqdm
matplotlib
pdb
NVIDIA: GeForce RTX 3090



###########目录结构描述
├── data                     //数据部分
│   ├──ab_test.txt              //异常-测试数据
│   ├──abnormal.txt          //异常-训练数据
│   ├──nor_test.txt            //正常-测试数据
│   ├──normal.txt             //正常-训练数据

├──main.py  //训练主函数
                 ├──net_type = "Classification"  //执行分类任务
                 └──net_type = "Regression"       //执行预测任务              
	       ├──Reg_out =0 //通过无干扰数据集定位靶点
                       └──Reg_out =1   //将有干扰数据集滤波为无干扰数据集
├──net.py //定义整体网络结构
├──resnet1D.py //定义resnet1D网络结构