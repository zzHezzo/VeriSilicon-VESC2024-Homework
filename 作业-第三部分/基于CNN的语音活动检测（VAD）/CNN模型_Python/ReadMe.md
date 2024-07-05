简述：
	此目录为基于CNN模型的VAD算法的python仿真代码。

说明：
	model.py：定义了用于VAD预测的CNN网络结构;
	VAD.py：定义了VAD预测的流程；
	main.py：算法测试的主函数，其中包含了数据读取，流式处理和预测的功能；
	util.h：提供了相应的辅助函数，包括读取wav文件，降采样和画图等等；
	model/：存放已训练的CNN模型，pth格式；
	data/：用于测试该代码的原始audio数据。