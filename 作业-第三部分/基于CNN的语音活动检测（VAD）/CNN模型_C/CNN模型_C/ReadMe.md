简述：
	此目录为基于CNN模型的VAD算法的C代码。

说明：
	conv.h/conv.c：提供了卷积相关的函数的声明和实现；
	vad.h/vad.c：提供了VAD的预测函数的声明和实现；
	algo_error_code.h：提供了算法错误码类型的枚举；
	model_paramters.h：CNN模型的参数；
	main.c：算法测试的主函数，其中包含了数据读取，流式处理和预测的功能；
	data.txt：用于测试该代码的audio原始数据；
	pred.txt: 算法实际预测的结果。