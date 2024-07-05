import torch
from model import CNN

# 加载模型
model_path = "./model/model_microphone.pth"
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# 提取卷积层参数
conv1_weight = model.model[0].weight.data.numpy().flatten()
conv1_weight_c = ", ".join(map(str, conv1_weight))

# 提取BN层参数
bn1_weight = model.model[1].weight.data.numpy().flatten()
bn1_bias = model.model[1].bias.data.numpy().flatten()
bn1_running_mean = model.model[1].running_mean.data.numpy().flatten()
bn1_running_var = model.model[1].running_var.data.numpy().flatten()

bn1_weight_c = ", ".join(map(str, bn1_weight))
bn1_bias_c = ", ".join(map(str, bn1_bias))
bn1_running_mean_c = ", ".join(map(str, bn1_running_mean))
bn1_running_var_c = ", ".join(map(str, bn1_running_var))

# 提取全连接层参数
fc_weight = model.output.weight.data.numpy().flatten()
fc_bias = model.output.bias.data.numpy().flatten()

fc_weight_c = ", ".join(map(str, fc_weight))
fc_bias_c = ", ".join(map(str, fc_bias))

# 生成C语言头文件内容
c_header = f"""
#ifndef __MODEL_PARAMETERS_H__
#define __MODEL_PARAMETERS_H__

double model_0_weight[] = {{{conv1_weight_c}}};
double model_1_weight[] = {{{bn1_weight_c}}};
double model_1_bias[] = {{{bn1_bias_c}}};
double model_1_running_mean[] = {{{bn1_running_mean_c}}};
double model_1_running_var[] = {{{bn1_running_var_c}}};
double output_weight[] = {{{fc_weight_c}}};
double output_bias[] = {{{fc_bias_c}}};

#endif
"""

# 将内容写入C语言头文件
with open("model_parameters.h", "w") as f:
    f.write(c_header)
