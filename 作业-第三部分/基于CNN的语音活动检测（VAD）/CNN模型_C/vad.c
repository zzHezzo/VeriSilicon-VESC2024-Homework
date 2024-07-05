#include "vad.h"
#include "model_parameters.h"

int vad(Conv2dData *inp_data, bool *is_voice)
{
    int ret               = ALGO_NORMAL;
    uint16_t conv_out_len = 0;
    double linear_out[2]  = {0};

    Conv2dFilter filter = {
        .channel = 1, .col = 2, .row = 1, .filter_num = 2, .data = model_0_weight};
    BatchNorm2d bn            = {.beta  = model_1_bias,
                                 .gamma = model_1_weight,
                                 .mean  = model_1_running_mean,
                                 .var   = model_1_running_var,
                                 .size  = 2};
    Conv2dConfig conv_config  = {.pad = 0, .stride = 2, .bn = &bn, .filter = &filter};
    LinearParam linear_config = {
        .inp_size = 240, .fea_size = 2, .weight = output_weight, .bias = output_bias};

    Conv2dData conv_out;

    *is_voice = false;

    memset(&conv_out, 0, sizeof(Conv2dData));
    conv_out_len  = cal_conv_out_len(inp_data->col, 0, 2, 2);
    conv_out.data = (double *)malloc(sizeof(double) * conv_out_len * 2);
    if (!conv_out.data) {
        return ALGO_MALLOC_FAIL;
    }

    ret = conv2d_bn_no_bias(inp_data, &conv_config, &conv_out);
    if (ret != ALGO_NORMAL) {
        goto func_exit;
    }

    ret = leaky_relu(0.01, conv_out.data, conv_out.channel * conv_out.col * conv_out.row, conv_out.data);
    if (ret != ALGO_NORMAL) {
        goto func_exit;
    }

    ret = linear_layer(conv_out.data, &linear_config, linear_out);
    if (ret != ALGO_NORMAL) {
        goto func_exit;
    }

    if (linear_out[1] > linear_out[0]) {
        *is_voice = true;
    }

func_exit:
    if (conv_out.data) {
        free(conv_out.data);
    }

    return ret;
}
