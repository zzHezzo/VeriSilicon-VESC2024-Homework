#ifndef __CONV_H__
#define __CONV_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "algo_error_code.h"

/**
 * Input and output data of the convolutional layer
 */
typedef struct _Conv2dData {
    uint16_t row;
    uint16_t col;
    uint16_t channel;
    double *data;
} Conv2dData;

/**
 * convolutional layer weights
 */
typedef struct _Conv2dFilter {
    uint16_t row;
    uint16_t col;
    uint16_t channel;
    uint16_t filter_num;
    double *data;
} Conv2dFilter;

/**
 * Batch Normalization
 */
typedef struct _BatchNorm2d {
    uint16_t size;
    double *mean;
    double *var;
    double *gamma;
    double *beta;
} BatchNorm2d;

/**
 * configuration of convolutional layers, including convolution weights and BN
 */
typedef struct _Conv2dConfig {
    uint16_t stride;
    uint16_t pad;
    Conv2dFilter *filter;
    BatchNorm2d *bn;
} Conv2dConfig;

/**
 * configuration of the linear layer, including weights and biases
 */
typedef struct _LinearConfig {
    uint16_t inp_size;
    uint16_t fea_size;
    double *weight;
    double *bias;
} LinearParam;

/**
 * @brief conv2d with BN layer without bias
 *
 * @param[in] input_feat: input feature map
 * @param[in] param: configuration of the convolutional layer
 * @param[out] output_feat: output feature map
 * @return error code
 */
int conv2d_bn_no_bias(Conv2dData *input_feat, Conv2dConfig *param, Conv2dData *output_feat);

/**
 * @brief leak_relu activation function
 *
 * @param[in] neg_slope: controls the angle of the negative slope
 * @param[in] inp: input data
 * @param[in] inp_size: input data size
 * @param[out] out: output data
 * @return error code
 */
int leaky_relu(double neg_slope, double *inp, uint16_t inp_size, double *out);

/**
 * @brief linear layer
 *
 * @param[in] inp: input data
 * @param[in] linear_config: configuration of the linear layer
 * @param[out] out: output data
 * @return error code
 */
int linear_layer(double *inp, LinearParam *linear_config, double *out);

/**
 * @brief calculate the length of different dimensions of the convolutional layer output feature map
 *
 * @param[in] raw_len: the length of input feature map
 * @param[in] pad_len: the length of padding
 * @param[in] filter_len: kernel size
 * @param[in] stride: the stride length of conv
 * @return the length of output feature map
 */
uint16_t cal_conv_out_len(uint16_t raw_len, uint16_t pad_len, uint16_t filter_len, uint16_t stride);

#endif
