#ifndef __VAD_H__
#define __VAD_H__

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "algo_error_code.h"

#define FRAME_LENGTH (240)  // The length of a frame of audio data
#define HOP_LENGTH (80)     // The number of samples between successive frames
#define SAMPLE_RATE (8000)
#define THR1 (0.99)
#define THR2 (0.96)
#define NIS (20)            // Number of initial silence frames

/**
 * @brief voice activity detection function
 *
 * @param[in] data: input audio data
 * @param[in] frame_num: number of frames
 * @param[out] vad_output: the result of vad, a two-dimensional array of starting and ending points,
 * e.g.: [[10,20],[30,40],[60,100]]
 * @param[out] output_size: the size of vad_output
 * @return error code
 */
int vad_proc(float *data, uint16_t frame_num, uint32_t **vad_output, uint32_t *output_size);

#endif /* __VAD_H__ */
