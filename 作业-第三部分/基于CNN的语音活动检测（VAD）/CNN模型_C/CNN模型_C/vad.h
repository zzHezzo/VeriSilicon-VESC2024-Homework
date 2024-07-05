#ifndef __VAD_H__
#define __VAD_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "conv.h"
#include "algo_error_code.h"

/**
 * @brief voice detection function
 *
 * @param[in] inp_data: raw audio data
 * @param[out] is_voice: the result of voice detection
 * @return error code
 */
int vad(Conv2dData *inp_data, bool *is_voice);

#endif
