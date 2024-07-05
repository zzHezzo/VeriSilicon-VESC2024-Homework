#include <stdio.h>
#include "vad.h"
#include "test_data.h"

int main() {
    uint32_t output_size = 0;
    uint16_t frame_num = DATA_SIZE / FRAME_LENGTH;
    uint32_t **vad_output = (uint32_t **)malloc(frame_num * sizeof(uint32_t *));
    for (int i = 0; i < frame_num; i++) {
        vad_output[i] = (uint32_t *)malloc(2 * sizeof(uint32_t));
    }

    vad_proc(test_data, frame_num, vad_output, &output_size);

    printf("Vocal signal startpoint and endpoint:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%d %d\n", vad_output[i][0], vad_output[i][1]);
    }

    for (int i = 0; i < frame_num; i++) {
        free(vad_output[i]);
    }
    free(vad_output);

    return ALGO_NORMAL;
}
