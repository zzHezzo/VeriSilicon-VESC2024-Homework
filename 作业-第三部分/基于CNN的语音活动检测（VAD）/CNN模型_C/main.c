#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "vad.h"
#include "algo_error_code.h"

#define RAW_FS     (8000)
#define OBJ_FS     (8000)
#define FRAME_STEP (120) // 0.015 * 8000
#define FRAME_LEN  (240) // 0.03 * 8000

uint64_t get_rows(char *file_dir)
{
    char line[1024];
    uint64_t i = 0;
    FILE *stream = fopen(file_dir, "r");
    if (stream) {
        while (fgets(line, 1024, stream)) {
            i++;
        }
        fclose(stream);
    }
    return i;
}

void get_data(char *file_dir, double *data_buf)
{
    char line[1024];
    uint64_t i = 0;
    FILE *stream = fopen(file_dir, "r");
    if (stream) {
        while (fgets(line, 1024, stream)) {
            data_buf[i] = strtod(line, NULL);
            i++;
        }
        fclose(stream);
    }
}

void downsample(double *raw_data, uint64_t raw_size, uint16_t raw_fs, uint16_t obj_fs, double *out, uint64_t *out_size)
{
    uint16_t interval = raw_fs / obj_fs;
    uint64_t i        = 0;
    *out_size = 0;
    for (i = 0; i < raw_size; i += interval) {
        out[(*out_size)++] = raw_data[i];
    }
}

/**
 * voice_segment: 2n: start index, 2n+1:end index
 */
void cal_voice_segment(int8_t *pred_class, const uint64_t *pred_idx_in_data, uint64_t pred_class_size, uint64_t raw_data_size, uint64_t *voice_segment, uint64_t *voice_segment_size)
{
    uint64_t i = 0, voice_segment_cnt = 0;
    int8_t diff_vaule = 0;
    bool is_start     = true;
    *voice_segment_size = 0;
    for (i = 1; i < pred_class_size; i++) {
        diff_vaule = pred_class[i] - pred_class[i - 1];
        if (diff_vaule == 1) {
            voice_segment[voice_segment_cnt++] = pred_idx_in_data[i];
            is_start                           = false;
        }
        if (diff_vaule == -1) {
            if (is_start) {
                voice_segment[voice_segment_cnt++] = 0;
            }
            voice_segment[voice_segment_cnt++] = pred_idx_in_data[i];
            is_start                           = true;
        }
    }
    if (!is_start) {
        voice_segment[voice_segment_cnt++] = raw_data_size - 1;
    }
    *voice_segment_size = voice_segment_cnt;
}

int main()
{
    char file_dir[] = "./data.txt";
    FILE *file      = fopen("./pred.txt", "w");
    int ret            = ALGO_NORMAL;
    uint64_t data_size = 0, down_size = 0, pred_cnt = 0, i = 0, voice_seg_size = 0;
    double *total_data       = NULL;
    bool vad_out             = false;
    int8_t *total_pred       = NULL;
    uint64_t *total_pred_idx = NULL, *all_voice_segment = NULL;
    Conv2dData vad_inp = {.channel = 1, .row = 1, .col = FRAME_LEN, .data = NULL};
    data_size = get_rows(file_dir);
    printf("data_size = %llu\n", data_size);
    total_data = (double *)malloc(sizeof(double) * data_size);
    if (!total_data) {
        printf("malloc fail\n");
        return 0;
    }
    // get data and downsample
    get_data(file_dir, total_data);
    downsample(total_data, data_size, RAW_FS, OBJ_FS, total_data, &down_size);
    printf("down_size = %llu\n", down_size);
    total_pred = (int8_t *)malloc(sizeof(int8_t) * ((down_size - FRAME_LEN) / FRAME_STEP + 1));
    if (!total_pred) {
        printf("malloc fail\n");
        goto exit;
    }
    total_pred_idx =
        (uint64_t *)malloc(sizeof(uint64_t) * ((down_size - FRAME_LEN) / FRAME_STEP + 1));
    if (!total_pred_idx) {
        printf("malloc fail\n");
        goto exit;
    }
    all_voice_segment =
        (uint64_t *)malloc(sizeof(uint64_t) * ((down_size - FRAME_LEN) / FRAME_STEP + 1));
    if (!all_voice_segment) {
        printf("malloc fail\n");
        goto exit;
    }
    // streaming audio data, frame by frame
    for (i = 0; i < down_size; i += FRAME_STEP) {
        if (i + FRAME_LEN - 1 > down_size) {
            break;
        }
        vad_inp.data = total_data + i;
        ret = vad(&vad_inp, &vad_out);
        if (ret != ALGO_NORMAL) {
            printf("ret = %d\n", ret);
            goto exit;
        }
        total_pred[pred_cnt]       = (int8_t)vad_out;
        total_pred_idx[pred_cnt++] = i;
    }
    // calaulate voice segments
    cal_voice_segment(total_pred, total_pred_idx, pred_cnt, down_size, all_voice_segment, &voice_seg_size);
    // save the results to a file
    if (file) {
        for (i = 0; i < voice_seg_size; i += 2) {
            printf("%llu, %llu\n", all_voice_segment[i],all_voice_segment[i+1]);
            fprintf(file, "%llu, %llu\n", all_voice_segment[i], all_voice_segment[i + 1]);
        }
    }
    fclose(file);
exit:
    free(total_data);
    free(total_pred);
    free(total_pred_idx);
    free(all_voice_segment);
    return 0;
}
