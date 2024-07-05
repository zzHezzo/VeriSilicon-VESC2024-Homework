#include "vad.h"
#include <fftw3.h>
#include <math.h>

void enframe(float *signal, int signal_length, float **frames, int frame_length, int hop_length, int *num_frames) {
    *num_frames = (signal_length - frame_length + hop_length) / hop_length;
    *frames = (float *)malloc((*num_frames) * frame_length * sizeof(float));
    for (int i = 0; i < *num_frames; i++) {
        for (int j = 0; j < frame_length; j++) {
            (*frames)[i * frame_length + j] = signal[i * hop_length + j];
        }
    }
}

void calculate_spectral_entropy(float *frames, int num_frames, int frame_length, float *entropy) {
    for (int i = 0; i < num_frames; i++) {
        double spectrum[frame_length];
        fftw_plan plan = fftw_plan_r2r_1d(frame_length, frames + i * frame_length, spectrum, FFTW_R2HC, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        double sum = 0;
        for (int j = 1; j < frame_length / 2; j++) {
            spectrum[j] = fabs(spectrum[j]);
            sum += spectrum[j];
        }

        double prob[frame_length / 2];
        for (int j = 1; j < frame_length / 2; j++) {
            prob[j] = spectrum[j] / sum;
        }

        entropy[i] = 0;
        for (int j = 1; j < frame_length / 2; j++) {
            if (prob[j] > 0) {
                entropy[i] -= prob[j] * log10(prob[j]);
            }
        }
    }
}

void vad_revr(float *entropy, int num_frames, float T1, float T2, uint32_t **vad_output, uint32_t *output_size) {
    int maxsilence = 8;
    int minlen = 5;
    int status = 0;
    int count = 0;
    int silence = 0;
    int *x1 = (int *)malloc(num_frames * sizeof(int));
    int *x2 = (int *)malloc(num_frames * sizeof(int));
    *output_size = 0;

    for (int n = 1; n < num_frames; n++) {
        if (status == 0 || status == 1) {
            if (entropy[n] < T2) {
                x1[*output_size] = n;
                status = 2;
                silence = 0;
                count = 1;
            } else if (entropy[n] < T1) {
                status = 1;
                count++;
            } else {
                status = 0;
                count = 0;
            }
        } else if (status == 2) {
            if (entropy[n] < T1) {
                count++;
            } else {
                silence++;
                if (silence < maxsilence) {
                    count++;
                } else if (count < minlen) {
                    status = 0;
                } else {
                    x2[*output_size] = x1[*output_size] + count;
                    (*output_size)++;
                    status = 3;
                }
            }
        } else if (status == 3) {
            status = 0;
            count = 0;
        }
    }

    for (int i = 0; i < *output_size; i++) {
        vad_output[i][0] = x1[i] * HOP_LENGTH;
        vad_output[i][1] = x2[i] * HOP_LENGTH;
    }

    free(x1);
    free(x2);
}

int vad_proc(float *data, uint16_t frame_num, uint32_t **vad_output, uint32_t *output_size) {
    if (!data || !vad_output || !output_size) {
        return ALGO_POINTER_NULL;
    }

    int signal_length = frame_num * FRAME_LENGTH;
    float *frames;
    int num_frames;
    enframe(data, signal_length, &frames, FRAME_LENGTH, HOP_LENGTH, &num_frames);

    float *entropy = (float *)malloc(num_frames * sizeof(float));
    calculate_spectral_entropy(frames, num_frames, FRAME_LENGTH, entropy);

    float mean_entropy = 0;
    for (int i = 0; i < num_frames; i++) {
        mean_entropy += entropy[i];
    }
    mean_entropy /= num_frames;

    float initial_silence_entropy = 0;
    for (int i = 0; i < NIS; i++) {
        initial_silence_entropy += entropy[i];
    }
    initial_silence_entropy /= NIS;

    float threshold_diff = initial_silence_entropy - mean_entropy;
    float T1 = THR1 * threshold_diff + mean_entropy;
    float T2 = THR2 * threshold_diff + mean_entropy;

    vad_revr(entropy, num_frames, T1, T2, vad_output, output_size);

    free(frames);
    free(entropy);

    return ALGO_NORMAL;
}
