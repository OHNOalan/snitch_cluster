#pragma once

#include <stdbool.h>
#include "batchnorm_data_structures.h"
#include "snrt.h"

#define PERF_DEBUG 1
#define PERF_WHOLE_BLOCK 1

#if PERF_WHOLE_BLOCK
#define SNRT_SECTIONED_MCYCLE() 0xdeadbeef
#else
#define SNRT_SECTIONED_MCYCLE() (snrt_mcycle())
#endif
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)
#if PERF_DEBUG
NAMED_DUMP(uint32_t, PERF_RESULT, 0x7C4)
#endif

static inline uint32_t __attribute__((__const__))
get_core_num_work_items(uint32_t num_work_items, uint32_t num_compute_cores,
                        uint32_t compute_id) {
    return num_work_items / num_compute_cores +
           (compute_id < (num_work_items % num_compute_cores));
}

static inline uint32_t __attribute__((__const__))
get_offset_for_core_work_blocked(uint32_t num_work_items,
                                 uint32_t num_compute_cores,
                                 uint32_t compute_id) {
    return num_work_items / num_compute_cores * compute_id +
           min(compute_id, (num_work_items % num_compute_cores));
}

// use ALIGN_UP for powers of 2 - it'll be much faster
static inline uint32_t __attribute__((const))
align_up_non_power_of_2(uint32_t n, uint32_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}
static inline int positive_modulo(int i, int n) { return (i % n + n) % n; }

static inline void reset_and_start_perf_single_core(
    uint32_t compute_id, enum snrt_perf_cnt counter_idx,
    enum snrt_perf_cnt_type counter_type) {
#if PERF_DEBUG
    if (compute_id == 0) {
        snrt_reset_perf_counter(counter_idx);

        // Start performance counters
        snrt_start_perf_counter(counter_idx, counter_type, 0);
        DUMP(111000 + counter_idx);
    }
    snrt_cluster_hw_barrier();
#endif
}

static inline void end_perf_and_dump_single_core(
    uint32_t compute_id, enum snrt_perf_cnt counter_idx) {
#if PERF_DEBUG
    uint32_t res;
    if (compute_id == 0) {
        snrt_stop_perf_counter(counter_idx);

        res = snrt_get_perf_counter(counter_idx);
        DUMP(222000 + counter_idx);
        dump_PERF_RESULT(res);
    }
    snrt_cluster_hw_barrier();
#endif
}

static inline snrt_dma_txid_t initiate_dma_1d_or_2d(
    void* dst, void* src, size_t size, size_t dst_stride, size_t src_stride,
    size_t repeat, uint32_t is_1d) {
    if (is_1d) {
        return snrt_dma_start_1d(dst, src, size * repeat);
    } else {
        return snrt_dma_start_2d(dst, src, size, dst_stride, src_stride,
                                 repeat);
    }
}

static inline void __attribute__((always_inline))
batchnorm_forward_fp64_no_loop(
    const double* ifmap_scratch, double* ofmap_scratch,
    const double* gamma_scratch, const double* beta_scratch,
    uint32_t num_bytes_per_point,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_sub_1,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels,
    //   then over the different points
    // Split work over channels to maximize efficacy of frep and avoid tcdm
    // contention. outside loop: channels inside loop: points
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core,  // dimension of inner loop
        num_channels_to_process,   // dimension of outer loop
        num_bytes_per_point,       // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration

    register volatile uint32_t i =
        0;  // updated during frep for pseudo-dual issue
    snrt_cluster_hw_barrier();
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, ofmap_scratch);
    snrt_ssr_enable();
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        register double gamma = *gamma_scratch;
        register double beta = *beta_scratch;
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fmadd.d ft1, ft0, %[gamma], %[beta]"
            :
            : [gamma] "fr"(gamma), [beta] "fr"(beta),
              [n_frep] "r"(work_sub_1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2");

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only
                                                          // 1 channel
            "add %[gamma_scratch], %[gamma_scratch], %[channel_stride_in_bytes]\n"
            "add %[beta_scratch], %[beta_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [gamma_scratch] "+r"(gamma_scratch),
              [beta_scratch] "+r"(beta_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "+r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

    } while (i < num_channels_to_process);
    snrt_fpu_fence();
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_forward_tile_fp64_looped(
    const double* ifmap_scratch, double* ofmap_scratch,
    const double* gamma_scratch, const double* beta_scratch,
    uint32_t num_bytes_per_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_sub_1, uint32_t tile_stride_in_doubles,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels,
    //   then over the different points
    // Split work over channels to maximize efficacy of frep and avoid tcdm
    // contention. outside loop: channels inside loop: points

    uint32_t buf_flag = 0;
    uint32_t prev_work = work_in_tile;
    const uint32_t inner_loop_stride = num_bytes_per_point;
    const uint32_t buf_flag_offset = tile_stride_in_doubles * sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration

    do {  // while (work_in_tile != 0)
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue

        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, ofmap_scratch);
        snrt_ssr_enable();
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            register double gamma = *gamma_scratch;
            register double beta = *beta_scratch;
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmadd.d ft1, ft0, %[gamma], %[beta]\n"
                :
                : [gamma] "fr"(gamma), [beta] "fr"(beta),
                  [n_frep] "r"(work_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "addi %[i], %[i], 1\n"
                "add %[gamma_scratch], %[gamma_scratch], %[channel_stride_in_bytes]\n"
                "add %[beta_scratch], %[beta_scratch],%[channel_stride_in_bytes]\n"
                : [gamma_scratch] "+r"(gamma_scratch),
                  [beta_scratch] "+r"(beta_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "+r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[gamma_scratch], %[gamma_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[beta_scratch], %[beta_scratch],%[input_channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[work_div_1_sub_1], %[work_div_1_sub_1_offset](%[dm_comm])\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ofmap_scratch], %[ofmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ofmap_scratch], %[ofmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                :
                [buf_flag] "+r"(buf_flag), [gamma_scratch] "+r"(gamma_scratch),
                [beta_scratch] "+r"(beta_scratch),
                [work_in_tile] "+r"(work_in_tile), [prev_work] "+r"(prev_work),
                [work_div_1_sub_1] "=r"(work_sub_1),
                [ifmap_scratch] "+r"(ifmap_scratch),
                [ofmap_scratch] "+r"(ofmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_div_1_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_1_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL), [DM_0] "i"(SNRT_SSR_DM0),
                  [DM_1] "i"(SNRT_SSR_DM1), [DM_2] "i"(SNRT_SSR_DM2),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    } while (work_in_tile != 0);
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void __attribute__((always_inline))
batchnorm_collect_statistics_fp64_no_loop(
    const double* ifmap_scratch, double* current_mean_scratch,
    double* current_var_scratch,
    uint32_t num_points,  // of all batches. represents N in computation of mean
                          // and var
    uint32_t num_bytes_per_point,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_div_4_sub_1, uint32_t work_mod_4,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels,
    //   then over the different points
    // Split work over channels to maximize efficacy of frep and avoid tcdm
    // contention. outside loop: channels inside loop: points
    snrt_ssr_loop_3d(
        SNRT_SSR_DM0,
        num_points_work_for_core,  // dimension of inner loop
        2,                         // repeat values once
        num_channels_to_process,   // dimension of outer loop
        num_bytes_per_point,       // stride per inner loop iteration: 1 point
        0,                         // repeat values once per channel
        channel_stride * sizeof(double));  // stride per outer loop iteration

    register volatile uint32_t i =
        0;  // updated during frep for pseudo-dual issue
    register uint32_t frep = num_points_work_for_core >= 4;

    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    register double num_points_double = num_points;
    snrt_cluster_hw_barrier();
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, ifmap_scratch);
    snrt_ssr_enable();
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        register double mean;
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 4, 0, 0 \n"
                "fadd.d %[sum0], ft0, %[sum0]\n"
                "fadd.d %[sum1], ft0, %[sum1]\n"
                "fadd.d %[sum2], ft0, %[sum2]\n"
                "fadd.d %[sum3], ft0, %[sum3]\n"
                : [sum0] "+fr"(sum0), [sum1] "+fr"(sum1), [sum2] "+fr"(sum2),
                  [sum3] "+fr"(sum3)
                : [n_frep] "r"(work_div_4_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[current_mean_scratch], %[current_mean_scratch],%[channel_stride_in_bytes]\n"
            "add %[current_var_scratch], %[current_var_scratch],%[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            : [current_mean_scratch] "+r"(current_mean_scratch),
              [current_var_scratch] "+r"(current_var_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=&r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_4], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                    // then mod is 2
            "beqz %[mod_temp], 2f\n"                // jump to 2 if no
            "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                    // then mod is 1
            "beqz %[mod_temp], 1f\n"                // jump to 1 if no
            "3:\n"
            "fadd.d %[sum0], ft0, %[sum0]\n"
            "fadd.d %[sum1], ft0, %[sum1]\n"
            "fadd.d %[sum2], ft0, %[sum2]\n"
            "j 0f\n"
            "2:\n"
            "fadd.d %[sum0], ft0, %[sum0]\n"
            "fadd.d %[sum1], ft0, %[sum1]\n"
            "j 0f\n"
            "1:\n"
            "fadd.d %[sum0], ft0, %[sum0]\n"
            "0:\n"
            "fadd.d %[sum2], %[sum2], %[sum3]\n"
            "fadd.d %[sum0], %[sum0], %[sum1]\n"
            "fsgnj.d %[sum3], %[ZERO], %[ZERO]\n"
            "fsgnj.d %[sum1], %[ZERO], %[ZERO]\n"
            "fadd.d %[sum0], %[sum0], %[sum2]\n"
            "fsgnj.d %[sum2], %[ZERO], %[ZERO]\n"
            "fdiv.d %[mean], %[sum0], %[num_points_double]\n"
            "fsgnj.d %[sum0], %[ZERO], %[ZERO]\n"
            : [mod_temp] "=r"(mod_temp), [sum0] "+fr"(sum0), [sum1] "+fr"(sum1),
              [sum2] "+fr"(sum2), [sum3] "+fr"(sum3), [mean] "=fr"(mean)
            : [work_mod_4] "r"(work_mod_4), [ZERO] "fr"(ZERO),
              [num_points_double] "fr"(num_points_double)
            : "ft0", "ft1", "ft2");
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                "fsub.d ft3, ft0, %[mean]\n"
                "fsub.d ft4, ft0, %[mean]\n"
                "fsub.d ft5, ft0, %[mean]\n"
                "fsub.d ft6, ft0, %[mean]\n"
                "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
                "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
                "fmadd.d %[sum2], ft5, ft5, %[sum2]\n"
                "fmadd.d %[sum3], ft6, ft6, %[sum3]\n"
                : [sum0] "+fr"(sum0), [sum1] "+fr"(sum1), [sum2] "+fr"(sum2),
                  [sum3] "+fr"(sum3)
                : [n_frep] "r"(work_div_4_sub_1),  // we repeat n_frep+1 times
                  [mean] "fr"(mean)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }
        register double pop_variance;
        asm volatile(
            "beqz %[work_mod_4], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                    // then mod is 2
            "beqz %[mod_temp], 2f\n"                // jump to 2 if no
            "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                    // then mod is 1
            "beqz %[mod_temp], 1f\n"                // jump to 1 if no
            "3:\n"
            "fsub.d ft3, ft0, %[mean]\n"
            "fsub.d ft4, ft0, %[mean]\n"
            "fsub.d ft5, ft0, %[mean]\n"
            "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
            "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
            "fmadd.d %[sum2], ft5, ft5, %[sum2]\n"
            "j 0f\n"
            "2:\n"
            "fsub.d ft3, ft0, %[mean]\n"
            "fsub.d ft4, ft0, %[mean]\n"
            "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
            "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
            "j 0f\n"
            "1:\n"
            "fsub.d ft3, ft0, %[mean]\n"
            "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
            "0:\n"
            "fadd.d %[sum2], %[sum2], %[sum3]\n"
            "fadd.d %[sum0], %[sum0], %[sum1]\n"
            "fsgnj.d %[sum3], %[ZERO], %[ZERO]\n"
            "fsgnj.d %[sum1], %[ZERO], %[ZERO]\n"
            "fadd.d %[sum0], %[sum0], %[sum2]\n"
            "fsgnj.d %[sum2], %[ZERO], %[ZERO]\n"
            "fdiv.d %[pop_variance], %[sum0], %[num_points_double]\n"
            "fsgnj.d %[sum0], %[ZERO], %[ZERO]\n"
            "fsd %[mean], 0(%[current_mean_scratch])\n"
            "fsd %[pop_variance], 0(%[current_var_scratch])\n"
            : [mod_temp] "=&r"(mod_temp), [sum0] "+fr"(sum0),
              [sum1] "+fr"(sum1), [sum2] "+fr"(sum2), [sum3] "+fr"(sum3),
              [pop_variance] "=&fr"(pop_variance)
            : [work_mod_4] "r"(work_mod_4), [ZERO] "fr"(ZERO),
              [num_points_double] "fr"(num_points_double), [mean] "fr"(mean),
              [current_mean_scratch] "r"(current_mean_scratch),
              [current_var_scratch] "r"(current_var_scratch)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
    } while (i < num_channels_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_collect_mean_statistics_tile_fp64_looped(
    const double* ifmap_scratch, double* current_mean_scratch,
    uint32_t num_points,  // of all batches. represents N in computation of mean
                          // and var
    uint32_t num_bytes_per_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_div_4_sub_1, uint32_t work_mod_4,
    uint32_t tile_stride_in_doubles,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels,
    //   then over the different points
    // Split work over channels to maximize efficacy of frep and avoid tcdm
    // contention. outside loop: channels inside loop: points

    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_4;
    register uint32_t frep = work_in_tile >= 4;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    uint32_t buf_flag = 0;

    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_stride_in_doubles * sizeof(double);
    const uint32_t inner_loop_stride = num_bytes_per_point;
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);

    snrt_ssr_loop_2d(
        SNRT_SSR_DM0,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration

    register double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    register double num_points_double = num_points;
    snrt_ssr_enable();
    do {  // while (work_in_tile != 0)
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "fadd.d %[sum3], ft0, %[sum3]\n"
                    "fadd.d %[sum2], ft0, %[sum2]\n"
                    "fadd.d %[sum1], ft0, %[sum1]\n"
                    "fadd.d %[sum0], ft0, %[sum0]\n"
                    :
                    [sum0] "+fr"(sum0), [sum1] "+fr"(sum1), [sum2] "+fr"(sum2),
                    [sum3] "+fr"(sum3)
                    : [n_frep] "r"(
                        work_div_4_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[current_mean_scratch], %[current_mean_scratch],%[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                :
                [current_mean_scratch] "+r"(current_mean_scratch), [i] "+r"(i),
                [channel_stride_in_bytes] "=&r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_4], %[work_mod_4_offset](%[dm_comm])\n"
                "lw %[work_div_4_sub_1], %[work_div_4_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 4\n"  // cmp frep < 2, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_0] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_0] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag), [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_4] "=r"(next_work_mod_4),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_4_sub_1] "+r"(work_div_4_sub_1),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_4_offset] "i"(offsetof(dm_comm_t, work_mod_4)),
                  [work_div_4_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_4_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2), [DM_0] "i"(SNRT_SSR_DM0),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            register uint32_t mod_temp;
            register double running_sum;
            asm volatile(
                "fld %[running_sum], 0(%[current_mean_scratch])\n"
                "beqz %[work_mod_4], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                        // then mod is 2
                "beqz %[mod_temp], 2f\n"                // jump to 2 if no
                "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                        // then mod is 1
                "beqz %[mod_temp], 1f\n"                // jump to 1 if no
                "3:\n"
                "fadd.d %[sum2], ft0, %[sum2]\n"
                "fadd.d %[sum1], ft0, %[sum1]\n"
                "fadd.d %[sum0], ft0, %[sum0]\n"
                "j 0f\n"
                "2:\n"
                "fadd.d %[sum1], ft0, %[sum1]\n"
                "fadd.d %[sum0], ft0, %[sum0]\n"
                "j 0f\n"
                "1:\n"
                "fadd.d %[sum0], ft0, %[sum0]\n"
                "0:\n"
                "fadd.d %[sum2], %[sum2], %[sum3]\n"
                "fsgnj.d %[sum3], %[ZERO], %[ZERO]\n"
                "fadd.d %[sum0], %[sum0], %[sum1]\n"
                "fsgnj.d %[sum1], %[ZERO], %[ZERO]\n"
                "fadd.d %[sum0], %[sum0], %[sum2]\n"
                "fsgnj.d %[sum2], %[ZERO], %[ZERO]\n"
                "fadd.d %[running_sum], %[running_sum], %[sum0]\n"
                "fsgnj.d %[sum0], %[ZERO], %[ZERO]\n"
                "fsd %[running_sum], 0(%[current_mean_scratch])\n"
                : [mod_temp] "=r"(mod_temp), [sum0] "+fr"(sum0),
                  [sum1] "+fr"(sum1), [sum2] "+fr"(sum2), [sum3] "+fr"(sum3),
                  [running_sum] "=&fr"(running_sum)
                : [work_mod_4] "r"(work_mod_4), [ZERO] "fr"(ZERO),
                  [current_mean_scratch] "r"(current_mean_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_channels_to_process);
        work_mod_4 = next_work_mod_4;
        current_mean_scratch -= channel_stride * (num_channels_to_process - 1);
        // don't need to fpu_fence since last 3 instructions are inconsequential
    } while (work_in_tile != 0);
    snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_channels_to_process,
                     outer_loop_stride);
    snrt_fpu_fence();
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, current_mean_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, current_mean_scratch);

    asm volatile(
        "frep.o %[n_frep], 1, 0, 0\n"
        "fdiv.d ft1, ft0, %[num_points_double]\n"
        :
        : [n_frep] "r"(num_channels_to_process - 1), [num_points_double] "fr"(
                                                         num_points_double)
        : "ft0", "ft1", "ft2");
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void __attribute__((always_inline))
batchnorm_collect_var_statistics_tile_fp64_looped(
    const double* ifmap_scratch, const double* current_mean_scratch,
    double* current_var_scratch,
    uint32_t num_points,  // of all batches. represents N in computation of mean
                          // and var
    uint32_t num_bytes_per_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_div_4_sub_1, uint32_t work_mod_4,
    uint32_t tile_stride_in_doubles,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels,
    //   then over the different points
    // Split work over channels to maximize efficacy of frep and avoid tcdm
    // contention. outside loop: channels inside loop: points

    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_4;
    register uint32_t frep = work_in_tile >= 4;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    uint32_t buf_flag = 0;

    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_stride_in_doubles * sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = num_bytes_per_point;
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);

    snrt_ssr_loop_2d(
        SNRT_SSR_DM0,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration

    register double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    register double num_points_double = num_points;
    snrt_ssr_enable();
    do {  // while (work_in_tile != 0)
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            register double mean = *current_mean_scratch;
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fsub.d ft6, ft0, %[mean]\n"
                    "fsub.d ft5, ft0, %[mean]\n"
                    "fsub.d ft4, ft0, %[mean]\n"
                    "fsub.d ft3, ft0, %[mean]\n"
                    "fmadd.d %[sum3], ft6, ft6, %[sum3]\n"
                    "fmadd.d %[sum2], ft5, ft5, %[sum2]\n"
                    "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
                    "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
                    :
                    [sum0] "+fr"(sum0), [sum1] "+fr"(sum1), [sum2] "+fr"(sum2),
                    [sum3] "+fr"(sum3)
                    : [n_frep] "r"(
                          work_div_4_sub_1),  // we repeat n_frep+1 times
                      [mean] "fr"(mean)
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[current_var_scratch], %[current_var_scratch],%[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[current_mean_scratch], %[current_mean_scratch],%[channel_stride_in_bytes]\n"
                :
                [current_mean_scratch] "+r"(current_mean_scratch), [i] "+r"(i),
                [channel_stride_in_bytes] "=&r"(channel_stride_in_bytes),
                [current_var_scratch] "+r"(current_var_scratch)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[current_mean_scratch], %[current_mean_scratch], %[input_channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_4], %[work_mod_4_offset](%[dm_comm])\n"
                "lw %[work_div_4_sub_1], %[work_div_4_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 4\n"  // cmp frep < 2, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_0] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_0] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag), [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_4] "=r"(next_work_mod_4),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_4_sub_1] "+r"(work_div_4_sub_1),
                  [ifmap_scratch] "+r"(ifmap_scratch),
                  [current_mean_scratch] "+r"(current_mean_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_4_offset] "i"(offsetof(dm_comm_t, work_mod_4)),
                  [work_div_4_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_4_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2), [DM_0] "i"(SNRT_SSR_DM0),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm),
                  [buf_flag_offset] "r"(buf_flag_offset),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist)
                : "ft0", "ft1", "ft2", "x0", "memory");

            register uint32_t mod_temp;
            register double running_sum;
            asm volatile(
                "fld %[running_sum], 0(%[current_var_scratch])\n"
                "beqz %[work_mod_4], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                        // then mod is 2
                "beqz %[mod_temp], 2f\n"                // jump to 2 if no
                "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                        // then mod is 1
                "beqz %[mod_temp], 1f\n"                // jump to 1 if no
                "3:\n"
                "fsub.d ft5, ft0, %[mean]\n"
                "fsub.d ft4, ft0, %[mean]\n"
                "fsub.d ft3, ft0, %[mean]\n"
                "fmadd.d %[sum2], ft5, ft5, %[sum2]\n"
                "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
                "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
                "j 0f\n"
                "2:\n"
                "fsub.d ft4, ft0, %[mean]\n"
                "fsub.d ft3, ft0, %[mean]\n"
                "fmadd.d %[sum1], ft4, ft4, %[sum1]\n"
                "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
                "j 0f\n"
                "1:\n"
                "fsub.d ft3, ft0, %[mean]\n"
                "fmadd.d %[sum0], ft3, ft3, %[sum0]\n"
                "0:\n"
                "fadd.d %[sum2], %[sum2], %[sum3]\n"
                "fsgnj.d %[sum3], %[ZERO], %[ZERO]\n"
                "fadd.d %[sum0], %[sum0], %[sum1]\n"
                "fsgnj.d %[sum1], %[ZERO], %[ZERO]\n"
                "fadd.d %[sum0], %[sum0], %[sum2]\n"
                "fsgnj.d %[sum2], %[ZERO], %[ZERO]\n"
                "fadd.d %[running_sum], %[running_sum], %[sum0]\n"
                "fsgnj.d %[sum0], %[ZERO], %[ZERO]\n"
                "fsd %[running_sum], 0(%[current_var_scratch])\n"
                : [mod_temp] "=&r"(mod_temp), [sum0] "+&fr"(sum0),
                  [sum1] "+&fr"(sum1), [sum2] "+&fr"(sum2), [sum3] "+&fr"(sum3),
                  [running_sum] "=&fr"(running_sum)
                : [work_mod_4] "r"(work_mod_4), [ZERO] "fr"(ZERO),
                  [num_points_double] "fr"(num_points_double),
                  [mean] "fr"(mean),
                  [current_var_scratch] "r"(current_var_scratch)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
        } while (i < num_channels_to_process);
        work_mod_4 = next_work_mod_4;
        current_var_scratch -= channel_stride * (num_channels_to_process - 1);
        // don't need to fpu_fence since last 3 instructions are inconsequential
    } while (work_in_tile != 0);
    snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_channels_to_process,
                     outer_loop_stride);
    snrt_fpu_fence();
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, current_var_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, current_var_scratch);

    asm volatile(
        "frep.o %[n_frep], 1, 0, 0\n"
        "fdiv.d ft1, ft0, %[num_points_double]\n"
        :
        : [n_frep] "r"(num_channels_to_process - 1), [num_points_double] "fr"(
                                                         num_points_double)
        : "ft0", "ft1", "ft2");
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void batchnorm_forward_dma_main_loop_fp_agnostic(
    double* ifmap, double* ofmap, uint32_t num_doubles_per_aligned_point,
    uint32_t num_bytes_per_packed_point, uint32_t num_bytes_per_aligned_point,
    uint32_t is_point_aligned_to_8_byte_boundary,
    uint32_t work_left,  // only present for dma
    uint32_t initial_work_in_tile, dm_comm_t* dm_comm, uint32_t unroll,
    uint32_t tile_size_in_points, uint32_t tile_stride_in_doubles,
    double* ifmap_scratch, double* ofmap_scratch, uint32_t buf_flag) {
    snrt_dma_wait_all();

    // signal first iteration
    // compute cores don't have to read dm comm the first time
    snrt_cluster_hw_barrier();

    // skip the first iteration in looping
    uint32_t point_start = initial_work_in_tile;
    uint32_t work_in_tile = initial_work_in_tile;
    uint32_t prev_point_start = 0;
    uint32_t num_points_work_in_prev_tile = initial_work_in_tile;
    // split the remaining work "nicely"
    uint32_t min_loops = ceildiv(work_left, tile_size_in_points);
    // assume unroll is either 1 or 2 right now
    uint32_t ideal_work_in_tile = min(
        ALIGN_UP(ceildiv(work_left, min_loops), unroll), tile_size_in_points);

    while (work_left > 0) {
        // uint32_t estimated_max_tileable_work = tile_size_in_points;
        // (work_in_tile * ceildiv(C, num_compute_cores) * 5 *
        //  NUM_DOUBLES_LOADED_PER_CYCLE) /
        // (3 * C);
        work_in_tile = min(ideal_work_in_tile, work_left);
        work_left -= work_in_tile;

        // update comms
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_unroll = work_in_tile % unroll;
        dm_comm->work_div_unroll_sub_1 = work_in_tile / unroll - 1;
        // comm what the next iteration will be
        // wait for potential previous grad_ifmap write out?
        snrt_dma_wait_all();
        initiate_dma_1d_or_2d(
            &ifmap_scratch[tile_stride_in_doubles * buf_flag],
            &((char*)ifmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);

        // signal to core that current tile has information ready
        snrt_cluster_hw_barrier();
        DUMP(55);
        snrt_dma_wait_all();
        // wait for previous tile to be finished computing, signify current
        // tile inputs done loading
        snrt_cluster_hw_barrier();
        DUMP(56);

        // DUMP(prev_point_start);

        // DUMP(&ofmap_scratch[tile_stride_in_doubles * (!buf_flag)]);
        initiate_dma_1d_or_2d(
            &((char*)ofmap)[prev_point_start * num_bytes_per_packed_point],
            &ofmap_scratch[tile_stride_in_doubles * (!buf_flag)],
            num_bytes_per_packed_point, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_points_work_in_prev_tile,
            is_point_aligned_to_8_byte_boundary);

        prev_point_start = point_start;
        num_points_work_in_prev_tile = work_in_tile;
        point_start += work_in_tile;
        buf_flag = !buf_flag;
    }
    dm_comm->num_points_work_in_tile = 0;
    dm_comm->work_mod_unroll = 0;
    dm_comm->work_div_unroll_sub_1 = 0xdeadbeef;
    // signal last iteration that there is no more work
    snrt_cluster_hw_barrier();
    // wait for last tile to finish
    snrt_cluster_hw_barrier();

    initiate_dma_1d_or_2d(
        &((char*)ofmap)[prev_point_start * num_bytes_per_packed_point],
        &ofmap_scratch[tile_stride_in_doubles * (!buf_flag)],
        num_bytes_per_packed_point, num_bytes_per_packed_point,
        num_bytes_per_aligned_point, num_points_work_in_prev_tile,
        is_point_aligned_to_8_byte_boundary);
    snrt_dma_wait_all();
}

static inline void batchnorm_collect_statistics_dma_main_loop_fp_agnostic(
    batchnorm_training_layer_t* l, uint32_t num_doubles_per_aligned_point,
    uint32_t num_bytes_per_packed_point, uint32_t num_bytes_per_aligned_point,
    uint32_t is_point_aligned_to_8_byte_boundary,
    uint32_t work_left,  // only present for dma
    uint32_t initial_work_in_tile, dm_comm_t* dm_comm, uint32_t unroll,
    uint32_t tile_size_in_points, uint32_t tile_stride_in_doubles,
    double* ifmap_scratch, uint32_t buf_flag) {
    snrt_dma_wait_all();

    // signal first iteration
    // compute cores don't have to read dm comm the first time
    snrt_cluster_hw_barrier();

    // skip the first iteration in looping
    uint32_t point_start = initial_work_in_tile;
    uint32_t work_in_tile = initial_work_in_tile;
    uint32_t prev_point_start = 0;
    uint32_t num_points_work_in_prev_tile = initial_work_in_tile;
    // split the remaining work "nicely"
    uint32_t min_loops = ceildiv(work_left, tile_size_in_points);
    // min_loops = 7;
    // assume unroll is either 1 or 2 right now
    uint32_t ideal_work_in_tile = min(
        ALIGN_UP(ceildiv(work_left, min_loops), unroll), tile_size_in_points);

    while (work_left > 0) {
        // uint32_t estimated_max_tileable_work = tile_size_in_points;
        // (work_in_tile * ceildiv(C, num_compute_cores) * 5 *
        //  NUM_DOUBLES_LOADED_PER_CYCLE) /
        // (3 * C);
        work_in_tile = min(ideal_work_in_tile, work_left);
        work_left -= work_in_tile;

        // update comms
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_unroll = work_in_tile % unroll;
        dm_comm->work_div_unroll_sub_1 = work_in_tile / unroll - 1;
        // comm what the next iteration will be

        initiate_dma_1d_or_2d(
            &ifmap_scratch[tile_stride_in_doubles * buf_flag],
            &((char*)l->ifmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);

        // signal to core that current tile has information ready
        snrt_cluster_hw_barrier();
        DUMP(55);
        snrt_dma_wait_all();
        // wait for previous tile to be finished computing, signify current
        // tile inputs done loading
        snrt_cluster_hw_barrier();
        DUMP(56);
        prev_point_start = point_start;
        num_points_work_in_prev_tile = work_in_tile;
        point_start += work_in_tile;
        buf_flag = !buf_flag;
    }
    dm_comm->num_points_work_in_tile = 0;
    dm_comm->work_mod_unroll = 0;
    dm_comm->work_div_unroll_sub_1 = 0xdeadbeef;
    // signal last iteration that there is no more work
    snrt_cluster_hw_barrier();
    // wait for last tile to finish
    snrt_cluster_hw_barrier();
    snrt_dma_wait_all();
}

static inline void __attribute__((always_inline))
batchnorm_backward_fp64_no_loop(
    const double* grad_ofmap_scratch, double* grad_ifmap_scratch,
    const double* ifmap_scratch, const double* running_mean_scratch,
    const double* weight_scratch, const double* invstd_scratch,
    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t C,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_mod_2,  // precompute to avoid icache branch misses
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core,  // dimension of inner loop
        num_channels_to_process,   // dimension of outer loop
        C * sizeof(double),        // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    // thought: how could I minimize the # of reads to grad_ofmap?
    // dy is used for: grad_bias (addition)
    //                 grad_weight (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //                             (can it become a fused somehow? not really..
    //                             can precompute invstd * running_mean though)
    //                             then you get an fmsub(x[i,C], invstd[C],
    //                             invstd[C]*running_mean[C])
    //                 grad_ifmap (dy * invstd[C] * weight[C])
    // from this I think that the best result is to tile dy and x.
    // need to also tile the write out to grad_ifmap. This fills up all 3 ssrs.
    uint32_t frep = num_points_work_for_core >= 2;
    uint32_t work_div_2_sub_1 = num_points_work_for_core / 2 -
                                1;  // can underflow, but then frep won't happen
    register volatile uint32_t i =
        0;  // updated during frep for pseudo-dual issue
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double grad_weight_0 = ZERO;
    register double grad_weight_1 = ZERO;
    register double grad_bias_0 = ZERO;
    register double grad_bias_1 = ZERO;
    register double invstd = *invstd_scratch;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    register double weight_times_invstd = *weight_scratch;
    register double running_mean_times_invstd = *running_mean_scratch;
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fmsub.d ft6, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                "fadd.d %[grad_bias_1], ft0, %[grad_bias_1]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_1], ft6, ft0, %[grad_weight_1]\n"
                : [grad_weight_0] "+fr"(grad_weight_0),
                  [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd),
                  [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [n_frep] "r"(work_div_2_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only
                                                          // 1 channel
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
            "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        asm volatile(
            "beqz %[work_mod_2], 0f\n"  // mod is 0
            "1:\n"
            "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
            "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
            "fmul.d ft1, ft0, %[weight_times_invstd]\n"
            "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0), [grad_bias_0] "+fr"(
                                                        grad_bias_0)
            : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
              [weight_times_invstd] "fr"(weight_times_invstd),
              [invstd] "fr"(invstd), [zero] "fr"(ZERO),
              [work_mod_2] "r"(work_mod_2)
            : "ft0", "ft1", "ft2", "ft4");

        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1;
        // }
        // invstd = *invstd_scratch;
        // weight = *weight_scratch;
        // running_mean = *running_mean_scratch;
        // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
        register double temp_grad_bias, temp_grad_weight;
        asm volatile(
            // interleave 0 resetting and loading between fadd latency
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "fld %[running_mean_times_invstd],0(%[running_mean_scratch])\n"
            "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
            : [temp_grad_bias] "+fr"(temp_grad_bias),
              [temp_grad_weight] "+fr"(temp_grad_weight),
              [grad_weight_0] "+fr"(grad_weight_0),
              [grad_weight_1] "+fr"(grad_weight_1),
              [grad_bias_0] "+fr"(grad_bias_0),
              [grad_bias_1] "+fr"(grad_bias_1),
              [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
              [weight_times_invstd] "=fr"(weight_times_invstd),
              [invstd] "=fr"(invstd)
            : [ZERO] "fr"(ZERO), [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch),
              [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch),
              [grad_weight_scratch] "r"(grad_weight_scratch)
            : "ft0", "ft1", "ft2");
    } while (i < num_channels_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_backward_tile_fp64_looped(
    const double* grad_ofmap_scratch,
    double*
        grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const double* ifmap_scratch, const double* running_mean_scratch,
    const double* weight_scratch, const double* invstd_scratch,
    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t C,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_2,    // precompute to avoid icache branch misses
    uint32_t work_div_2_sub_1, uint32_t tile_size_in_points,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_2 = work_mod_2;
    // use uint32_t for the uint32_t, otherwise the compiler will insert an
    // `andi` for each uint32_t
    register uint32_t frep = work_in_tile >= 2;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    uint32_t buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    // resets for inputs that can be immediately offset
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    DUMP(33);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    snrt_ssr_enable();

    register double grad_weight_0 = ZERO;
    register double grad_weight_1 = ZERO;
    register double grad_weight_2 = ZERO;
    register double grad_bias_0 = ZERO;
    register double grad_bias_1 = ZERO;
    register double grad_bias_2 = ZERO;
    register double invstd = *invstd_scratch;
    register double weight_times_invstd = *weight_scratch;
    register double running_mean_times_invstd = *running_mean_scratch;
    do {  // while (work_in_tile != 0)
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            if (frep != 0) {
                asm volatile(
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                    "fmsub.d ft6, ft2, %[invstd], %[running_mean_times_invstd]\n"
                    "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                    "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                    "fadd.d %[grad_bias_1], ft0, %[grad_bias_1]\n"
                    "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_1], ft6, ft0, %[grad_weight_1]\n"
                    : [grad_weight_0] "+fr"(grad_weight_0),
                      [grad_weight_1] "+fr"(grad_weight_1),
                      [grad_bias_0] "+fr"(grad_bias_0),
                      [grad_bias_1] "+fr"(grad_bias_1)
                    : [running_mean_times_invstd] "fr"(
                          running_mean_times_invstd),
                      [weight_times_invstd] "fr"(weight_times_invstd),
                      [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                      [n_frep] "r"(
                          work_div_2_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[grad_bias_scratch], %[grad_bias_scratch], %[channel_stride_in_bytes]\n"
                "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
                "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n "
                : [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch),
                  [grad_bias_scratch] "+r"(grad_bias_scratch),
                  [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2", "memory");

            // if (is_last_channel) {
            //     invstd_scratch -= channel_stride * num_channels_to_process;
            //     running_mean_scratch -= channel_stride *
            //     num_channels_to_process; weight_scratch -= channel_stride *
            //     num_channels_to_process; buf_flag = !buf_flag;
            //     snrt_cluster_hw_barrier();
            //     work_in_tile = (dm_comm)->num_points_work_in_tile;
            //     work_mod_2 = (dm_comm)->work_mod_2;
            //     work_div_2_sub_1 = (dm_comm)->work_div_2_sub_1;
            //     frep = work_in_tile >= 3;
            //     if (prev_work != work_in_tile) {
            //         prev_work = work_in_tile;
            //         update b0 for 2d ssr
            //     }
            //     if (buf_flag) {
            //         grad_ofmap_scratch += buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch += buf_flag_offset / sizeof(double);
            //         ifmap_scratch += buf_flag_offset / sizeof(double);
            //     } else {
            //         grad_ofmap_scratch -= buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch -= buf_flag_offset / sizeof(double);
            //         ifmap_scratch -= buf_flag_offset / sizeof(double);
            //     }
            // }

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[invstd_scratch], %[invstd_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[weight_scratch], %[weight_scratch],%[input_channel_array_reset_dist]\n"
                "sub %[running_mean_scratch],%[running_mean_scratch],%[input_channel_array_reset_dist]\n "
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_2], %[work_mod_2_offset](%[dm_comm])\n"
                "lw %[work_div_2_sub_1], %[work_div_2_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 2\n"  // cmp frep < 2, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                // // write ssr config if this is not the last iteration
                // "beqz %[work_in_tile], 5f\n"
                // "scfgwi %[grad_ofmap_scratch],%[DM_0] | %[REG_RPTR_2D]<<5\n"
                // "scfgwi %[grad_ifmap_scratch],%[DM_1] | %[REG_WPTR_2D]<<5\n"
                // "scfgwi %[ifmap_scratch],%[DM_2] | %[REG_RPTR_2D]<<5\n"
                // "5:\n"
                : [buf_flag] "+&r"(buf_flag),
                  [invstd_scratch] "+&r"(invstd_scratch),
                  [weight_scratch] "+&r"(weight_scratch),
                  [running_mean_scratch] "+&r"(running_mean_scratch),
                  [work_in_tile] "+&r"(work_in_tile),
                  [next_work_mod_2] "=&r"(next_work_mod_2),
                  [prev_work] "+&r"(prev_work), [frep] "+&r"(frep),
                  [work_div_2_sub_1] "+&r"(work_div_2_sub_1),
                  [grad_ofmap_scratch] "+&r"(grad_ofmap_scratch),
                  [grad_ifmap_scratch] "+&r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_2_offset] "i"(offsetof(dm_comm_t, work_mod_2)),
                  [work_div_2_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_2_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL), [DM_0] "i"(SNRT_SSR_DM0),
                  [DM_1] "i"(SNRT_SSR_DM1), [DM_2] "i"(SNRT_SSR_DM2),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2");

            asm volatile(
                "beqz %[work_mod_2], 0f\n"  // mod is 0
                "1:\n"
                "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                "0:\n"
                // "mv %[work_mod_2], %[next_work_mod_2]\n"
                : [grad_weight_0] "+&fr"(grad_weight_0),
                  [grad_bias_0] "+&fr"(grad_bias_0),
                  [work_mod_2] "+r"(work_mod_2)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd),
                  [invstd] "fr"(invstd), [zero] "fr"(ZERO)
                : "ft0", "ft1", "ft2", "ft4");

            // in plain C:
            // grad_bias_scratch[channel] +=
            //     grad_bias_0 + grad_bias_1;
            // grad_weight_scratch[channel] +=
            //     grad_weight_0 + grad_weight_1;
            // invstd = *invstd_scratch;
            // weight = *weight_scratch;
            // running_mean = *running_mean_scratch;
            // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
            register double temp_grad_bias, temp_grad_weight;
            asm volatile(
                // interleave 0 resetting and loading between fadd latency
                // don't need to synchronize here because the integer core can't
                // issue these instructions until the previous increments have
                // happened
                "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
                "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
                "fld %[invstd],0(%[invstd_scratch])\n"
                // 3 cycles of buffer above for the end of frep
                "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
                "fld %[weight_times_invstd],0(%[weight_scratch])\n"
                "fld %[running_mean_times_invstd],0(%[running_mean_scratch])\n"
                "fadd.d %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
                "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
                "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
                "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
                "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
                : [temp_grad_bias] "+fr"(temp_grad_bias),
                  [temp_grad_weight] "+fr"(temp_grad_weight),
                  [grad_weight_0] "+fr"(grad_weight_0),
                  [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1),
                  [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
                  [weight_times_invstd] "=fr"(weight_times_invstd),
                  [invstd] "=fr"(invstd)
                : [ZERO] "fr"(ZERO), [invstd_scratch] "r"(invstd_scratch),
                  [weight_scratch] "r"(weight_scratch),
                  [running_mean_scratch] "r"(running_mean_scratch),
                  [grad_bias_scratch] "r"(grad_bias_scratch),
                  [grad_weight_scratch] "r"(grad_weight_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        // snrt_ssr_disable();
        // notify that computations for this tile are done
        work_mod_2 = next_work_mod_2;
        grad_weight_scratch -= channel_stride * (num_channels_to_process - 1);
        grad_bias_scratch -= channel_stride * (num_channels_to_process - 1);
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    } while (work_in_tile != 0);
    // // notify last tile done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void __attribute__((always_inline))
batchnorm_backward_fp32_no_loop(
    const v2s* grad_ofmap_scratch,
    v2s* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const v2s* ifmap_scratch, const v2s* running_mean_scratch,
    const v2s* weight_scratch, const v2s* invstd_scratch,
    v2s* grad_bias_scratch, v2s* grad_weight_scratch,
    uint32_t num_bytes_per_point,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_mod_2,  // precompute to avoid icache branch misses
    uint32_t num_doubles_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core,  // dimension of inner loop
        num_doubles_to_process,    // dimension of outer loop
        num_bytes_per_point,       // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    // thought: how to minimize # flops?
    // dy is used for: grad_bias: grad_bias[C] += dy
    //                 grad_ifmap: grad_ifmap[C] =
    //                    (dy * invstd[C] * weight[C])
    //                 grad_weight: grad_weight[C] +=
    //                    (dy * (x[i,C]-running_mean[C]) * invstd[C])
    // for grad_bias: vfadd is sufficient (grad_bias <- dy + grad_bias)
    // for grad_ifmap: vfmul is sufficient (grad_ifmap <- dy * (invstd*weight))
    // for grad_weight: previously i did an fmsub. but there is no vfmsub right
    // now.
    //              Intermediate steps harder though:
    //              (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //              = (dy * (x[i,C]*invstd[C]-(running_mean[C] * invstd[C])))
    // Option 1: sub x-mu, mul (x-mu)*invstd, mul by dy. 3 instr
    // Option 2: mul invstd*dy, sub (x-mu), mul previous two results. 3 instr.
    //   Option 2 is better because we can do the first 2 in parallel without
    //   dependencies
    // Conclusion: I think you have to do 3 instructions without fmadd/fmsub

    uint32_t frep = num_points_work_for_core >= 2;
    uint32_t work_div_2_sub_1 = num_points_work_for_core / 2 -
                                1;  // can underflow, but then frep won't happen
    register volatile uint32_t i =
        0;                         // updated during frep for pseudo-dual issue
    register v2s ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"  // vfcvt.s.x raises exception
                                             // despite smallfloat spec
                 : [ZERO] "=fr"(ZERO.f64)::"ft0", "ft1", "ft2");
    register v2s grad_weight_0 = ZERO;
    register v2s grad_weight_1 = ZERO;
    register v2s grad_bias_0 = ZERO;
    register v2s grad_bias_1 = ZERO;
    register v2s invstd;
    invstd.f64 = invstd_scratch->f64;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    register v2s weight_times_invstd;
    weight_times_invstd.f64 = weight_scratch->f64;
    register v2s running_mean;
    running_mean.f64 = running_mean_scratch->f64;
    // do 1 loop
    do {  // while (i < num_channels_to_process)

        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 10, 0, 0 \n"
                // for grad_ifmap: x - running_mean
                "vfsub.s ft3, ft2, %[running_mean]\n"
                "vfsub.s ft5, ft2, %[running_mean]\n"
                // for grad_ifmap: dy * invstd
                "vfmul.s ft4, %[invstd], ft0\n"
                "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                "vfmul.s ft6, %[invstd], ft0\n"
                "vfadd.s %[grad_bias_1], ft0, %[grad_bias_1]\n"
                "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                // for grad_ifmap: (x - running_mean) * (dy * invstd)
                "vfmac.s %[grad_weight_0], ft3, ft4\n"
                "vfmac.s %[grad_weight_1], ft5, ft6\n"
                // use .f64 instead of .vec because .vec causes everything to be
                // fld/fsd each asm statement
                : [grad_weight_0] "+fr"(grad_weight_0.f64),
                  [grad_weight_1] "+fr"(grad_weight_1.f64),
                  [grad_bias_0] "+fr"(grad_bias_0.f64),
                  [grad_bias_1] "+fr"(grad_bias_1.f64)
                : [running_mean] "fr"(running_mean.f64),
                  [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
                  [n_frep] "r"(work_div_2_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_doubles_to_process], %[i], 2f\n"  // shortcut when only
                                                         // 1 double to process
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
            "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_doubles_to_process] "r"(num_doubles_to_process)
            : "ft0", "ft1", "ft2");

        asm volatile(
            "beqz %[work_mod_2], 0f\n"  // mod is 0
            "1:\n"
            // for grad_ifmap: x - running_mean
            "vfsub.s ft3, ft2, %[running_mean]\n"
            // for grad_ifmap: dy * invstd
            "vfmul.s ft4, %[invstd], ft0\n"
            "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
            "vfmul.s ft1, %[weight_times_invstd], ft0\n"
            // for grad_ifmap: (x - running_mean) * (dy * invstd)
            "vfmac.s %[grad_weight_0], ft3, ft4\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64), [grad_bias_0] "+fr"(
                                                            grad_bias_0.f64)
            : [running_mean] "fr"(running_mean.f64),
              [weight_times_invstd] "fr"(weight_times_invstd.f64),
              [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
              [work_mod_2] "r"(work_mod_2)
            : "ft0", "ft1", "ft2", "ft3", "ft4");

        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1;
        // }
        // invstd = *invstd_scratch;
        // weight = *weight_scratch;
        // running_mean = *running_mean_scratch;
        // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
        asm volatile(
            // interleave 0 resetting and loading between fadd latency
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "fld %[running_mean],0(%[running_mean_scratch])\n"
            "vfadd.s %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "vfadd.s %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "vfsgnj.s %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "vfsgnj.s %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[grad_weight_0],%[ZERO],%[ZERO]\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64),
              [grad_weight_1] "+fr"(grad_weight_1.f64),
              [grad_bias_0] "+fr"(grad_bias_0.f64),
              [grad_bias_1] "+fr"(grad_bias_1.f64),
              [running_mean] "=fr"(running_mean.f64),
              [weight_times_invstd] "=fr"(weight_times_invstd.f64),
              [invstd] "=fr"(invstd.f64)
            : [ZERO] "fr"(ZERO.f64), [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch),
              [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch),
              [grad_weight_scratch] "r"(grad_weight_scratch)
            : "ft0", "ft1", "ft2");
    } while (i < num_doubles_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_backward_tile_fp32_looped(
    const v2s* grad_ofmap_scratch,
    v2s* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const v2s* ifmap_scratch, const v2s* running_mean_scratch,
    const v2s* weight_scratch, const v2s* invstd_scratch,
    v2s* grad_bias_scratch, v2s* grad_weight_scratch,
    uint32_t num_doubles_per_aligned_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_2,    // precompute to avoid icache branch misses
    uint32_t work_div_2_sub_1, uint32_t tile_size_in_aligned_points,
    uint32_t num_doubles_work_for_core_per_point,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_2;
    register uint32_t frep = work_in_tile >= 2;
    register v2s ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=fr"(ZERO.f64)::"ft0", "ft1", "ft2");

    uint32_t buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_aligned_points *
                                     num_doubles_per_aligned_point *
                                     sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_doubles_work_for_core_per_point * sizeof(double);
    const uint32_t inner_loop_stride =
        num_doubles_per_aligned_point * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);

    DUMP(33);
    DUMP(num_doubles_work_for_core_per_point);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,                         // dimension of inner loop
        num_doubles_work_for_core_per_point,  // dimension of outer loop
        inner_loop_stride,   // stride per inner loop iteration: 1 point
        outer_loop_stride);  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    snrt_ssr_enable();

    register v2s grad_weight_0 = ZERO;
    register v2s grad_weight_1 = ZERO;
    register v2s grad_weight_2 = ZERO;
    register v2s grad_bias_0 = ZERO;
    register v2s grad_bias_1 = ZERO;
    register v2s grad_bias_2 = ZERO;
    register v2s invstd;
    invstd.f64 = invstd_scratch->f64;
    register v2s weight_times_invstd;
    weight_times_invstd.f64 = weight_scratch->f64;
    register v2s running_mean;
    running_mean.f64 = running_mean_scratch->f64;
    // thought: how to minimize # flops?
    // dy is used for: grad_bias: grad_bias[C] += dy
    //                 grad_ifmap: grad_ifmap[C] =
    //                    (dy * invstd[C] * weight[C])
    //                 grad_weight: grad_weight[C] +=
    //                    (dy * (x[i,C]-running_mean[C]) * invstd[C])
    // for grad_bias: vfadd is sufficient (grad_bias <- dy + grad_bias)
    // for grad_ifmap: vfmul is sufficient (grad_ifmap <- dy * (invstd*weight))
    // for grad_weight: previously i did an fmsub. but there is no vfmsub right
    // now.
    //              Intermediate steps harder though:
    //              (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //              = (dy * (x[i,C]*invstd[C]-(running_mean[C] * invstd[C])))
    // Option 1: sub x-mu, mul (x-mu)*invstd, mul by dy. 3 instr
    // Option 2: mul invstd*dy, sub (x-mu), mul previous two results. 3 instr.
    //   Option 2 is better because we can do the first 2 in parallel without
    //   dependencies
    // Conclusion: I think you have to do 3 instructions without fmadd/fmsub

    do {  //
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 10, 0, 0 \n"
                    // for grad_ifmap: x - running_mean
                    "vfsub.s ft3, ft2, %[running_mean]\n"
                    "vfsub.s ft5, ft2, %[running_mean]\n"
                    // for grad_ifmap: dy * invstd
                    "vfmul.s ft4, %[invstd], ft0\n"
                    "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
                    "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                    "vfmul.s ft6, %[invstd], ft0\n"
                    "vfadd.s %[grad_bias_1], ft0, %[grad_bias_1]\n"
                    "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                    // for grad_ifmap: (x - running_mean) * (dy * invstd)
                    "vfmac.s %[grad_weight_0], ft3, ft4\n"
                    "vfmac.s %[grad_weight_1], ft5, ft6\n"
                    // use .f64 instead of .vec because .vec causes everything
                    // to be fld/fsd each asm statement
                    : [grad_weight_0] "+fr"(grad_weight_0.f64),
                      [grad_weight_1] "+fr"(grad_weight_1.f64),
                      [grad_bias_0] "+fr"(grad_bias_0.f64),
                      [grad_bias_1] "+fr"(grad_bias_1.f64)
                    : [running_mean] "fr"(running_mean.f64),
                      [weight_times_invstd] "fr"(weight_times_invstd.f64),
                      [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
                      [n_frep] "r"(
                          work_div_2_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
                "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
                "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
                "2:\n"
                : [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch),
                  [grad_bias_scratch] "+r"(grad_bias_scratch),
                  [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point)
                : "ft0", "ft1", "ft2");

            // if (is_last_channel) {
            //     invstd_scratch -= channel_stride * num_channels_to_process;
            //     running_mean_scratch -= channel_stride *
            //     num_channels_to_process; weight_scratch -= channel_stride *
            //     num_channels_to_process; buf_flag = !buf_flag;
            //     snrt_cluster_hw_barrier();
            //     work_in_tile = (dm_comm)->num_points_work_in_tile;
            //     work_mod_2 = (dm_comm)->work_mod_2;
            //     work_div_2_sub_1 = (dm_comm)->work_div_2_sub_1;
            //     frep = work_in_tile >= 3;
            //     if (prev_work != work_in_tile) {
            //         prev_work = work_in_tile;
            //         update b0 for 2d ssr
            //     }
            //     if (buf_flag) {
            //         grad_ofmap_scratch += buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch += buf_flag_offset / sizeof(double);
            //         ifmap_scratch += buf_flag_offset / sizeof(double);
            //     } else {
            //         grad_ofmap_scratch -= buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch -= buf_flag_offset / sizeof(double);
            //         ifmap_scratch -= buf_flag_offset / sizeof(double);
            //     }
            // }

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_doubles_work_for_core_per_point], 2f\n"
                "sub %[invstd_scratch], %[invstd_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[weight_scratch], %[weight_scratch],%[input_channel_array_reset_dist]\n"
                "sub %[running_mean_scratch],%[running_mean_scratch],%[input_channel_array_reset_dist]\n "
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_2], %[work_mod_2_offset](%[dm_comm])\n"
                "lw %[work_div_2_sub_1], %[work_div_2_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 2\n"  // cmp frep < 2, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                // // write ssr config if this is not the last iteration
                // "beqz %[work_in_tile], 5f\n"
                // "scfgwi %[grad_ofmap_scratch],%[DM_0] | %[REG_RPTR_2D]<<5\n"
                // "scfgwi %[grad_ifmap_scratch],%[DM_1] | %[REG_WPTR_2D]<<5\n"
                // "scfgwi %[ifmap_scratch],%[DM_2] | %[REG_RPTR_2D]<<5\n"
                // "5:\n"
                : [buf_flag] "+r"(buf_flag),
                  [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch),
                  [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_2] "=r"(next_work_mod_2),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_2_sub_1] "+r"(work_div_2_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [grad_ifmap_scratch] "+r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_2_offset] "i"(offsetof(dm_comm_t, work_mod_2)),
                  [work_div_2_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_2_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL), [DM_0] "i"(SNRT_SSR_DM0),
                  [DM_1] "i"(SNRT_SSR_DM1), [DM_2] "i"(SNRT_SSR_DM2),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            asm volatile(
                "beqz %[work_mod_2], 0f\n"  // mod is 0
                "1:\n"
                // for grad_ifmap: x - running_mean
                "vfsub.s ft3, ft2, %[running_mean]\n"
                // for grad_ifmap: dy * invstd
                "vfmul.s ft4, %[invstd], ft0\n"
                "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                // for grad_ifmap: (x - running_mean) * (dy * invstd)
                "vfmac.s %[grad_weight_0], ft3, ft4\n"
                "0:\n"
                : [grad_weight_0] "+fr"(grad_weight_0.f64), [grad_bias_0] "+fr"(
                                                                grad_bias_0.f64)
                : [running_mean] "fr"(running_mean.f64),
                  [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
                  [work_mod_2] "r"(work_mod_2)
                : "ft0", "ft1", "ft2", "ft3", "ft4");

            // in plain C:
            // grad_bias_scratch[channel] +=
            //     grad_bias_0 + grad_bias_1;
            // grad_weight_scratch[channel] +=
            //     grad_weight_0 + grad_weight_1;
            // invstd = *invstd_scratch;
            // weight = *weight_scratch;
            // running_mean = *running_mean_scratch;
            // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
            register double temp_grad_bias, temp_grad_weight;
            asm volatile(
                "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
                "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
                "fld %[invstd],0(%[invstd_scratch])\n"
                "vfadd.s %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
                "vfadd.s %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
                // interleave 0 resetting and loading between fadd latency
                // don't need to synchronize here because the integer core can't
                // issue these instructions until the previous increments have
                // happened
                "fld %[weight_times_invstd],0(%[weight_scratch])\n"
                "fld %[running_mean],0(%[running_mean_scratch])\n"
                "vfadd.s %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
                "vfadd.s %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
                "vfsgnj.s %[grad_bias_1],%[ZERO],%[ZERO]\n"
                "vfsgnj.s %[grad_weight_1],%[ZERO],%[ZERO]\n"
                "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
                "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
                "vfsgnj.s %[grad_bias_0],%[ZERO],%[ZERO]\n"
                "vfsgnj.s %[grad_weight_0],%[ZERO],%[ZERO]\n"
                : [temp_grad_bias] "+fr"(temp_grad_bias),
                  [temp_grad_weight] "+fr"(temp_grad_weight),
                  [grad_weight_0] "+fr"(grad_weight_0.f64),
                  [grad_weight_1] "+fr"(grad_weight_1.f64),
                  [grad_bias_0] "+fr"(grad_bias_0.f64),
                  [grad_bias_1] "+fr"(grad_bias_1.f64),
                  [running_mean] "=fr"(running_mean.f64),
                  [weight_times_invstd] "=fr"(weight_times_invstd.f64),
                  [invstd] "=fr"(invstd.f64)
                : [ZERO] "fr"(ZERO.f64), [invstd_scratch] "r"(invstd_scratch),
                  [weight_scratch] "r"(weight_scratch),
                  [running_mean_scratch] "r"(running_mean_scratch),
                  [grad_bias_scratch] "r"(grad_bias_scratch),
                  [grad_weight_scratch] "r"(grad_weight_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_doubles_work_for_core_per_point);
        // don't need to fpu_fence since last 3 instructions are inconsequential

        work_mod_2 = next_work_mod_2;
        grad_weight_scratch -=
            channel_stride * (num_doubles_work_for_core_per_point - 1);
        grad_bias_scratch -=
            channel_stride * (num_doubles_work_for_core_per_point - 1);
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    } while (work_in_tile != 0);
    // Signal that last tile is done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void __attribute__((always_inline))
batchnorm_backward_fp16_no_loop(
    const v4s* grad_ofmap_scratch,
    v4s* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const v4s* ifmap_scratch, const v4s* running_mean_scratch,
    const v4s* weight_scratch, const v4s* invstd_scratch,
    v4s* grad_bias_scratch, v4s* grad_weight_scratch,
    uint32_t num_bytes_per_point,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_mod_2,  // precompute to avoid icache branch misses
    uint32_t num_doubles_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core,  // dimension of inner loop
        num_doubles_to_process,    // dimension of outer loop
        num_bytes_per_point,       // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    // thought: how to minimize # flops?
    // dy is used for: grad_bias: grad_bias[C] += dy
    //                 grad_ifmap: grad_ifmap[C] =
    //                    (dy * invstd[C] * weight[C])
    //                 grad_weight: grad_weight[C] +=
    //                    (dy * (x[i,C]-running_mean[C]) * invstd[C])
    // for grad_bias: vfadd is sufficient (grad_bias <- dy + grad_bias)
    // for grad_ifmap: vfmul is sufficient (grad_ifmap <- dy * (invstd*weight))
    // for grad_weight: previously i did an fmsub. but there is no vfmsub right
    // now.
    //              Intermediate steps harder though:
    //              (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //              = (dy * (x[i,C]*invstd[C]-(running_mean[C] * invstd[C])))
    // Option 1: sub x-mu, mul (x-mu)*invstd, mul by dy. 3 instr
    // Option 2: mul invstd*dy, sub (x-mu), mul previous two results. 3 instr.
    //   Option 2 is better because we can do the first 2 in parallel without
    //   dependencies
    // Conclusion: I think you have to do 3 instructions without fmadd/fmsub

    uint32_t frep = num_points_work_for_core >= 2;
    uint32_t work_div_2_sub_1 = num_points_work_for_core / 2 -
                                1;  // can underflow, but then frep won't happen
    register volatile uint32_t i =
        0;                         // updated during frep for pseudo-dual issue
    register v4s ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"  // vfcvt.s.x raises exception
                                             // despite smallfloat spec
                 : [ZERO] "=fr"(ZERO.f64)::"ft0", "ft1", "ft2");
    register v4s grad_weight_0 = ZERO;
    register v4s grad_weight_1 = ZERO;
    register v4s grad_bias_0 = ZERO;
    register v4s grad_bias_1 = ZERO;
    register v4s invstd;
    invstd.f64 = invstd_scratch->f64;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    register v4s weight_times_invstd;
    weight_times_invstd.f64 = weight_scratch->f64;
    register v4s running_mean;
    running_mean.f64 = running_mean_scratch->f64;
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        asm volatile(
            "vfmul.h %[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
            : [weight_times_invstd] "+fr"(weight_times_invstd.f64)
            : [invstd] "fr"(invstd.f64)
            : "ft0", "ft1", "ft2");

        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 10, 0, 0 \n"
                // for grad_ifmap: x - running_mean
                "vfsub.h ft3, ft2, %[running_mean]\n"
                "vfsub.h ft5, ft2, %[running_mean]\n"
                // for grad_ifmap: dy * invstd
                "vfmul.h ft4, %[invstd], ft0\n"
                "vfadd.h %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "vfmul.h ft1, %[weight_times_invstd], ft0\n"
                "vfmul.h ft6, %[invstd], ft0\n"
                "vfadd.h %[grad_bias_1], ft0, %[grad_bias_1]\n"
                "vfmul.h ft1, %[weight_times_invstd], ft0\n"
                // for grad_ifmap: (x - running_mean) * (dy * invstd)
                "vfmac.h %[grad_weight_0], ft3, ft4\n"
                "vfmac.h %[grad_weight_1], ft5, ft6\n"
                // use .f64 instead of .vec because .vec causes everything to be
                // fld/fsd each asm statement
                : [grad_weight_0] "+fr"(grad_weight_0.f64),
                  [grad_weight_1] "+fr"(grad_weight_1.f64),
                  [grad_bias_0] "+fr"(grad_bias_0.f64),
                  [grad_bias_1] "+fr"(grad_bias_1.f64)
                : [running_mean] "fr"(running_mean.f64),
                  [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
                  [n_frep] "r"(work_div_2_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_doubles_to_process], %[i], 2f\n"  // shortcut when only
                                                         // 1 double to process
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
            "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_doubles_to_process] "r"(num_doubles_to_process)
            : "ft0", "ft1", "ft2");

        asm volatile(
            "beqz %[work_mod_2], 0f\n"  // mod is 0
            "1:\n"
            // for grad_ifmap: x - running_mean
            "vfsub.h ft3, ft2, %[running_mean]\n"
            // for grad_ifmap: dy * invstd
            "vfmul.h ft4, %[invstd], ft0\n"
            "vfadd.h %[grad_bias_0], ft0, %[grad_bias_0]\n"
            "vfmul.h ft1, %[weight_times_invstd], ft0\n"
            // for grad_ifmap: (x - running_mean) * (dy * invstd)
            "vfmac.h %[grad_weight_0], ft3, ft4\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64), [grad_bias_0] "+fr"(
                                                            grad_bias_0.f64)
            : [running_mean] "fr"(running_mean.f64),
              [weight_times_invstd] "fr"(weight_times_invstd.f64),
              [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
              [work_mod_2] "r"(work_mod_2)
            : "ft0", "ft1", "ft2", "ft3", "ft4");

        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1;
        // }
        // invstd = *invstd_scratch;
        // weight = *weight_scratch;
        // running_mean = *running_mean_scratch;
        // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
        asm volatile(
            // interleave 0 resetting and loading between fadd latency
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "vfadd.h %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "vfadd.h %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fld %[running_mean],0(%[running_mean_scratch])\n"
            "vfsgnj.h %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "vfsgnj.h %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "vfsgnj.h %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "vfsgnj.h %[grad_weight_0],%[ZERO],%[ZERO]\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64),
              [grad_weight_1] "+fr"(grad_weight_1.f64),
              [grad_bias_0] "+fr"(grad_bias_0.f64),
              [grad_bias_1] "+fr"(grad_bias_1.f64),
              [running_mean] "=fr"(running_mean.f64),
              [weight_times_invstd] "=fr"(weight_times_invstd.f64),
              [invstd] "=fr"(invstd.f64)
            : [ZERO] "fr"(ZERO.f64), [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch),
              [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch),
              [grad_weight_scratch] "r"(grad_weight_scratch)
            : "ft0", "ft1", "ft2");
    } while (i < num_doubles_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void batchnorm_backward_dma_main_loop_fp_agnostic(
    batchnorm_backward_layer_t* l, uint32_t num_doubles_per_aligned_point,
    uint32_t num_bytes_per_packed_point, uint32_t num_bytes_per_aligned_point,
    uint32_t is_point_aligned_to_8_byte_boundary,
    uint32_t work_left,  // only present for dma
    uint32_t initial_work_in_tile, dm_comm_t* dm_comm,
    uint32_t tile_size_in_points, double* grad_ofmap_scratch,
    double* ifmap_scratch, double* grad_ifmap_scratch, uint32_t buf_flag) {
    snrt_dma_wait_all();

    // signal first iteration
    // compute cores don't have to read dm comm the first time
    snrt_cluster_hw_barrier();

    // skip the first iteration in looping
    uint32_t point_start = initial_work_in_tile;
    uint32_t work_in_tile = initial_work_in_tile;
    DUMP(work_in_tile);
    DUMP(work_left);
    uint32_t prev_point_start = 0;
    uint32_t num_points_work_in_prev_tile = initial_work_in_tile;
    uint32_t prev_prev_work_in_tile = 0;
    const uint32_t target_points_for_last_tile =
        512 / num_doubles_per_aligned_point;
    DUMP(target_points_for_last_tile);

    while (work_left > 0) {
        const uint32_t cycles_per_double = l->dtype == FP64 ? 4 : 5;
        const uint32_t bytes_per_cycle =
            is_point_aligned_to_8_byte_boundary ? 60 : 12;
        uint32_t time_left = num_points_work_in_prev_tile *
                                 num_doubles_per_aligned_point *
                                 cycles_per_double / 8 -
                             prev_prev_work_in_tile *
                                 num_bytes_per_aligned_point / bytes_per_cycle;
        // units: points
        uint32_t max_work_allowed =
            time_left * bytes_per_cycle / (2 * num_bytes_per_aligned_point);
        // need to ensure at least some progress
        max_work_allowed = max(max_work_allowed, target_points_for_last_tile);
        // now max_work_allowed is legal.
        max_work_allowed = ALIGN_UP(max_work_allowed, 2);

        max_work_allowed =
            min(max_work_allowed, min(work_left, tile_size_in_points));

        uint32_t write_out_time =
            work_in_tile * num_bytes_per_aligned_point / bytes_per_cycle;
        if (target_points_for_last_tile * 2 < work_left &&
            write_out_time > (work_left - max_work_allowed) *
                                 num_doubles_per_aligned_point *
                                 cycles_per_double) {
            max_work_allowed = work_left - target_points_for_last_tile;
        }
        work_in_tile = max_work_allowed;
        work_left -= work_in_tile;

        // update comms
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_2 = work_in_tile % 2;
        dm_comm->work_div_2_sub_1 = work_in_tile / 2 - 1;
        // comm what the next iteration will be
        // wait for potential previous grad_ifmap write out?
        snrt_dma_wait_all();

        initiate_dma_1d_or_2d(
            &grad_ofmap_scratch[tile_size_in_points *
                                num_doubles_per_aligned_point * buf_flag],
            &((char*)l->grad_ofmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);
        initiate_dma_1d_or_2d(
            &ifmap_scratch[tile_size_in_points * num_doubles_per_aligned_point *
                           buf_flag],
            &((char*)l->ifmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);

        // signal to core that current tile has information ready
        snrt_cluster_hw_barrier();
        DUMP(55);
        snrt_dma_wait_all();
        // wait for previous tile to be finished computing, signify current
        // tile inputs done loading
        snrt_cluster_hw_barrier();
        DUMP(56);

        // DUMP(prev_point_start);

        initiate_dma_1d_or_2d(
            &((char*)
                  l->grad_ifmap)[prev_point_start * num_bytes_per_packed_point],
            &grad_ifmap_scratch[tile_size_in_points *
                                num_doubles_per_aligned_point * (!buf_flag)],
            num_bytes_per_packed_point, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_points_work_in_prev_tile,
            is_point_aligned_to_8_byte_boundary);
        // snrt_dma_start_1d(
        //     &l->grad_ifmap[prev_point_start * num_doubles_per_aligned_point],
        //     &grad_ifmap_scratch[tile_size_in_points *
        //                         num_doubles_per_aligned_point *
        //                         (!buf_flag)],  // take !buf_flag
        //                                        // dma core is one
        //                                        // iteration ahead
        //                                        // of compute core
        //     num_points_work_in_prev_tile * num_doubles_per_aligned_point *
        //         sizeof(double));
        prev_point_start = point_start;
        prev_prev_work_in_tile = num_points_work_in_prev_tile;
        num_points_work_in_prev_tile = work_in_tile;
        point_start += work_in_tile;
        buf_flag = !buf_flag;
    }
    dm_comm->num_points_work_in_tile = 0;
    dm_comm->work_mod_2 = 0;
    dm_comm->work_div_2_sub_1 = 0xdeadbeef;
    // signal last iteration that there is no more work
    snrt_cluster_hw_barrier();
    // wait for last tile to finish
    snrt_cluster_hw_barrier();

    initiate_dma_1d_or_2d(
        &((char*)l->grad_ifmap)[prev_point_start * num_bytes_per_packed_point],
        &grad_ifmap_scratch[tile_size_in_points *
                            num_doubles_per_aligned_point * (!buf_flag)],
        num_bytes_per_packed_point, num_bytes_per_packed_point,
        num_bytes_per_aligned_point, num_points_work_in_prev_tile,
        is_point_aligned_to_8_byte_boundary);
    snrt_dma_wait_all();
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_fp64_no_loop_1(
    const double* grad_ofmap_scratch, const double* ifmap_scratch,
    const double* current_mean_scratch, double* sum_scratch,
    double* dotp_scratch, uint32_t C, uint32_t num_points_work_for_core_in_tile,
    uint32_t work_mod_3, uint32_t work_div_3_sub_1,
    uint32_t num_channels_to_process, uint32_t channel_stride) {
    uint32_t frep = num_points_work_for_core_in_tile >= 3;
    register volatile uint32_t i = 0;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double sum_0 = ZERO;
    register double sum_1 = ZERO;
    register double sum_2 = ZERO;
    register double dotp_0 = ZERO;
    register double dotp_1 = ZERO;
    register double dotp_2 = ZERO;
    register double current_mean = *current_mean_scratch;
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core_in_tile,  // dimension of inner loop
        num_channels_to_process,           // dimension of outer loop
        C * sizeof(double),  // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 2);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    do {
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 9, 0, 0 \n"
                "fsub.d ft8, ft2, %[current_mean]\n"
                "fsub.d ft6, ft2, %[current_mean]\n"
                "fsub.d ft4, ft2, %[current_mean]\n"
                "fadd.d %[sum_2], ft0, %[sum_2] \n"
                "fmadd.d %[dotp_2], ft8, ft0, %[dotp_2]\n"
                "fadd.d %[sum_1], ft0, %[sum_1] \n"
                "fmadd.d %[dotp_1], ft6, ft0, %[dotp_1]\n"
                "fadd.d %[sum_0], ft0, %[sum_0] \n"
                "fmadd.d %[dotp_0], ft4, ft0, %[dotp_0]\n"
                : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1),
                  [sum_2] "+fr"(sum_2), [dotp_0] "+fr"(dotp_0),
                  [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2)
                : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                  [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft4", "ft6", "ft8");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
            "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
            : [current_mean_scratch] "+r"(current_mean_scratch),
              [sum_scratch] "+r"(sum_scratch),
              [dotp_scratch] "+r"(dotp_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_3], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if
                                                    // yes, then mod is 1
            "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
            "2:\n"
            "fsub.d ft6, ft2, %[current_mean]\n"
            "fsub.d ft4, ft2, %[current_mean]\n"
            "fadd.d %[sum_1], ft0, %[sum_1] \n"
            "fmadd.d %[dotp_1], ft6, ft0,%[dotp_1]\n"
            "fadd.d %[sum_0], ft0, %[sum_0] \n"
            "fmadd.d %[dotp_0], ft4, ft0,%[dotp_0]\n"
            "j 0f\n"
            "1:\n"
            "fsub.d ft4, ft2, %[current_mean]\n"
            "fadd.d %[sum_0], ft0, %[sum_0] \n"
            "fmadd.d %[dotp_0], ft4, ft0,%[dotp_0]\n"
            "0:\n"
            : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
              [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1),
              [dotp_2] "+fr"(dotp_2), [mod_temp] "=r"(mod_temp)
            : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
              [work_mod_3] "r"(work_mod_3), [n_frep] "r"(work_div_3_sub_1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

        register double temp_sum, temp_dotp;

        asm volatile(
            "fld %[temp_sum], 0(%[sum_scratch])\n"
            "fld %[temp_dotp], 0(%[dotp_scratch])\n"
            "fld %[current_mean], 0(%[current_mean_scratch])\n"
            // 1+= 2
            "fadd.d %[sum_1], %[sum_1], %[sum_2]\n"
            "fadd.d %[dotp_1], %[dotp_1], %[dotp_2]\n"
            "fsgnj.d %[sum_2],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_2],%[ZERO],%[ZERO]\n"
            // temp+=0
            "fadd.d %[temp_sum], %[temp_sum], %[sum_0]\n"
            "fadd.d %[temp_dotp], %[temp_dotp], %[dotp_0]\n"
            "fsgnj.d %[sum_0],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_0],%[ZERO],%[ZERO]\n"
            // temp+=1
            "fadd.d %[temp_sum], %[sum_1], %[temp_sum]\n"
            "fadd.d %[temp_dotp], %[dotp_1], %[temp_dotp]\n"
            "fsgnj.d %[sum_1],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_1],%[ZERO],%[ZERO]\n"
            "fsd %[temp_sum], 0(%[sum_scratch])\n"
            "fsd %[temp_dotp], 0(%[dotp_scratch])\n"
            : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
              [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
              [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1),
              [dotp_2] "+fr"(dotp_2), [current_mean] "=&fr"(current_mean)
            : [ZERO] "fr"(ZERO), [sum_scratch] "r"(sum_scratch),
              [dotp_scratch] "r"(dotp_scratch),
              [current_mean_scratch] "r"(current_mean_scratch)
            : "ft0", "ft1", "ft2");

    } while (i < num_channels_to_process);

    snrt_ssr_disable();
    snrt_ssr_repeat(SNRT_SSR_DM0, 1);
}

static inline uint32_t __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_looped_1(
    const double* grad_ofmap_scratch, const double* ifmap_scratch,
    const double* current_mean_scratch, double* sum_scratch,
    double* dotp_scratch, uint32_t C,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_3,    // precompute to avoid icache branch misses
    uint32_t work_div_3_sub_1, uint32_t tile_size_in_points,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_3;
    register uint32_t frep = work_in_tile >= 3;
    register double ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    uint32_t buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);

    register double sum_0 = ZERO;
    register double sum_1 = ZERO;
    register double sum_2 = ZERO;
    register double dotp_0 = ZERO;
    register double dotp_1 = ZERO;
    register double dotp_2 = ZERO;
    register double current_mean = *current_mean_scratch;

    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 2);
    snrt_ssr_enable();
    do {
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            // Can only manual unroll 5 times since the max for frep is 16
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 9, 0, 0 \n"
                    "fsub.d ft8, ft2, %[current_mean]\n"
                    "fsub.d ft6, ft2, %[current_mean]\n"
                    "fsub.d ft4, ft2, %[current_mean]\n"
                    "fadd.d %[sum_2], ft0, %[sum_2] \n"
                    "fmadd.d %[dotp_2], ft8, ft0, %[dotp_2]\n"
                    "fadd.d %[sum_1], ft0, %[sum_1] \n"
                    "fmadd.d %[dotp_1], ft6, ft0, %[dotp_1]\n"
                    "fadd.d %[sum_0], ft0, %[sum_0] \n"
                    "fmadd.d %[dotp_0], ft4, ft0, %[dotp_0]\n"
                    : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1),
                      [sum_2] "+fr"(sum_2), [dotp_0] "+fr"(dotp_0),
                      [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2)
                    : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                      [n_frep] "r"(work_div_3_sub_1)
                    : "ft0", "ft1", "ft2", "ft4", "ft6", "ft8");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
                "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
                : [current_mean_scratch] "+r"(current_mean_scratch),
                  [sum_scratch] "+r"(sum_scratch),
                  [dotp_scratch] "+r"(dotp_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[current_mean_scratch], %[current_mean_scratch], %[input_channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_3], %[work_mod_3_offset](%[dm_comm])\n"
                "lw %[work_div_3_sub_1], %[work_div_3_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 3\n"  // cmp frep < 3, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag),
                  [current_mean_scratch] "+r"(current_mean_scratch),
                  [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_3] "=r"(next_work_mod_3),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_3_sub_1] "+r"(work_div_3_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_3_offset] "i"(offsetof(dm_comm_t, work_mod_3)),
                  [work_div_3_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_3_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0");

            register uint32_t mod_temp;
            asm volatile(
                "beqz %[work_mod_3], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if
                                                        // yes, then mod is 1
                "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
                "2:\n"
                "fsub.d ft6, ft2, %[current_mean]\n"
                "fsub.d ft4, ft2, %[current_mean]\n"
                "fadd.d %[sum_1], ft0, %[sum_1] \n"
                "fmadd.d %[dotp_1], ft6, ft0,%[dotp_1]\n"
                "fadd.d %[sum_0], ft0, %[sum_0] \n"
                "fmadd.d %[dotp_0], ft4, ft0,%[dotp_0]\n"
                "j 0f\n"
                "1:\n"
                "fsub.d ft4, ft2, %[current_mean]\n"
                "fadd.d %[sum_0], ft0, %[sum_0] \n"
                "fmadd.d %[dotp_0], ft4, ft0,%[dotp_0]\n"
                "0:\n"
                : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1),
                  [sum_2] "+fr"(sum_2), [dotp_0] "+fr"(dotp_0),
                  [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2),
                  [mod_temp] "=r"(mod_temp)
                : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                  [work_mod_3] "r"(work_mod_3), [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register double temp_sum, temp_dotp;
            asm volatile(
                "fld %[temp_sum], 0(%[sum_scratch])\n"
                "fld %[temp_dotp], 0(%[dotp_scratch])\n"
                "fld %[current_mean], 0(%[current_mean_scratch])\n"
                // 1+= 2
                "fadd.d %[sum_1], %[sum_1], %[sum_2]\n"
                "fadd.d %[dotp_1], %[dotp_1], %[dotp_2]\n"
                "fsgnj.d %[sum_2],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_2],%[ZERO],%[ZERO]\n"
                // temp+=0
                "fadd.d %[temp_sum], %[temp_sum], %[sum_0]\n"
                "fadd.d %[temp_dotp], %[temp_dotp], %[dotp_0]\n"
                "fsgnj.d %[sum_0],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_0],%[ZERO],%[ZERO]\n"
                // temp+=1
                "fadd.d %[temp_sum], %[sum_1], %[temp_sum]\n"
                "fadd.d %[temp_dotp], %[dotp_1], %[temp_dotp]\n"
                "fsgnj.d %[sum_1],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_1],%[ZERO],%[ZERO]\n"
                "fsd %[temp_sum], 0(%[sum_scratch])\n"
                "fsd %[temp_dotp], 0(%[dotp_scratch])\n"
                : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
                  [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1),
                  [sum_2] "+fr"(sum_2), [dotp_0] "+fr"(dotp_0),
                  [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2),
                  [current_mean] "=&fr"(current_mean)
                : [ZERO] "fr"(ZERO), [sum_scratch] "r"(sum_scratch),
                  [dotp_scratch] "r"(dotp_scratch),
                  [current_mean_scratch] "r"(current_mean_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        // __builtin_ssr_barrier(SNRT_SSR_DM1);
        work_mod_3 = next_work_mod_3;
        sum_scratch -= channel_stride * (num_channels_to_process - 1);
        dotp_scratch -= channel_stride * (num_channels_to_process - 1);
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_ssr_disable();
    snrt_ssr_repeat(SNRT_SSR_DM0, 1);
    snrt_cluster_hw_barrier();
    return buf_flag;
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_fp64_no_loop_2(
    const double* grad_ofmap_scratch, double* grad_ifmap_scratch,
    const double* ifmap_scratch, const double* weight_times_invstd_scratch,
    const double* k_scratch,
    const double* winvstd_times_meank_sub_dmean_scratch, uint32_t C,
    uint32_t num_points_work_for_core_in_tile, uint32_t work_mod_4,
    uint32_t work_div_4_sub_1, uint32_t num_channels_to_process,
    uint32_t channel_stride) {
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core_in_tile,  // dimension of inner loop
        num_channels_to_process,           // dimension of outer loop
        C * sizeof(double),  // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    uint32_t frep = num_points_work_for_core_in_tile >= 4;
    register volatile uint32_t i = 0;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
    do {
        register double weight_times_invstd = *weight_times_invstd_scratch;
        register double winvstd_times_meank_sub_dmean =
            *winvstd_times_meank_sub_dmean_scratch;
        register double k = *k_scratch;
        snrt_ssr_enable();
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                // -xk+dy
                "fnmsub.d ft3, ft0, %[k], ft2 \n"
                "fnmsub.d ft4, ft0, %[k], ft2 \n"
                "fnmsub.d ft5, ft0, %[k], ft2 \n"
                "fnmsub.d ft6, ft0, %[k], ft2 \n"
                // weight*invstd*(dy-xk)+(weight*invstd*(mean*k-grad_mean))
                "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft5, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft6, %[winvstd_times_meank_sub_dmean]\n"
                :
                : [weight_times_invstd] "fr"(weight_times_invstd),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean),
                  [k] "fr"(k), [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only
                                                          // 1 channel
            "add %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
            "add %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[channel_stride_in_bytes]\n"
            "2:\n"
            : [weight_times_invstd_scratch] "+r"(weight_times_invstd_scratch),
              [k_scratch] "+r"(k_scratch),
              [winvstd_times_meank_sub_dmean_scratch] "+r"(
                  winvstd_times_meank_sub_dmean_scratch),
              [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_4], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                    // then mod is 2
            "beqz %[mod_temp], 2f\n"                // jump to 2 if no
            "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                    // then mod is 1
            "beqz %[mod_temp], 1f\n"                // jump to 1 if no
            "3:\n"
            "fnmsub.d ft3, ft0, %[k], ft2 \n"
            "fnmsub.d ft4, ft0, %[k], ft2 \n"
            "fnmsub.d ft5, ft0, %[k], ft2 \n"
            "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
            "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
            "fmadd.d ft1, %[weight_times_invstd], ft5, %[winvstd_times_meank_sub_dmean]\n"
            "j 0f\n"
            "2:\n"
            "fnmsub.d ft3, ft0, %[k], ft2 \n"
            "fnmsub.d ft4, ft0, %[k], ft2 \n"
            "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
            "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
            "j 0f\n"
            "1:\n"
            "fnmsub.d ft3, ft0, %[k], ft2 \n"
            "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
            "0:\n"
            : [mod_temp] "=r"(mod_temp)
            : [weight_times_invstd] "fr"(weight_times_invstd),
              [winvstd_times_meank_sub_dmean] "fr"(
                  winvstd_times_meank_sub_dmean),
              [k] "fr"(k), [work_mod_4] "r"(work_mod_4)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
        snrt_ssr_disable();
    } while (i < num_channels_to_process);
    __builtin_ssr_barrier(SNRT_SSR_DM1);
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_looped_2(
    const double* grad_ofmap_scratch,
    double*
        grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const double* ifmap_scratch, const double* weight_times_invstd_scratch,
    const double* k_scratch,
    const double* winvstd_times_meank_sub_dmean_scratch, uint32_t buf_flag,
    uint32_t C,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_4,    // precompute to avoid icache branch misses
    uint32_t work_div_4_sub_1, uint32_t tile_size_in_points,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_4;
    register uint32_t frep = work_in_tile >= 4;
    register double ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration
    snrt_ssr_enable();

    if (buf_flag) {
        ifmap_scratch += tile_size_in_points * C;
        grad_ifmap_scratch += tile_size_in_points * C;
        grad_ofmap_scratch += tile_size_in_points * C;
    }

    do {
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            register double weight_times_invstd = *weight_times_invstd_scratch;
            register double winvstd_times_meank_sub_dmean =
                *winvstd_times_meank_sub_dmean_scratch;
            register double k = *k_scratch;
            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                // -xk+dy
                "fnmsub.d ft3, ft0, %[k], ft2 \n"
                "fnmsub.d ft4, ft0, %[k], ft2 \n"
                "fnmsub.d ft5, ft0, %[k], ft2 \n"
                "fnmsub.d ft6, ft0, %[k], ft2 \n"
                // weight*invstd*(dy-xk)+(weight*invstd*(mean*k-grad_mean))
                "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft5, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft6, %[winvstd_times_meank_sub_dmean]\n"
                :
                : [weight_times_invstd] "fr"(weight_times_invstd),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean),
                  [k] "fr"(k), [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "addi %[i], %[i], 1\n"
                "add %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
                "add %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[channel_stride_in_bytes]\n"
                : [weight_times_invstd_scratch] "+r"(
                      weight_times_invstd_scratch),
                  [k_scratch] "+r"(k_scratch),
                  [winvstd_times_meank_sub_dmean_scratch] "+r"(
                      winvstd_times_meank_sub_dmean_scratch),
                  [i] "+r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                // extra check here for channels == 1. THen don't sub
                "sub %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[k_scratch], %[k_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[input_channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_4], %[work_mod_4_offset](%[dm_comm])\n"
                "lw %[work_div_4_sub_1], %[work_div_4_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 4\n"  // cmp frep < 4, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag),
                  [weight_times_invstd_scratch] "+r"(
                      weight_times_invstd_scratch),
                  [k_scratch] "+r"(k_scratch),
                  [winvstd_times_meank_sub_dmean_scratch] "+r"(
                      winvstd_times_meank_sub_dmean_scratch),
                  [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_4] "=r"(next_work_mod_4),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_4_sub_1] "+r"(work_div_4_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [grad_ifmap_scratch] "+r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_4_offset] "i"(offsetof(dm_comm_t, work_mod_4)),
                  [work_div_4_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_4_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            register uint32_t mod_temp;
            asm volatile(
                "beqz %[work_mod_4], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                        // then mod is 2
                "beqz %[mod_temp], 2f\n"                // jump to 2 if no
                "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                        // then mod is 1
                "beqz %[mod_temp], 1f\n"                // jump to 1 if no
                "3:\n"
                "fnmsub.d ft3, ft0, %[k], ft2 \n"
                "fnmsub.d ft4, ft0, %[k], ft2 \n"
                "fnmsub.d ft5, ft0, %[k], ft2 \n"
                "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft5, %[winvstd_times_meank_sub_dmean]\n"
                "j 0f\n"
                "2:\n"
                "fnmsub.d ft3, ft0, %[k], ft2 \n"
                "fnmsub.d ft4, ft0, %[k], ft2 \n"
                "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
                "fmadd.d ft1, %[weight_times_invstd], ft4, %[winvstd_times_meank_sub_dmean]\n"
                "j 0f\n"
                "1:\n"
                "fnmsub.d ft3, ft0, %[k], ft2 \n"
                "fmadd.d ft1, %[weight_times_invstd], ft3, %[winvstd_times_meank_sub_dmean]\n"
                "0:\n"
                : [mod_temp] "=r"(mod_temp)
                : [weight_times_invstd] "fr"(weight_times_invstd),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean),
                  [k] "fr"(k), [work_mod_4] "r"(work_mod_4)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        work_mod_4 = next_work_mod_4;
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline uint32_t batchnorm_backward_training_dma_main_loop_fp_agnostic(
    batchnorm_backward_training_layer_t* l,
    uint32_t num_doubles_per_aligned_point, uint32_t num_bytes_per_packed_point,
    uint32_t num_bytes_per_aligned_point,
    uint32_t is_point_aligned_to_8_byte_boundary, uint32_t num_points,
    uint32_t work_left,  // only present for dma
    uint32_t initial_work_in_tile, dm_comm_t* dm_comm,
    uint32_t tile_size_in_points, double* grad_ofmap_scratch,
    double* ifmap_scratch, double* grad_ifmap_scratch, uint32_t buf_flag,
    uint32_t unroll, uint32_t cycles_per_double) {
    snrt_dma_wait_all();

    // signal first iteration
    // compute cores don't have to read dm comm the first time
    snrt_cluster_hw_barrier();

    // skip the first iteration in looping
    uint32_t point_start = initial_work_in_tile;
    uint32_t work_in_tile = initial_work_in_tile;
    uint32_t prev_point_start = 0;
    uint32_t num_points_work_in_prev_tile = initial_work_in_tile;
    // split the remaining work "nicely"
    uint32_t min_loops = ceildiv(work_left, tile_size_in_points);
    // align up to multiple of 2 because that avoids stalling in fpu the
    // best

    uint32_t ideal_work_in_tile =
        min(align_up_non_power_of_2(ceildiv(work_left, min_loops), unroll),
            tile_size_in_points);

    const uint32_t target_points_for_last_tile =
        512 / num_doubles_per_aligned_point;

    while (work_left > 0) {
        // TODO: adaptive scaling but only for loop 1
        if (grad_ifmap_scratch != NULL) {
            work_in_tile = min(ideal_work_in_tile, work_left);
            if (work_in_tile == work_left &&
                target_points_for_last_tile * 2 < work_in_tile) {
                work_in_tile -= target_points_for_last_tile;
            }
        } else {
            work_in_tile = min(ideal_work_in_tile, work_left);
        }

        work_left -= work_in_tile;
        // update comms
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_unroll = work_in_tile % unroll;
        dm_comm->work_div_unroll_sub_1 = work_in_tile / unroll - 1;
        // comm what the next iteration will be
        // wait for potential previous grad_ifmap write out?
        snrt_dma_wait_all();

        initiate_dma_1d_or_2d(
            &grad_ofmap_scratch[tile_size_in_points *
                                num_doubles_per_aligned_point * buf_flag],
            &((char*)l->grad_ofmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);
        initiate_dma_1d_or_2d(
            &ifmap_scratch[tile_size_in_points * num_doubles_per_aligned_point *
                           buf_flag],
            &((char*)l->ifmap)[point_start * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, work_in_tile,
            is_point_aligned_to_8_byte_boundary);
        // signal to core that current tile has information ready
        snrt_cluster_hw_barrier();
        DUMP(55);
        snrt_dma_wait_all();
        // wait for previous tile to be finished computing, signify current
        // tile inputs done loading
        snrt_cluster_hw_barrier();
        DUMP(56);
        if (grad_ifmap_scratch != NULL) {
            initiate_dma_1d_or_2d(
                &((char*)l->grad_ifmap)[prev_point_start *
                                        num_bytes_per_packed_point],
                &grad_ifmap_scratch[tile_size_in_points *
                                    num_doubles_per_aligned_point *
                                    (!buf_flag)],
                num_bytes_per_packed_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point, num_points_work_in_prev_tile,
                is_point_aligned_to_8_byte_boundary);
        }
        prev_point_start = point_start;
        num_points_work_in_prev_tile = work_in_tile;
        point_start += work_in_tile;
        buf_flag = !buf_flag;
    }
    dm_comm->num_points_work_in_tile = 0;
    dm_comm->work_mod_unroll = 0;
    dm_comm->work_div_unroll_sub_1 = 0xdeadbeef;
    if (grad_ifmap_scratch == NULL) {
        // load in next batch
        initiate_dma_1d_or_2d(
            &grad_ofmap_scratch[tile_size_in_points *
                                num_doubles_per_aligned_point * buf_flag],
            &((char*)l->grad_ofmap)[0 * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, min(tile_size_in_points, num_points),
            is_point_aligned_to_8_byte_boundary);
        initiate_dma_1d_or_2d(
            &ifmap_scratch[tile_size_in_points * num_doubles_per_aligned_point *
                           buf_flag],
            &((char*)l->ifmap)[0 * num_bytes_per_packed_point],
            num_bytes_per_packed_point, num_bytes_per_aligned_point,
            num_bytes_per_packed_point, min(tile_size_in_points, num_points),
            is_point_aligned_to_8_byte_boundary);
    }
    // signal last iteration that there is no more work
    snrt_cluster_hw_barrier();
    // wait for last tile to finish
    snrt_cluster_hw_barrier();
    if (grad_ifmap_scratch != NULL) {
        initiate_dma_1d_or_2d(
            &((char*)
                  l->grad_ifmap)[prev_point_start * num_bytes_per_packed_point],
            &grad_ifmap_scratch[tile_size_in_points *
                                num_doubles_per_aligned_point * (!buf_flag)],
            num_bytes_per_packed_point, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_points_work_in_prev_tile,
            is_point_aligned_to_8_byte_boundary);
    }
    buf_flag = !buf_flag;
    return buf_flag;
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_fp32_no_loop_1(
    const v2s* grad_ofmap_scratch, const v2s* ifmap_scratch,
    const v2s* current_mean_scratch, v2s* sum_scratch, v2s* dotp_scratch,
    uint32_t num_bytes_per_point, uint32_t num_points_work_for_core_in_tile,
    uint32_t work_mod_3, uint32_t work_div_3_sub_1,
    uint32_t num_doubles_to_process, uint32_t channel_stride) {
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core_in_tile,  // dimension of inner loop
        num_doubles_to_process,            // dimension of outer loop
        num_bytes_per_point,  // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    uint32_t frep = num_points_work_for_core_in_tile >= 3;
    register volatile uint32_t i = 0;
    register v2s ZERO asm("ft9");            // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"  // vfcvt.s.x raises exception
                                             // despite smallfloat spec
                 : [ZERO] "=fr"(ZERO.f64)::"ft0", "ft1", "ft2");
    register v2s sum_0 = ZERO;
    register v2s sum_1 = ZERO;
    register v2s sum_2 = ZERO;
    register v2s dotp_0 = ZERO;
    register v2s dotp_1 = ZERO;
    register v2s dotp_2 = ZERO;
    register v2s current_mean;
    current_mean.f64 = current_mean_scratch->f64;
    snrt_ssr_repeat(SNRT_SSR_DM0, 2);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, ifmap_scratch);
    do {
        snrt_ssr_enable();
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 9, 0, 0 \n"
                "vfsub.s ft4, ft1, %[current_mean]\n"
                "vfsub.s ft6, ft1, %[current_mean]\n"
                "vfsub.s ft8, ft1, %[current_mean]\n"
                "vfadd.s %[sum_0], ft0, %[sum_0] \n"
                "vfmac.s %[dotp_0], ft4, ft0\n"
                "vfadd.s %[sum_1], ft0, %[sum_1] \n"
                "vfmac.s %[dotp_1], ft6, ft0\n"
                "vfadd.s %[sum_2], ft0, %[sum_2] \n"
                "vfmac.s %[dotp_2], ft8, ft0\n"
                : [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
                  [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
                  [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64)
                : [current_mean] "fr"(current_mean.f64), [zero] "fr"(ZERO.f64),
                  [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                  "ft8");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
            "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_doubles_to_process], %[i], 2f\n"  // shortcut when only
                                                         // 1 channel
            "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
            "2:\n"
            : [current_mean_scratch] "+r"(current_mean_scratch),
              [sum_scratch] "+r"(sum_scratch),
              [dotp_scratch] "+r"(dotp_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_doubles_to_process] "r"(num_doubles_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_3], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if no,
                                                    // then mod is 2
            "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
            "2:\n"
            "vfsub.s ft4, ft1, %[current_mean]\n"
            "vfsub.s ft6, ft1, %[current_mean]\n"
            "vfadd.s %[sum_0], ft0, %[sum_0] \n"
            "vfmac.s %[dotp_0], ft4, ft0\n"
            "vfadd.s %[sum_1], ft0, %[sum_1] \n"
            "vfmac.s %[dotp_1], ft6, ft0\n"
            "j 0f\n"
            "1:\n"
            "vfsub.s ft4, ft1, %[current_mean]\n"
            "vfadd.s %[sum_0], ft0, %[sum_0] \n"
            "vfmac.s %[dotp_0], ft4, ft0\n"
            "0:\n"
            : [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
              [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
              [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64),
              [mod_temp] "=r"(mod_temp)
            : [current_mean] "fr"(current_mean.f64), [zero] "fr"(ZERO.f64),
              [work_mod_3] "r"(work_mod_3), [n_frep] "r"(work_div_3_sub_1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

        register double temp_sum, temp_dotp;
        asm volatile(
            "fld %[temp_sum], 0(%[sum_scratch])\n"
            "fld %[temp_dotp], 0(%[dotp_scratch])\n"
            "fld %[current_mean], 0(%[current_mean_scratch])\n"
            // 1+= 2
            "vfadd.s %[sum_1], %[sum_1], %[sum_2]\n"
            "vfadd.s %[dotp_1], %[dotp_1], %[dotp_2]\n"
            "vfsgnj.s %[sum_2],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[dotp_2],%[ZERO],%[ZERO]\n"
            // temp+=0
            "vfadd.s %[temp_sum], %[temp_sum], %[sum_0]\n"
            "vfadd.s %[temp_dotp], %[temp_dotp], %[dotp_0]\n"
            "vfsgnj.s %[sum_0],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[dotp_0],%[ZERO],%[ZERO]\n"
            // temp+=1
            "vfadd.s %[temp_sum], %[sum_1], %[temp_sum]\n"
            "vfadd.s %[temp_dotp], %[dotp_1], %[temp_dotp]\n"
            "vfsgnj.s %[sum_1],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[dotp_1],%[ZERO],%[ZERO]\n"
            "fsd %[temp_sum], 0(%[sum_scratch])\n"
            "fsd %[temp_dotp], 0(%[dotp_scratch])\n"
            : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
              [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
              [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
              [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64),
              [current_mean] "=&fr"(current_mean.f64)
            : [ZERO] "fr"(ZERO.f64), [sum_scratch] "r"(sum_scratch),
              [dotp_scratch] "r"(dotp_scratch),
              [current_mean_scratch] "r"(current_mean_scratch)
            : "ft0", "ft1", "ft2");
        snrt_fpu_fence();
        snrt_ssr_disable();
    } while (i < num_doubles_to_process);
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_repeat(SNRT_SSR_DM0, 1);
}

static inline uint32_t __attribute__((always_inline))
batchnorm_backward_training_tile_fp32_looped_1(
    const v2s* grad_ofmap_scratch, const v2s* ifmap_scratch,
    const v2s* current_mean_scratch, v2s* sum_scratch, v2s* dotp_scratch,
    uint32_t num_doubles_per_aligned_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_3,    // precompute to avoid icache branch misses
    uint32_t work_div_3_sub_1, uint32_t tile_size_in_aligned_points,
    uint32_t num_doubles_work_for_core_per_point,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_3;
    register uint32_t frep = work_in_tile >= 3;
    register v2s ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    uint32_t buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_aligned_points *
                                     num_doubles_per_aligned_point *
                                     sizeof(double);
    const uint32_t channel_array_reset_dist =
        channel_stride * num_doubles_work_for_core_per_point * sizeof(double);
    const uint32_t inner_loop_stride =
        num_doubles_per_aligned_point * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    register v2s sum_0 = ZERO;
    register v2s sum_1 = ZERO;
    register v2s sum_2 = ZERO;
    register v2s dotp_0 = ZERO;
    register v2s dotp_1 = ZERO;
    register v2s dotp_2 = ZERO;
    register v2s current_mean;
    current_mean.f64 = current_mean_scratch->f64;
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,                         // dimension of inner loop
        num_doubles_work_for_core_per_point,  // dimension of outer loop
        inner_loop_stride,   // stride per inner loop iteration: 1 point
        outer_loop_stride);  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 2);
    snrt_ssr_enable();
    // TODO: fix num_channels_work_for_core == 0.
    do {
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 9, 0, 0 \n"
                    "vfsub.s ft4, ft1, %[current_mean]\n"
                    "vfsub.s ft6, ft1, %[current_mean]\n"
                    "vfsub.s ft8, ft1, %[current_mean]\n"
                    "vfadd.s %[sum_0], ft0, %[sum_0] \n"
                    "vfmac.s %[dotp_0], ft4, ft0\n"
                    "vfadd.s %[sum_1], ft0, %[sum_1] \n"
                    "vfmac.s %[dotp_1], ft6, ft0\n"
                    "vfadd.s %[sum_2], ft0, %[sum_2] \n"
                    "vfmac.s %[dotp_2], ft8, ft0\n"
                    : [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
                      [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
                      [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64)
                    : [current_mean] "fr"(current_mean.f64),
                      [zero] "fr"(ZERO.f64),
                      [n_frep] "r"(
                          work_div_3_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft4", "ft6", "ft8");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
                "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
                : [current_mean_scratch] "+r"(current_mean_scratch),
                  [sum_scratch] "+r"(sum_scratch),
                  [dotp_scratch] "+r"(dotp_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_doubles_work_for_core_per_point], 2f\n"
                "sub %[current_mean_scratch], %[current_mean_scratch], %[channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_3], %[work_mod_3_offset](%[dm_comm])\n"
                "lw %[work_div_3_sub_1], %[work_div_3_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 3\n"  // cmp frep < 3, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag),
                  [current_mean_scratch] "+r"(current_mean_scratch),
                  [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_3] "=r"(next_work_mod_3),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_3_sub_1] "+r"(work_div_3_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point),
                  [channel_array_reset_dist] "r"(channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_3_offset] "i"(offsetof(dm_comm_t, work_mod_3)),
                  [work_div_3_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_3_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0");

            asm volatile(
                "beqz %[work_mod_3], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if
                                                        // yes, then mod is 1
                "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
                "2:\n"
                "vfsub.s ft4, ft1, %[current_mean]\n"
                "vfsub.s ft6, ft1, %[current_mean]\n"
                "vfadd.s %[sum_0], ft0, %[sum_0] \n"
                "vfmac.s %[dotp_0], ft4, ft0\n"
                "vfadd.s %[sum_1], ft0, %[sum_1] \n"
                "vfmac.s %[dotp_1], ft6, ft0\n"
                "j 0f\n"
                "1:\n"
                "vfsub.s ft4, ft1, %[current_mean]\n"
                "vfadd.s %[sum_0], ft0, %[sum_0] \n"
                "vfmac.s %[dotp_0], ft4, ft0\n"
                "0:\n"
                : [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
                  [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
                  [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64),
                  [mod_temp] "=r"(temp)
                : [current_mean] "fr"(current_mean.f64), [zero] "fr"(ZERO.f64),
                  [work_mod_3] "r"(work_mod_3), [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register double temp_sum, temp_dotp;
            asm volatile(
                "fld %[temp_sum], 0(%[sum_scratch])\n"
                "fld %[temp_dotp], 0(%[dotp_scratch])\n"
                "fld %[current_mean], 0(%[current_mean_scratch])\n"
                // 1+= 2
                "vfadd.s %[sum_1], %[sum_1], %[sum_2]\n"
                "vfadd.s %[dotp_1], %[dotp_1], %[dotp_2]\n"
                "vfsgnj.s %[sum_2],%[ZERO],%[ZERO]\n"
                "vfsgnj.s %[dotp_2],%[ZERO],%[ZERO]\n"
                // temp+=0
                "vfadd.s %[temp_sum], %[temp_sum], %[sum_0]\n"
                "vfadd.s %[temp_dotp], %[temp_dotp], %[dotp_0]\n"
                "vfsgnj.s %[sum_0],%[ZERO],%[ZERO]\n"
                "vfsgnj.s %[dotp_0],%[ZERO],%[ZERO]\n"
                // temp+=1
                "vfadd.s %[temp_sum], %[sum_1], %[temp_sum]\n"
                "vfadd.s %[temp_dotp], %[dotp_1], %[temp_dotp]\n"
                "vfsgnj.s %[sum_1],%[ZERO],%[ZERO]\n"
                "vfsgnj.s %[dotp_1],%[ZERO],%[ZERO]\n"
                "fsd %[temp_sum], 0(%[sum_scratch])\n"
                "fsd %[temp_dotp], 0(%[dotp_scratch])\n"
                : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
                  [sum_0] "+fr"(sum_0.f64), [sum_1] "+fr"(sum_1.f64),
                  [sum_2] "+fr"(sum_2.f64), [dotp_0] "+fr"(dotp_0.f64),
                  [dotp_1] "+fr"(dotp_1.f64), [dotp_2] "+fr"(dotp_2.f64),
                  [current_mean] "=&fr"(current_mean.f64)
                : [ZERO] "fr"(ZERO.f64),
                  [current_mean_scratch] "r"(current_mean_scratch),
                  [sum_scratch] "r"(sum_scratch),
                  [dotp_scratch] "r"(dotp_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_doubles_work_for_core_per_point);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        // __builtin_ssr_barrier(SNRT_SSR_DM1);
        work_mod_3 = next_work_mod_3;
        sum_scratch -=
            channel_stride * (num_doubles_work_for_core_per_point - 1);
        dotp_scratch -=
            channel_stride * (num_doubles_work_for_core_per_point - 1);
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_ssr_disable();
    snrt_ssr_repeat(SNRT_SSR_DM0, 1);
    snrt_cluster_hw_barrier();
    return buf_flag;
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_fp32_no_loop_2(
    const v2s* grad_ofmap_scratch, v2s* grad_ifmap_scratch,
    const v2s* ifmap_scratch, const v2s* weight_times_invstd_scratch,
    const v2s* k_scratch, const v2s* winvstd_times_meank_sub_dmean_scratch,
    uint32_t num_bytes_per_point, uint32_t num_points_work_for_core_in_tile,
    uint32_t work_mod_4, uint32_t work_div_4_sub_1,
    uint32_t num_doubles_to_process, uint32_t channel_stride) {
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core_in_tile,  // dimension of inner loop
        num_doubles_to_process,            // dimension of outer loop
        num_bytes_per_point,  // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_enable();
    uint32_t frep = num_points_work_for_core_in_tile >= 4;
    register volatile uint32_t i = 0;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
    do {
        register v2s k;
        k.f64 = k_scratch->f64;
        register v2s weight_times_invstd;
        weight_times_invstd.f64 = weight_times_invstd_scratch->f64;
        register v2s winvstd_times_meank_sub_dmean;
        winvstd_times_meank_sub_dmean.f64 =
            winvstd_times_meank_sub_dmean_scratch->f64;
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 16, 0, 0 \n"
                "vfmul.s ft3, ft0, %[k]\n"
                "vfmul.s ft4, ft0, %[k]\n"
                "vfmul.s ft5, ft0, %[k]\n"
                "vfmul.s ft6, ft0, %[k]\n"
                "vfsub.s ft3, ft2, ft3\n"
                "vfsub.s ft4, ft2, ft4\n"
                "vfsub.s ft5, ft2, ft5\n"
                "vfsub.s ft6, ft2, ft6\n"
                "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
                "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
                "vfmul.s ft5, ft5, %[weight_times_invstd] \n"
                "vfmul.s ft6, ft6, %[weight_times_invstd] \n"
                "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft5, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft6, %[winvstd_times_meank_sub_dmean] \n"
                :
                : [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean.f64),
                  [k] "fr"(k.f64), [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "addi %[i], %[i], 1\n"
            "beq %[num_doubles_to_process], %[i], 2f\n"  // shortcut when only
                                                         // 1 channel
            "add %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
            "add %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[channel_stride_in_bytes]\n"
            "2:\n"
            : [weight_times_invstd_scratch] "+r"(weight_times_invstd_scratch),
              [k_scratch] "+r"(k_scratch),
              [winvstd_times_meank_sub_dmean_scratch] "+r"(
                  winvstd_times_meank_sub_dmean_scratch),
              [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_doubles_to_process] "r"(num_doubles_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_4], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                    // then mod is 2
            "beqz %[mod_temp], 2f\n"                // jump to 2 if no
            "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                    // then mod is 1
            "beqz %[mod_temp], 1f\n"                // jump to 1 if no
            "3:\n"
            "vfmul.s ft3, ft0, %[k]\n"
            "vfmul.s ft4, ft0, %[k]\n"
            "vfmul.s ft5, ft0, %[k]\n"
            "vfsub.s ft3, ft2, ft3\n"
            "vfsub.s ft4, ft2, ft4\n"
            "vfsub.s ft5, ft2, ft5\n"
            "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
            "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
            "vfmul.s ft5, ft5, %[weight_times_invstd] \n"
            "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
            "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
            "vfadd.s ft1, ft5, %[winvstd_times_meank_sub_dmean] \n"
            "j 0f\n"
            "2:\n"
            "vfmul.s ft3, ft0, %[k]\n"
            "vfmul.s ft4, ft0, %[k]\n"
            "vfsub.s ft3, ft2, ft3\n"
            "vfsub.s ft4, ft2, ft4\n"
            "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
            "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
            "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
            "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
            "j 0f\n"
            "1:\n"
            "vfmul.s ft3, ft0, %[k]\n"
            "vfsub.s ft3, ft2, ft3\n"
            "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
            "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
            "0:\n"
            : [mod_temp] "=r"(mod_temp)
            : [weight_times_invstd] "fr"(weight_times_invstd.f64),
              [winvstd_times_meank_sub_dmean] "fr"(
                  winvstd_times_meank_sub_dmean.f64),
              [k] "fr"(k.f64), [work_mod_4] "r"(work_mod_4)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
    } while (i < num_doubles_to_process);
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp32_looped_2(
    const v2s* grad_ofmap_scratch, v2s* grad_ifmap_scratch,
    const v2s* ifmap_scratch, const v2s* weight_times_invstd_scratch,
    const v2s* k_scratch, const v2s* winvstd_times_meank_sub_dmean_scratch,
    uint32_t buf_flag, uint32_t num_doubles_per_aligned_point,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_4,    // precompute to avoid icache branch misses
    uint32_t work_div_4_sub_1, uint32_t tile_size_in_aligned_points,
    uint32_t num_doubles_work_for_core_per_point,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_4;
    register uint32_t frep = work_in_tile >= 4;
    register v2s ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_aligned_points *
                                     num_doubles_per_aligned_point *
                                     sizeof(double);
    const uint32_t input_channel_array_reset_dist =
        channel_stride * num_doubles_work_for_core_per_point * sizeof(double);
    const uint32_t inner_loop_stride =
        num_doubles_per_aligned_point * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,                         // dimension of inner loop
        num_doubles_work_for_core_per_point,  // dimension of outer loop
        inner_loop_stride,   // stride per inner loop iteration: 1 point
        outer_loop_stride);  // stride per outer loop iteration
    snrt_ssr_enable();

    if (buf_flag) {
        ifmap_scratch +=
            tile_size_in_aligned_points * num_doubles_per_aligned_point;
        grad_ifmap_scratch +=
            tile_size_in_aligned_points * num_doubles_per_aligned_point;
        grad_ofmap_scratch +=
            tile_size_in_aligned_points * num_doubles_per_aligned_point;
    }
    do {
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            // Can only manual unroll 3 times since the max for frep is 16
            register v2s k;
            k.f64 = k_scratch->f64;
            register v2s weight_times_invstd;
            weight_times_invstd.f64 = weight_times_invstd_scratch->f64;
            register v2s winvstd_times_meank_sub_dmean;
            winvstd_times_meank_sub_dmean.f64 =
                winvstd_times_meank_sub_dmean_scratch->f64;
            asm volatile(
                "frep.o %[n_frep], 16, 0, 0 \n"
                "vfmul.s ft3, ft0, %[k]\n"
                "vfmul.s ft4, ft0, %[k]\n"
                "vfmul.s ft5, ft0, %[k]\n"
                "vfmul.s ft6, ft0, %[k]\n"
                "vfsub.s ft3, ft2, ft3\n"
                "vfsub.s ft4, ft2, ft4\n"
                "vfsub.s ft5, ft2, ft5\n"
                "vfsub.s ft6, ft2, ft6\n"
                "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
                "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
                "vfmul.s ft5, ft5, %[weight_times_invstd] \n"
                "vfmul.s ft6, ft6, %[weight_times_invstd] \n"
                "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft5, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft6, %[winvstd_times_meank_sub_dmean] \n"
                :
                : [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean.f64),
                  [k] "fr"(k.f64), [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "addi %[i], %[i], 1\n"
                "add %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
                "add %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[channel_stride_in_bytes]\n"
                : [weight_times_invstd_scratch] "+r"(
                      weight_times_invstd_scratch),
                  [k_scratch] "+r"(k_scratch),
                  [winvstd_times_meank_sub_dmean_scratch] "+r"(
                      winvstd_times_meank_sub_dmean_scratch),
                  [i] "+r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_doubles_work_for_core_per_point], 2f\n"
                // extra check here for channels == 1. THen don't sub
                "sub %[weight_times_invstd_scratch], %[weight_times_invstd_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[k_scratch], %[k_scratch], %[input_channel_array_reset_dist]\n"
                "sub %[winvstd_times_meank_sub_dmean_scratch], %[winvstd_times_meank_sub_dmean_scratch], %[input_channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_4], %[work_mod_4_offset](%[dm_comm])\n"
                "lw %[work_div_4_sub_1], %[work_div_4_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 4\n"  // cmp frep < 4, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag),
                  [weight_times_invstd_scratch] "+r"(
                      weight_times_invstd_scratch),
                  [k_scratch] "+r"(k_scratch),
                  [winvstd_times_meank_sub_dmean_scratch] "+r"(
                      winvstd_times_meank_sub_dmean_scratch),
                  [work_in_tile] "+r"(work_in_tile),
                  [next_work_mod_4] "=r"(next_work_mod_4),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_4_sub_1] "+r"(work_div_4_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [grad_ifmap_scratch] "+r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_doubles_work_for_core_per_point] "r"(
                      num_doubles_work_for_core_per_point),
                  [input_channel_array_reset_dist] "r"(
                      input_channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_4_offset] "i"(offsetof(dm_comm_t, work_mod_4)),
                  [work_div_4_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_4_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            asm volatile(
                "beqz %[work_mod_4], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no,
                                                        // then mod is 2
                "beqz %[mod_temp], 2f\n"                // jump to 2 if no
                "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no,
                                                        // then mod is 1
                "beqz %[mod_temp], 1f\n"                // jump to 1 if no
                "3:\n"
                "vfmul.s ft3, ft0, %[k]\n"
                "vfmul.s ft4, ft0, %[k]\n"
                "vfmul.s ft5, ft0, %[k]\n"
                "vfsub.s ft3, ft2, ft3\n"
                "vfsub.s ft4, ft2, ft4\n"
                "vfsub.s ft5, ft2, ft5\n"
                "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
                "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
                "vfmul.s ft5, ft5, %[weight_times_invstd] \n"
                "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft5, %[winvstd_times_meank_sub_dmean] \n"
                "j 0f\n"
                "2:\n"
                "vfmul.s ft3, ft0, %[k]\n"
                "vfmul.s ft4, ft0, %[k]\n"
                "vfsub.s ft3, ft2, ft3\n"
                "vfsub.s ft4, ft2, ft4\n"
                "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
                "vfmul.s ft4, ft4, %[weight_times_invstd] \n"
                "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
                "vfadd.s ft1, ft4, %[winvstd_times_meank_sub_dmean] \n"
                "j 0f\n"
                "1:\n"
                "vfmul.s ft3, ft0, %[k]\n"
                "vfsub.s ft3, ft2, ft3\n"
                "vfmul.s ft3, ft3, %[weight_times_invstd] \n"
                "vfadd.s ft1, ft3, %[winvstd_times_meank_sub_dmean] \n"
                "0:\n"
                : [mod_temp] "=r"(temp)
                : [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [winvstd_times_meank_sub_dmean] "fr"(
                      winvstd_times_meank_sub_dmean.f64),
                  [k] "fr"(k.f64), [work_mod_4] "r"(work_mod_4)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
        } while (i < num_doubles_work_for_core_per_point);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        work_mod_4 = next_work_mod_4;
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}
