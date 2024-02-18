// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdbool.h>

#include "dnn.h"

#include "batchnorm_data_structures.h"
#include "batchnorm_reference.h"
#include "batchnorm_utils.h"
#include "dnn.h"
#include "printf.h"
#include "snrt.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)

static inline void batchnorm_forward_multicore_fp64(batchnorm_layer_t *l) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    double *gamma_scratch = ptr;
    double *running_var_scratch = gamma_scratch;
    ptr += C;
    double *beta_scratch = ptr;
    double *running_mean_scratch = beta_scratch;
    ptr += C;
    double *weight_scratch = ptr;
    ptr += C;
    double *bias_scratch = ptr;
    ptr += C;

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left =
        used_tcdm_end_addr - ptr - 32;  // 64 for shifting buffer
    // first 2: ofmap (overliad ish with ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t ofmap_len = tile_size_in_points * C * 2, ifmap_len = ofmap_len;
    // want to ensure tile stride (in doubles)
    ptrdiff_t tile_stride_in_doubles = tile_size_in_points * C;

    double *ofmap_scratch = ptr;
    ptr += tile_stride_in_doubles * 2;
    int ofmap_bank_alignment = ((uintptr_t)ofmap_scratch & (uintptr_t)0xFF) >>
                               3;  // 32 banks of 8 byte width
    // want to ensure ifmap_scratch is on a different 64 byte alignment than
    // ofmapscratch.
    double *ifmap_scratch = ptr;
    int ifmap_bank_alignment =
        ((uintptr_t)ifmap_scratch & (uintptr_t)0xFF) >> 3;
    // worst case: difference is 17. Then we need to add 31
    // difference is
    int offset_by =
        positive_modulo(8 - (ifmap_bank_alignment - ofmap_bank_alignment), 32);
    ifmap_scratch += offset_by;

    bool buf_flag = 0;

    // throughput below 4KB=512 doubles is too high.
    uint32_t doubles_loadable = 512;
    uint32_t points_loadable = ceildiv(doubles_loadable, C);
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_points), num_points);

    uint32_t work_left = num_points;
    uint32_t work_sub_1 = work_in_tile - 1;
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_1 = 0;
        dm_comm->work_div_1_sub_1 = work_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    snrt_dma_txid_t load_running_mean, load_running_var, load_weight, load_bias;
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        load_running_mean = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, C * sizeof(double));
        load_running_var = snrt_dma_start_1d(
            running_var_scratch, l->running_var, C * sizeof(double));
        load_weight =
            snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        load_bias =
            snrt_dma_start_1d(bias_scratch, l->bias, C * sizeof(double));
        snrt_dma_wait_all();
        snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                          work_in_tile * C * sizeof(double));
        buf_flag = !buf_flag;
    } else {
        // first ssr: read from running_var, then running_mean
        // second ssr: write to gamma, then to beta
        // third ssr: read from weight, then bias
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, num_channels_work_for_core,
                         C * sizeof(double),
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    // this is necessary to ensure that dm_comm has been written to
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &running_var_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &gamma_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D,
                          &weight_scratch[compute_id]);
            register double eps = l->eps;
            snrt_ssr_enable();
            // Reduce the four variables running_mean, running_var, weight, bias
            // into an fmadd for main loop see
            // https://github.com/pytorch/pytorch/blob/3acaf8564da4c2f0faaea33ce4572ad9b3715b47/aten/src/ATen/native/cpu/batch_norm_kernel.cpp#L45-L55
            asm volatile(
                "frep.o %[n_frep], 5, 0, 0 \n"
                "fadd.d ft3, ft0, %[eps]\n"
                "fsqrt.d ft3, ft3\n"
                // gamma = weight / sqrt(var + eps)
                "fdiv.d ft3, ft2, ft3\n"
                "fsgnj.d ft1, ft3, ft3\n"
                // beta = bias - mean*gamma
                "fnmsub.d ft1, ft0, ft3, ft2\n"
                :
                : [eps] "fr"(eps),
                  [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3");

            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (work_in_tile == num_points) {
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait_all();

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->ofmap, ofmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            if (num_channels_work_for_core > 0) {
                batchnorm_forward_fp64_no_loop(
                    &ifmap_scratch[compute_id], &ofmap_scratch[compute_id],
                    &gamma_scratch[compute_id], &beta_scratch[compute_id],
                    C * sizeof(double), work_in_tile, work_sub_1,
                    num_channels_work_for_core, num_compute_cores);
            } else {
                snrt_cluster_hw_barrier();
            }
            snrt_cluster_hw_barrier();
        }

    } else {
        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_forward_dma_main_loop_fp_agnostic(
                l->ifmap, l->ofmap, C, C * sizeof(double), C * sizeof(double),
                true, work_left, work_in_tile, dm_comm, 1, tile_size_in_points,
                tile_stride_in_doubles, ifmap_scratch, ofmap_scratch, buf_flag);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                while (work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_forward_tile_fp64_looped(
                    &ifmap_scratch[compute_id], &ofmap_scratch[compute_id],
                    &gamma_scratch[compute_id], &beta_scratch[compute_id],
                    C * sizeof(double), work_in_tile, work_sub_1,
                    tile_stride_in_doubles, num_channels_work_for_core,
                    num_compute_cores, dm_comm);
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();

    // wait for all transactions to complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

static inline void batchnorm_forward_training_multicore_fp64(
    batchnorm_training_layer_t *l, double *temp) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;
    const double momentum = l->momentum;
    const double one_sub_momentum = 1 - momentum;
    const double momentum_times_unbiased_correction =
        momentum * ((double)num_points / (double)(num_points - 1));

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    // want to compute
    // running_var = running_var * (momentum) + current_var * (1-momentum)
    // running_mean = running_mean * (momentum) + current_mean * (1-momentum)
    // gamma = weight / sqrt(current_var + eps)
    // beta = bias - running_mean * gamma
    double *current_var_scratch = ptr;
    ptr += C;
    double *current_mean_scratch = ptr;
    ptr += C;
    double *running_var_scratch = ptr;
    ptr += C;
    double *running_mean_scratch = ptr;
    ptr += C;
    double *gamma_scratch = ptr;
    double *weight_scratch = ptr;
    ptr += C;
    double *beta_scratch = ptr;
    double *bias_scratch = ptr;
    ptr += C;

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;  // 64 for shifting buffer
    // first 2: ofmap (overlaid ish with ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * C);

    ptrdiff_t ofmap_len = tile_size_in_points * C * 2, ifmap_len = ofmap_len;
    // want to ensure tile stride (in doubles)
    ptrdiff_t tile_stride_in_doubles = tile_size_in_points * C;

    double *ofmap_scratch = ptr;
    double *ifmap_scratch = ofmap_scratch;

    bool buf_flag = 0;

    // throughput below 4KB=512 doubles is too high.
    uint32_t doubles_loadable = 512;
    uint32_t points_loadable = ceildiv(doubles_loadable, C);
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_points), num_points);

    uint32_t work_left = num_points;
    uint32_t work_div_4_sub_1 = work_in_tile / 4 - 1;
    uint32_t work_mod_4 = work_in_tile % 4;
    uint32_t work_sub_1 = work_in_tile - 1;

    // uint32_t work_sub_1 = work_in_tile - 1;
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        // TODO: figure out what mod to do
        dm_comm->work_mod_4 = work_mod_4;
        dm_comm->work_div_4_sub_1 = work_div_4_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    snrt_dma_txid_t load_running_mean, load_running_var, load_weight, load_bias,
        load_ifmap;
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        load_ifmap = snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                                       work_in_tile * C * sizeof(double));
        buf_flag = !buf_flag;

        load_running_mean = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, C * sizeof(double));
        load_running_var = snrt_dma_start_1d(
            running_var_scratch, l->running_var, C * sizeof(double));
        load_weight =
            snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        load_bias =
            snrt_dma_start_1d(bias_scratch, l->bias, C * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    // ensure that dm comm is written to
    snrt_cluster_hw_barrier();
    uint32_t start_statistic = SNRT_SECTIONED_MCYCLE();
    if (work_in_tile == num_points) {
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait(load_ifmap);

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
        } else {
            if (num_channels_work_for_core > 0) {
                batchnorm_collect_statistics_fp64_no_loop(
                    &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &current_var_scratch[compute_id], num_points,
                    C * sizeof(double), work_in_tile, work_div_4_sub_1,
                    work_mod_4, num_channels_work_for_core, num_compute_cores);
            } else {
                // wait for dma to be ready
                snrt_cluster_hw_barrier();
            }
            // notify compute done
            snrt_cluster_hw_barrier();
        }
    } else {
        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_collect_statistics_dma_main_loop_fp_agnostic(
                l, C, C * sizeof(double), C * sizeof(double), true, work_left,
                work_in_tile, dm_comm, 4, tile_size_in_points,
                tile_stride_in_doubles, ifmap_scratch, buf_flag);

            // reset buf flag and reload
            buf_flag = 0;
            dm_comm->num_points_work_in_tile = work_in_tile;
            // TODO: figure out what mod to do
            dm_comm->work_mod_4 = work_mod_4;
            dm_comm->work_div_4_sub_1 =
                work_div_4_sub_1;  // this is the frep value
            snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                              work_in_tile * C * sizeof(double));
            buf_flag = !buf_flag;
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                uint32_t dm_comm_work_in_tile = work_in_tile;
                while (dm_comm_work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    dm_comm_work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_collect_mean_statistics_tile_fp64_looped(
                    &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id], num_points,
                    C * sizeof(double), work_in_tile, work_div_4_sub_1,
                    work_mod_4, tile_stride_in_doubles,
                    num_channels_work_for_core, num_compute_cores, dm_comm);
            }
        }

        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_collect_statistics_dma_main_loop_fp_agnostic(
                l, C, C * sizeof(double), C * sizeof(double), true, work_left,
                work_in_tile, dm_comm, 4, tile_size_in_points,
                tile_stride_in_doubles, ifmap_scratch, buf_flag);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                uint32_t dm_comm_work_in_tile = work_in_tile;
                while (dm_comm_work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    dm_comm_work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_collect_var_statistics_tile_fp64_looped(
                    &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &current_var_scratch[compute_id], num_points,
                    C * sizeof(double), work_in_tile, work_div_4_sub_1,
                    work_mod_4, tile_stride_in_doubles,
                    num_channels_work_for_core, num_compute_cores, dm_comm);
            }
        }
    }

    // need to collect current_mean = sum(x) / N
    // need to collect current_var = sum((x-current_mean)^2) / N
    uint32_t end_statistic = SNRT_SECTIONED_MCYCLE();

    uint32_t start_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
        // notify that the rest has been loaded in
        snrt_cluster_hw_barrier();
        buf_flag = 0;
        dm_comm->num_points_work_in_tile = work_in_tile;
        // TODO: figure out what mod to do
        dm_comm->work_mod_1 = 0;
        dm_comm->work_div_1_sub_1 = work_sub_1;  // this is the frep value
        snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                          work_in_tile * C * sizeof(double));
        buf_flag = !buf_flag;
    } else {
        snrt_cluster_hw_barrier();
        if (num_channels_work_for_core > 0) {
            snrt_ssr_loop_2d(SNRT_SSR_DM1, 4, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM2, 4, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_repeat(SNRT_SSR_DM0, 2);
            // consume current var (twice) then current mean (twice)
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &current_var_scratch[compute_id]);
            // write to running_var, running_mean, gamma, beta
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &running_var_scratch[compute_id]);
            // consume running_var, running_mean, weight, bias
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D,
                          &running_var_scratch[compute_id]);
            register double eps = l->eps;
            snrt_ssr_enable();
            // Reduce the four variables running_mean, running_var, weight, bias
            // into an fmadd for main loop see
            // https://github.com/pytorch/pytorch/blob/3acaf8564da4c2f0faaea33ce4572ad9b3715b47/aten/src/ATen/native/cpu/batch_norm_kernel.cpp#L45-L55
            // Moreover, update statistics for running_mean and running_var.
            // Use unbiased estimator for updating statistic, but biased
            // estimator for gamma/beta
            asm volatile(
                "frep.o %[n_frep], 9, 0, 0 \n"
                // current_var + eps
                "fadd.d ft3, ft0, %[eps]\n"
                // current_var * momentum
                "fmul.d ft4, ft0, %[momentum_times_unbiased_correction]\n"
                // current_mean * momentum
                "fmul.d ft5, ft0, %[momentum]\n"
                // sqrt(current_var + eps)
                "fsqrt.d ft3, ft3\n"
                // running_var update
                "fmadd.d ft1, ft2, %[one_sub_momentum], ft4\n"
                // running_mean update
                "fmadd.d ft1, ft2, %[one_sub_momentum], ft5\n"
                // weight / sqrt()
                "fdiv.d ft3, ft2, ft3\n"
                // write gamma (weight already consumed)
                "fsgnj.d ft1, ft3, ft3\n"
                // consume bias, current_mean and write beta
                "fnmsub.d ft1, ft0, ft3, ft2\n"
                :
                : [eps] "fr"(eps), [one_sub_momentum] "fr"(one_sub_momentum),
                  [momentum] "fr"(momentum),
                  [momentum_times_unbiased_correction] "fr"(
                      momentum_times_unbiased_correction),
                  [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");

            snrt_fpu_fence();
            snrt_ssr_repeat(SNRT_SSR_DM0, 1);
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    // calc gamma beta, dma loads in first tile for ofmap
    uint32_t end_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_ofmap_calc = SNRT_SECTIONED_MCYCLE();

    if (work_in_tile == num_points) {
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait_all();

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->ofmap, ofmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            if (num_channels_work_for_core > 0) {
                batchnorm_forward_fp64_no_loop(
                    &ifmap_scratch[compute_id], &ofmap_scratch[compute_id],
                    &gamma_scratch[compute_id], &beta_scratch[compute_id],
                    C * sizeof(double), work_in_tile, work_sub_1,
                    num_channels_work_for_core, num_compute_cores);
            } else {
                snrt_cluster_hw_barrier();
            }
            snrt_cluster_hw_barrier();
        }

    } else {
        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_forward_dma_main_loop_fp_agnostic(
                l->ifmap, l->ofmap, C, C * sizeof(double), C * sizeof(double),
                true, work_left, work_in_tile, dm_comm, 1, tile_size_in_points,
                tile_stride_in_doubles, ifmap_scratch, ofmap_scratch, buf_flag);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                uint32_t dm_comm_work_in_tile = work_in_tile;
                while (dm_comm_work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    dm_comm_work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_forward_tile_fp64_looped(
                    &ifmap_scratch[compute_id], &ofmap_scratch[compute_id],
                    &gamma_scratch[compute_id], &beta_scratch[compute_id],
                    C * sizeof(double), work_in_tile, work_sub_1,
                    tile_stride_in_doubles, num_channels_work_for_core,
                    num_compute_cores, dm_comm);
            }
        }
    }
    uint32_t end_ofmap_calc = SNRT_SECTIONED_MCYCLE();

    // collect stats

    // update running mean and running var
    // pass << batch mean and batch var >> as parameters to compute beta/gamma
    // call batchnorm_layer

    // wait for all transactions to complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->running_var, running_var_scratch,
                          C * sizeof(double));
        snrt_dma_start_1d(l->running_mean, running_mean_scratch,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

static inline void batchnorm_backward_multicore_fp64(
    batchnorm_backward_layer_t *l) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    double *invstd_scratch = ptr;
    ptr += C;
    double *weight_scratch = ptr;
    ptr += C;
    double *running_mean_scratch = ptr;
    ptr += C;
    double *grad_bias_scratch = ptr;
    ptr += C;
    double *grad_weight_scratch = ptr;
    ptr += C;

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    // first 2: ofmap, ifmap (overlaid with grad_ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t grad_ofmap_len = tile_size_in_points * C * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    // Incrementally increase tile size.
    // Reason is because we want to minimize wait on the first iteration
    // how much time do i have? C/num_compute_cores * 40-50, approximately.
    // estimate 7 doubles per cycle
    // fixed cost below 1KB=128 doubles is too high.
    uint32_t doubles_loadable =
        max(ceildiv(C, num_compute_cores) * 50 * 7, 128);
    uint32_t points_loadable = doubles_loadable / C;
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_points), num_points);
    // num_points = num_points/2;
    uint32_t work_left = num_points;
    uint32_t work_mod_2 = work_in_tile % 2;
    uint32_t work_div_2_sub_1 = work_in_tile / 2 - 1;
    DUMP(work_in_tile);
    DUMP(tile_size_in_points);
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_2 = work_mod_2;
        dm_comm->work_div_2_sub_1 = work_div_2_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        // Initiate loads for everything but only wait for the running var load.
        // Q: is it better to wait then initiate the rest? we'll see
        snrt_dma_start_1d(invstd_scratch, l->running_var, C * sizeof(double));
        snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        snrt_dma_start_1d(running_mean_scratch, l->running_mean,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 3, num_channels_work_for_core,
                         C * sizeof(double),
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // load first tile in but only as much as we can in parallel while sqrt
        // runs
        snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                          work_in_tile * C * sizeof(double));
        snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                          work_in_tile * C * sizeof(double));

        buf_flag = !buf_flag;
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &invstd_scratch[compute_id]);
            register double eps = l->eps;  // any value in dma'ing this? idk
            const register double ONE = 1;
            snrt_ssr_enable();
            // unrolling does not help because fsqrt and fdiv are not pipelined
            asm volatile(
                "frep.o %[n_frep], 6, 0, 0 \n"
                "fadd.d ft3, ft0, %[eps]\n"
                "fsqrt.d ft3, ft3\n"
                "fdiv.d ft3, %[ONE], ft3\n"
                // write back invstd
                "fsgnj.d ft1, ft3, ft3\n"
                // write back weight * invstd
                "fmul.d ft1, ft0, ft3\n"
                // write back running mean * invstd
                "fmul.d ft1, ft0, ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();

    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    // compute grad_weight, grad_bias, grad_ifmap. Tile only if we can't fit all
    // the points in one tile.
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (work_in_tile == num_points) {
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait_all();

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            snrt_cluster_hw_barrier();
            if (num_channels_work_for_core > 0) {
                batchnorm_backward_fp64_no_loop(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &running_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id],
                    &grad_bias_scratch[compute_id],
                    &grad_weight_scratch[compute_id], C, work_in_tile,
                    work_mod_2, num_channels_work_for_core, num_compute_cores);
            }

            snrt_cluster_hw_barrier();
        }

    } else {
        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_backward_dma_main_loop_fp_agnostic(
                l, C, C * sizeof(double), C * sizeof(double), true, work_left,
                work_in_tile, dm_comm, tile_size_in_points, grad_ofmap_scratch,
                ifmap_scratch, grad_ifmap_scratch, buf_flag);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                while (work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_backward_tile_fp64_looped(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &running_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id],
                    &grad_bias_scratch[compute_id],
                    &grad_weight_scratch[compute_id], C, work_in_tile,
                    work_mod_2, work_div_2_sub_1, tile_size_in_points,
                    num_channels_work_for_core, num_compute_cores, dm_comm);
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();

    uint32_t start_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT2);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT3);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT4);
}

static inline void batchnorm_backward_multicore_fp32(
    batchnorm_backward_layer_t *l) {
    uint32_t start = snrt_mcycle();

    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;

    precision_t dtype_bytes = l->dtype;
    uint32_t num_dtypes_per_double = (FP64 / dtype_bytes);

    // including padding for non-aligned points.
    // Write input with padded stride since ssr only supports 64-bit width r/w
    uint32_t num_doubles_per_aligned_point = ceildiv(C, num_dtypes_per_double);
    uint32_t num_doubles = num_points * num_doubles_per_aligned_point;
    bool is_point_aligned_to_8_byte_boundary = C % num_dtypes_per_double == 0;
    uint32_t num_bytes_per_aligned_point =
        num_doubles_per_aligned_point * sizeof(double);
    uint32_t num_bytes_in_point_aligned_ifmap = num_doubles * sizeof(double);

    // not including padding for non-aligned points.
    // When is_point_aligned_to_8_byte_boundary == true, the expressions above
    // are equivalent. Output should be written according to the packed version
    uint32_t num_bytes_per_packed_point = C * dtype_bytes;

    const uint32_t num_doubles_work_for_core_per_aligned_point =
        get_core_num_work_items(num_doubles_per_aligned_point,
                                num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    v2s *ptr = (v2s *)raw_ptr;
    v2s *invstd_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *weight_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *running_mean_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *grad_bias_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *grad_weight_scratch = ptr;
    ptr += num_doubles_per_aligned_point;

    // Dynamically compute tile sizes
    v2s *used_tcdm_end_addr =
        (v2s *)(snrt_l1_end_addr() -
                ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                    (snrt_cluster_core_num() + 1));  // + 1 for safety
    DUMP(used_tcdm_end_addr);
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    // first 2: ofmap, ifmap (overlaid with grad_ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_aligned_points =
        (space_left) / (2 * 2 * num_doubles_per_aligned_point);

    ptrdiff_t grad_ofmap_len =
        tile_size_in_aligned_points * num_doubles_per_aligned_point * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    v2s *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    v2s *ifmap_scratch = ptr;
    ptr += ifmap_len;
    v2s *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    // Incrementally increase tile size.
    // Reason is because we want to minimize wait on the first iteration
    // how much time do i have? C/num_compute_cores * 40-50, approximately.
    // estimate 7 doubles per cycle
    // fixed cost below 1KB=128 doubles is too high.
    uint32_t doubles_loadable =
        max(ceildiv(num_doubles_per_aligned_point, num_compute_cores) * 50 * 7,
            128);
    uint32_t points_loadable = doubles_loadable / num_doubles_per_aligned_point;
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_aligned_points), num_points);
    // num_points = num_points/2;
    uint32_t work_left = num_points;
    uint32_t work_mod_2 = work_in_tile % 2;
    uint32_t work_div_2_sub_1 = work_in_tile / 2 - 1;
    DUMP(work_in_tile);
    DUMP(tile_size_in_aligned_points);
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_2 = work_mod_2;
        dm_comm->work_div_2_sub_1 = work_div_2_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        // Initiate loads for everything but only wait for the running var load.
        // Q: is it better to wait then initiate the rest? we'll see
        running_var_load = snrt_dma_start_1d(invstd_scratch, l->running_var,
                                             num_bytes_per_packed_point);
        weight_load = snrt_dma_start_1d(weight_scratch, l->weight,
                                        num_bytes_per_packed_point);
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, num_bytes_per_packed_point);
        snrt_dma_wait_all();
    } else {
        // PRECONFIGURE: operations on arrays of size C, split by core.

        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL, 2, num_doubles_work_for_core_per_aligned_point,
            num_bytes_per_aligned_point, num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // load first tile in but only as much as we can in parallel while sqrt
        // runs
        grad_ofmap_load = initiate_dma_1d_or_2d(
            grad_ofmap_scratch, l->grad_ofmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point,
            work_in_tile, is_point_aligned_to_8_byte_boundary);
        ifmap_load = initiate_dma_1d_or_2d(
            ifmap_scratch, l->ifmap, dtype_bytes * C,
            num_bytes_per_aligned_point, num_bytes_per_packed_point,
            work_in_tile, is_point_aligned_to_8_byte_boundary);

        buf_flag = !buf_flag;
    } else {
        if (num_doubles_work_for_core_per_aligned_point > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &invstd_scratch[compute_id]);
            register float eps = l->eps;  // any value in dma'ing this? idk
            const register float ONE = 1;
            snrt_ssr_enable();
            // unrolling does not help because fsqrt and fdiv are not pipelined
            asm volatile(
                "vfcpka.s.s %[ONE],%[ONE],%[ONE]\n"  // duplicate the 1
                "frep.o %[n_frep], 5, 0, 0 \n"
                "vfadd.r.s ft3, ft0, %[eps]\n"
                "vfsqrt.s ft3, ft3\n"
                "vfdiv.s ft3, %[ONE], ft3\n"
                // write back invstd
                "vfsgnj.s ft1, ft3, ft3\n"
                // write back weight * invstd
                "vfmul.s ft1, ft0, ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_doubles_work_for_core_per_aligned_point -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();                     // thought: do we need this?
            __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
            snrt_ssr_disable();
        }
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();

    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    // compute grad_weight, grad_bias, grad_ifmap. Tile only if we can't fit all
    // the points in one tile.
    if (work_in_tile == num_points) {
        uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait_all();

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
            initiate_dma_1d_or_2d(
                (void *)l->grad_ifmap, (void *)grad_ifmap_scratch,
                num_bytes_per_packed_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point, num_points,
                is_point_aligned_to_8_byte_boundary);
        } else {
            snrt_cluster_hw_barrier();
            if (num_doubles_work_for_core_per_aligned_point > 0) {
                batchnorm_backward_fp32_no_loop(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &running_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id],
                    &grad_bias_scratch[compute_id],
                    &grad_weight_scratch[compute_id],
                    num_bytes_per_aligned_point, work_in_tile, work_mod_2,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores);
            }

            snrt_cluster_hw_barrier();
        }

        uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    } else {
        if (snrt_is_dm_core()) {
            // buf flag should be 1 at this point
            batchnorm_backward_dma_main_loop_fp_agnostic(
                l, num_doubles_per_aligned_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point,
                is_point_aligned_to_8_byte_boundary, work_left, work_in_tile,
                dm_comm, tile_size_in_aligned_points, grad_ofmap_scratch,
                ifmap_scratch, grad_ifmap_scratch, buf_flag);
        } else {
            if (num_doubles_work_for_core_per_aligned_point == 0) {
                // start up first tile
                snrt_cluster_hw_barrier();
                while (work_in_tile != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_backward_tile_fp32_looped(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &running_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id],
                    &grad_bias_scratch[compute_id],
                    &grad_weight_scratch[compute_id],
                    num_doubles_per_aligned_point, work_in_tile, work_mod_2,
                    work_div_2_sub_1, tile_size_in_aligned_points,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores, dm_comm);
            }
        }
    }

    uint32_t start_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch,
                          num_bytes_per_packed_point);
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          num_bytes_per_packed_point);
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT2);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT3);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT4);
}

static inline void batchnorm_backward_training_multicore_fp64(
    batchnorm_backward_training_layer_t *l) {
    uint32_t start = snrt_mcycle();

    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id = snrt_cluster_core_idx();

    // Calculate output dimensions
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    const uint32_t num_points = N * H * W;
    const double num_points_inv = 1 / ((double)num_points);

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    double *dotp_scratch = ptr;
    ptr += C;
    double *grad_bias_scratch = ptr;
    ptr += C;
    double *current_mean_scratch = ptr;
    ptr += C;
    double *invstd_scratch = ptr;
    ptr += C;
    double *weight_times_invstd_scratch = ptr;
    ptr += C;
    double *grad_weight_scratch = ptr;
    ptr += C;
    double *k_scratch = ptr;
    ptr += C;
    double *winvstd_times_meank_sub_dmean_scratch = ptr;
    ptr += C;

    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t grad_ofmap_len = tile_size_in_points * C * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t invstd_load, current_var_load, weight_load,
        current_mean_load, grad_ofmap_load, ifmap_load;

    uint32_t doubles_loadable =
        max(ceildiv(C, num_compute_cores) * 50 * 7, 128);
    uint32_t points_loadable = doubles_loadable / C;
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_points), num_points);
    // work_in_tile = 4;
    bool is_tiling_enabled = work_in_tile != num_points;
    uint32_t work_left = num_points;
    uint32_t work_mod_3 = work_in_tile % 3;
    uint32_t work_div_3_sub_1 = work_in_tile / 3 - 1;
    uint32_t work_mod_4 = work_in_tile % 4;
    uint32_t work_div_4_sub_1 = work_in_tile / 4 - 1;
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_3 = work_mod_3;
        dm_comm->work_div_3_sub_1 = work_div_3_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        current_var_load = snrt_dma_start_1d(invstd_scratch, l->current_var,
                                             C * sizeof(double));
        weight_load = snrt_dma_start_1d(weight_times_invstd_scratch, l->weight,
                                        C * sizeof(double));
        snrt_dma_wait(current_var_load);
        snrt_dma_wait(weight_load);
        buf_flag = !buf_flag;
    } else {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, num_channels_work_for_core,
                         C * sizeof(double),
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        grad_ofmap_load = snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                                            work_in_tile * C * sizeof(double));
        ifmap_load = snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                                       work_in_tile * C * sizeof(double));
        current_mean_load = snrt_dma_start_1d(
            current_mean_scratch, l->current_mean, C * sizeof(double));
        snrt_dma_wait(grad_ofmap_load);
        snrt_dma_wait(ifmap_load);
        snrt_dma_wait(current_mean_load);
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &invstd_scratch[compute_id]);
            register double eps = l->eps;
            const register double ONE = 1;
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 5, 0, 0 \n"
                "fadd.d ft3, ft0, %[eps]\n"
                "fsqrt.d ft3, ft3\n"
                "fdiv.d ft3, %[ONE], ft3\n"
                // write out invstd
                "fsgnj.d ft1, ft3, ft3\n"
                // write out invstd * weight
                "fmul.d ft1, ft0, ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_channels_work_for_core - 1)
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    if (!is_tiling_enabled) {
        uint32_t start_main_loop_1 = SNRT_SECTIONED_MCYCLE();
        if (snrt_is_dm_core()) {
        } else {
            if (num_channels_work_for_core > 0) {
                batchnorm_backward_training_fp64_no_loop_1(
                    &grad_ofmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &grad_bias_scratch[compute_id], &dotp_scratch[compute_id],
                    C, work_in_tile, work_mod_3, work_div_3_sub_1,
                    num_channels_work_for_core, num_compute_cores);
            }
        }
        uint32_t end_main_loop_1 = SNRT_SECTIONED_MCYCLE();
    } else {
        if (snrt_is_dm_core()) {
            buf_flag = batchnorm_backward_training_dma_main_loop_fp_agnostic(
                l, C, C * sizeof(double), C * sizeof(double), true, num_points,
                work_left, work_in_tile, dm_comm, tile_size_in_points,
                grad_ofmap_scratch, ifmap_scratch, NULL, buf_flag, 3, 3);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                uint32_t work_in_tile_temp = work_in_tile;
                snrt_cluster_hw_barrier();
                while (work_in_tile_temp != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile_temp = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                buf_flag = batchnorm_backward_training_tile_fp64_looped_1(
                    &grad_ofmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &grad_bias_scratch[compute_id], &dotp_scratch[compute_id],
                    C, work_in_tile, work_mod_3, work_div_3_sub_1,
                    tile_size_in_points, num_channels_work_for_core,
                    num_compute_cores, dm_comm);
            }
        }

        work_in_tile = min(tile_size_in_points, num_points);
        work_mod_4 = work_in_tile % 4;
        work_div_4_sub_1 = work_in_tile / 4 - 1;
    }

    uint32_t start_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();

    uint32_t start_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        work_left = num_points - work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_4 = work_mod_4;
        dm_comm->work_div_4_sub_1 = work_div_4_sub_1;
    } else if (snrt_is_compute_core()) {
        if (num_channels_work_for_core > 0) {
            // need to compute
            // weight * invstd - can be done in pre-comp
            // grad_weight <- invstd*dotp
            // k <- grad_weight * invstd / num_points
            // (temp) grad_mean <- grad_bias / num_points
            // (mean * k - grad_mean) * (weight*invstd)
            //    = (mean * k)*weight*invstd-grad_mean * (weight*invstd)

            snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_repeat(SNRT_SSR_DM0, 2);
            snrt_ssr_loop_2d(SNRT_SSR_DM1, 3, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM2, 3, num_channels_work_for_core,
                             C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &grad_weight_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &dotp_scratch[compute_id]);
            snrt_ssr_enable();

            asm volatile(
                "frep.o %[n_frep], 9, 0, 0 \n"
                // grad_weight = invstd*dotp
                "fmul.d ft3, ft0, ft2\n"
                // grad_mean = grad_bias / num_points
                "fmul.d ft4, ft2, %[num_points_inv]\n"
                // ft5 = invstd / num_points
                "fmul.d ft5, ft0, %[num_points_inv]\n"
                // write out grad_weight
                "fsgnj.d ft1, ft3, ft3\n"
                // k = grad_weight * invstd / num_points
                "fmul.d ft6, ft3, ft5\n"
                // ft7 = grad_mean * (weight*invstd)
                "fmul.d ft7, ft4, ft0\n"
                // write out k
                "fsgnj.d ft1, ft6, ft6\n"
                // ft8 = mean * k
                "fmul.d ft8, ft2, ft6\n"
                // write out (mean*k)*(weight*invstd) -
                // (grad_mean)*(weight*invstd)
                "fmsub.d ft1, ft8, ft0, ft7\n"
                :
                : [n_frep] "r"(num_channels_work_for_core - 1),
                  [num_points_inv] "fr"(num_points_inv)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                  "ft8");
            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
            snrt_ssr_repeat(SNRT_SSR_DM0, 1);
        }
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
    }
    uint32_t end_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    // DUMP(5);
    if (!is_tiling_enabled) {
        uint32_t start_main_loop_2 = SNRT_SECTIONED_MCYCLE();
        if (snrt_is_dm_core()) {
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            if (num_channels_work_for_core > 0) {
                batchnorm_backward_training_fp64_no_loop_2(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &weight_times_invstd_scratch[compute_id],
                    &k_scratch[compute_id],
                    &winvstd_times_meank_sub_dmean_scratch[compute_id], C,
                    work_in_tile, work_mod_4, work_div_4_sub_1,
                    num_channels_work_for_core, num_compute_cores);
            }
            // notify finish
            snrt_cluster_hw_barrier();
        }
        uint32_t end_main_loop_2 = SNRT_SECTIONED_MCYCLE();
    } else {
        if (snrt_is_dm_core()) {
            // DUMP(56);
            batchnorm_backward_training_dma_main_loop_fp_agnostic(
                l, C, C * sizeof(double), C * sizeof(double), true, num_points,
                work_left, work_in_tile, dm_comm, tile_size_in_points,
                grad_ofmap_scratch, ifmap_scratch, grad_ifmap_scratch, buf_flag,
                4, 2);
        } else {
            if (num_channels_work_for_core == 0) {
                // start up first tile
                uint32_t work_in_tile_temp = work_in_tile;
                snrt_cluster_hw_barrier();
                while (work_in_tile_temp != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile_temp = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_backward_training_tile_fp64_looped_2(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &weight_times_invstd_scratch[compute_id],
                    &k_scratch[compute_id],
                    &winvstd_times_meank_sub_dmean_scratch[compute_id],
                    buf_flag, C, work_in_tile, work_mod_4, work_div_4_sub_1,
                    tile_size_in_points, num_channels_work_for_core,
                    num_compute_cores, dm_comm);
            }
        }
    }
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
    if (work_in_tile == num_points && snrt_is_compute_core() && compute_id == 0)
        DUMP(0);
    if (work_in_tile != num_points && snrt_is_compute_core() && compute_id == 0)
        DUMP(1);
}

static inline void batchnorm_backward_training_multicore_fp32(
    batchnorm_backward_training_layer_t *l) {
    uint32_t start = snrt_mcycle();

    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id = snrt_cluster_core_idx();

    // Calculate output dimensions
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    const uint32_t num_points = N * H * W;
    const float num_points_inv = 1 / ((float)num_points);

    precision_t dtype_bytes = l->dtype;
    uint32_t num_dtypes_per_double = (FP64 / dtype_bytes);

    uint32_t num_doubles_per_aligned_point = ceildiv(C, num_dtypes_per_double);
    uint32_t num_doubles = num_points * num_doubles_per_aligned_point;
    bool is_point_aligned_to_8_byte_boundary = C % num_dtypes_per_double == 0;
    uint32_t num_bytes_per_aligned_point =
        num_doubles_per_aligned_point * sizeof(double);
    uint32_t num_bytes_in_point_aligned_ifmap = num_doubles * sizeof(double);

    uint32_t num_bytes_per_packed_point = C * dtype_bytes;

    const uint32_t num_doubles_work_for_core_per_aligned_point =
        get_core_num_work_items(num_doubles_per_aligned_point,
                                num_compute_cores, compute_id);

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    v2s *ptr = (v2s *)raw_ptr;
    v2s *dotp_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *grad_bias_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *current_mean_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *invstd_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *weight_times_invstd_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *grad_weight_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *k_scratch = ptr;
    ptr += num_doubles_per_aligned_point;
    v2s *winvstd_times_meank_sub_dmean_scratch = ptr;
    ptr += num_doubles_per_aligned_point;

    v2s *used_tcdm_end_addr =
        (v2s *)(snrt_l1_end_addr() -
                (snrt_l1_end_addr() - snrt_l1_start_addr()) /
                    4);  // use 3/4 for now
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    ptrdiff_t tile_size_in_aligned_points =
        (space_left) / (2 * 2 * num_doubles_per_aligned_point);

    ptrdiff_t grad_ofmap_len =
        tile_size_in_aligned_points * num_doubles_per_aligned_point * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    v2s *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    v2s *ifmap_scratch = ptr;
    ptr += ifmap_len;
    v2s *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t invstd_load, current_var_load, weight_load,
        current_mean_load, grad_ofmap_load, ifmap_load;

    uint32_t doubles_loadable =
        max(ceildiv(num_doubles_per_aligned_point, num_compute_cores) * 50 * 7,
            128);
    uint32_t points_loadable = doubles_loadable / num_doubles_per_aligned_point;
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_aligned_points), num_points);

    bool is_tiling_enabled = work_in_tile != num_points;
    uint32_t work_left = num_points;
    uint32_t work_mod_3 = work_in_tile % 3;
    uint32_t work_div_3_sub_1 = work_in_tile / 3 - 1;
    uint32_t work_mod_4 = work_in_tile % 4;
    uint32_t work_div_4_sub_1 = work_in_tile / 4 - 1;
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_3 = work_mod_3;
        dm_comm->work_div_3_sub_1 = work_div_3_sub_1;  // this is the frep value
    }

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        current_var_load = snrt_dma_start_1d(invstd_scratch, l->current_var,
                                             num_bytes_per_packed_point);
        weight_load = snrt_dma_start_1d(weight_times_invstd_scratch, l->weight,
                                        num_bytes_per_packed_point);
        snrt_dma_wait(current_var_load);
        snrt_dma_wait(weight_load);
    } else {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL, 2, num_doubles_work_for_core_per_aligned_point,
            num_bytes_per_aligned_point, num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        grad_ofmap_load = initiate_dma_1d_or_2d(
            grad_ofmap_scratch, l->grad_ofmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point,
            work_in_tile, is_point_aligned_to_8_byte_boundary);
        ifmap_load = initiate_dma_1d_or_2d(
            ifmap_scratch, l->ifmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point,
            work_in_tile, is_point_aligned_to_8_byte_boundary);
        current_mean_load = snrt_dma_start_1d(
            current_mean_scratch, l->current_mean, num_bytes_per_packed_point);
        buf_flag = !buf_flag;
        snrt_dma_wait_all();
    } else {
        if (num_doubles_work_for_core_per_aligned_point > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &invstd_scratch[compute_id]);
            register float eps = l->eps;
            const register float ONE = 1;
            snrt_ssr_enable();
            asm volatile(
                "vfcpka.s.s %[ONE],%[ONE],%[ONE]\n"
                "frep.o %[n_frep], 5, 0, 0 \n"
                "vfadd.r.s ft3, ft0, %[eps]\n"
                "vfsqrt.s ft3, ft3\n"
                "vfdiv.s ft3, %[ONE], ft3\n"
                // write out invstd
                "vfsgnj.s ft1, ft3, ft3\n"
                // write out invstd * weight
                "vfmul.s ft1, ft0, ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_doubles_work_for_core_per_aligned_point - 1)
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    if (!is_tiling_enabled) {
        uint32_t start_main_loop_1 = SNRT_SECTIONED_MCYCLE();
        if (snrt_is_dm_core()) {
        } else {
            if (num_doubles_work_for_core_per_aligned_point > 0) {
                batchnorm_backward_training_fp32_no_loop_1(
                    &grad_ofmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &grad_bias_scratch[compute_id], &dotp_scratch[compute_id],
                    num_bytes_per_aligned_point, work_in_tile, work_mod_3,
                    work_div_3_sub_1,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores);
            }
        }
        uint32_t end_main_loop_1 = SNRT_SECTIONED_MCYCLE();
    } else {
        if (snrt_is_dm_core()) {
            buf_flag = batchnorm_backward_training_dma_main_loop_fp_agnostic(
                l, num_doubles_per_aligned_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point,
                is_point_aligned_to_8_byte_boundary, num_points, work_left,
                work_in_tile, dm_comm, tile_size_in_aligned_points,
                grad_ofmap_scratch, ifmap_scratch, NULL, buf_flag, 3, 4);
        } else {
            if (num_doubles_work_for_core_per_aligned_point == 0) {
                // start up first tile
                uint32_t work_in_tile_temp = work_in_tile;
                snrt_cluster_hw_barrier();
                while (work_in_tile_temp != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile_temp = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                buf_flag = batchnorm_backward_training_tile_fp32_looped_1(
                    &grad_ofmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &current_mean_scratch[compute_id],
                    &grad_bias_scratch[compute_id], &dotp_scratch[compute_id],
                    num_doubles_per_aligned_point, work_in_tile, work_mod_3,
                    work_div_3_sub_1, tile_size_in_aligned_points,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores, dm_comm);
            }
        }

        work_in_tile = min(tile_size_in_aligned_points, num_points);
        work_mod_4 = work_in_tile % 4;
        work_div_4_sub_1 = work_in_tile / 4 - 1;
    }
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();

    uint32_t start_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch,
                          num_bytes_per_packed_point);
        work_left = num_points - work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_4 = work_mod_4;
        dm_comm->work_div_4_sub_1 = work_div_4_sub_1;
    } else if (snrt_is_compute_core()) {
        if (num_doubles_work_for_core_per_aligned_point > 0) {
            register v2s num_points_inv_reg asm(
                "ft6");  // can consider fcvt instead
            asm volatile(
                "vfcpka.s.s %[num_points_inv_reg],%[num_points_inv],%[num_points_inv]\n"  // duplicate the num_points
                : [num_points_inv_reg] "+fr"(num_points_inv_reg.f64)
                : [num_points_inv] "fr"(num_points_inv)
                : "ft0", "ft1", "ft2");
            snrt_ssr_loop_2d(SNRT_SSR_DM0, 2,
                             num_doubles_work_for_core_per_aligned_point,
                             num_bytes_per_aligned_point,
                             num_compute_cores * sizeof(double));
            snrt_ssr_repeat(SNRT_SSR_DM0, 2);
            snrt_ssr_loop_2d(SNRT_SSR_DM1, 3,
                             num_doubles_work_for_core_per_aligned_point,
                             num_bytes_per_aligned_point,
                             num_compute_cores * sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM2, 3,
                             num_doubles_work_for_core_per_aligned_point,
                             num_bytes_per_aligned_point,
                             num_compute_cores * sizeof(double));
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D,
                           &grad_weight_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &dotp_scratch[compute_id]);
            snrt_ssr_enable();

            asm volatile(
                "frep.o %[n_frep], 10, 0, 0 \n"
                // grad_weight = invstd*dotp
                "vfmul.s ft3, ft0, ft2\n"
                // grad_mean = grad_bias / num_points
                "vfmul.s ft4, ft2, %[num_points_inv_reg]\n"
                // ft5 = invstd / num_points
                "vfmul.s ft5, ft0, %[num_points_inv_reg]\n"
                // write out grad_weight
                "vfsgnj.s ft1, ft3, ft3\n"
                // k = grad_weight * invstd / num_points
                "vfmul.s ft6, ft3, ft5\n"
                // ft7 = grad_mean * (weight*invstd)
                "vfmul.s ft7, ft4, ft0\n"
                // write out k
                "vfsgnj.s ft1, ft6, ft6\n"
                // ft8 = mean * k
                "vfmul.s ft8, ft2, ft6\n"
                // ft8 = (mean * k) * (weight * invstd)
                "vfmul.s ft8, ft8, ft0\n"
                // write out (mean*k)*(weight*invstd) -
                // (grad_mean)*(weight*invstd)
                "vfsub.s ft1, ft8, ft7\n"
                :
                : [n_frep] "r"(num_doubles_work_for_core_per_aligned_point - 1),
                  [num_points_inv_reg] "fr"(num_points_inv_reg.f64)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                  "ft8");
            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
            snrt_ssr_repeat(SNRT_SSR_DM0, 1);
        }
    }
    uint32_t end_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    if (!is_tiling_enabled) {
        uint32_t start_main_loop_2 = SNRT_SECTIONED_MCYCLE();
        if (snrt_is_dm_core()) {
            snrt_cluster_hw_barrier();
            initiate_dma_1d_or_2d(
                (void *)l->grad_ifmap, (void *)grad_ifmap_scratch,
                num_bytes_per_packed_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point, work_in_tile,
                is_point_aligned_to_8_byte_boundary);
        } else {
            if (num_doubles_work_for_core_per_aligned_point > 0) {
                batchnorm_backward_training_fp32_no_loop_2(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &weight_times_invstd_scratch[compute_id],
                    &k_scratch[compute_id],
                    &winvstd_times_meank_sub_dmean_scratch[compute_id],
                    num_bytes_per_aligned_point, work_in_tile, work_mod_4,
                    work_div_4_sub_1,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores);
            }
            // notify finish
            snrt_cluster_hw_barrier();
        }
        uint32_t end_main_loop_2 = SNRT_SECTIONED_MCYCLE();
    } else {
        if (snrt_is_dm_core()) {
            batchnorm_backward_training_dma_main_loop_fp_agnostic(
                l, num_doubles_per_aligned_point, num_bytes_per_packed_point,
                num_bytes_per_aligned_point,
                is_point_aligned_to_8_byte_boundary, num_points, work_left,
                work_in_tile, dm_comm, tile_size_in_aligned_points,
                grad_ofmap_scratch, ifmap_scratch, grad_ifmap_scratch, buf_flag,
                4, 5);
        } else {
            if (num_doubles_work_for_core_per_aligned_point == 0) {
                uint32_t work_in_tile_temp = work_in_tile;
                // start up first tile
                snrt_cluster_hw_barrier();
                while (work_in_tile_temp != 0) {
                    // wait for dma to compute result and signify work is done
                    snrt_cluster_hw_barrier();
                    work_in_tile_temp = dm_comm->num_points_work_in_tile;
                    // "signal" work is done
                    snrt_cluster_hw_barrier();
                }
            } else {
                batchnorm_backward_training_tile_fp32_looped_2(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &weight_times_invstd_scratch[compute_id],
                    &k_scratch[compute_id],
                    &winvstd_times_meank_sub_dmean_scratch[compute_id], buf_flag,
                    num_doubles_per_aligned_point, work_in_tile, work_mod_4,
                    work_div_4_sub_1, tile_size_in_aligned_points,
                    num_doubles_work_for_core_per_aligned_point,
                    num_compute_cores, dm_comm);
            }
        }
    }

    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          num_bytes_per_packed_point);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}
