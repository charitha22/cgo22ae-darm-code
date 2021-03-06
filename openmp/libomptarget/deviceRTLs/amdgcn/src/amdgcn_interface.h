//===--- amdgcn_interface.h - OpenMP interface definitions ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGCN_INTERFACE_H_
#define _AMDGCN_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#define EXTERN extern "C" __attribute__((device))
typedef uint64_t __kmpc_impl_lanemask_t;
typedef uint32_t omp_lock_t; /* arbitrary type of the right length */

////////////////////////////////////////////////////////////////////////////////
// OpenMP interface
////////////////////////////////////////////////////////////////////////////////

EXTERN int omp_get_device_num(void);
EXTERN int omp_ext_get_warp_id(void);
EXTERN int omp_ext_get_lane_id(void);
EXTERN int omp_ext_get_master_thread_id(void);
EXTERN int omp_ext_get_smid(void);
EXTERN int omp_ext_is_spmd_mode(void);
EXTERN unsigned long long omp_ext_get_active_threads_mask(void);

////////////////////////////////////////////////////////////////////////////////
// kmp specifc types
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// external interface
////////////////////////////////////////////////////////////////////////////////

typedef struct ident ident_t;
typedef ident_t kmp_Ident;

// sync barrier
EXTERN void __kmpc_amd_worker_start(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_worker_end(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_start(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_end(kmp_Ident *loc_ref, int32_t tid);
EXTERN void __kmpc_amd_master_terminate(kmp_Ident *loc_ref, int32_t tid);

// dimension queries
EXTERN uint32_t __kmpc_amd_grid_dim_x();
EXTERN uint32_t __kmpc_amd_grid_dim_y();
EXTERN uint32_t __kmpc_amd_grid_dim_z();
EXTERN uint32_t __kmpc_amd_workgroup_dim_x();
EXTERN uint32_t __kmpc_amd_workgroup_dim_y();
EXTERN uint32_t __kmpc_amd_workgroup_dim_z();

#endif
