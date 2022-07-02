/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDAD_SRC_OPERATIONS_OPERATIONS_H_
#define CUDAD_SRC_OPERATIONS_OPERATIONS_H_

#include "adam/adam.h"
#include "add/add.h"
#include "add_mv/add_mv.h"
#include "add_mv_bp/add_mv_bp.h"
#include "bucket/bucket.h"
#include "bucket_bp/bucket_bp.h"
#include "clipped_relu/clipped_relu.h"
#include "clipped_relu_bp/clipped_relu_bp.h"
#include "merge/merge.h"
#include "merge_bp/merge_bp.h"
#include "mle/mle.h"
#include "mm/mm.h"
#include "mm_bp/mm_bp.h"
#include "mpe/mpe.h"
#include "mse/mse.h"
#include "pairwise_multiply/pairwise_multiply.h"
#include "pairwise_multiply_bp/pairwise_multiply_bp.h"
#include "relu/relu.h"
#include "relu_bp/relu_bp.h"
#include "sigmoid/sigmoid.h"
#include "sigmoid_bp/sigmoid_bp.h"
#include "sparse_affine/sparse_affine.h"
#include "sparse_affine_bp/sparse_affine_bp.h"

#endif    // CUDAD_SRC_OPERATIONS_OPERATIONS_H_
