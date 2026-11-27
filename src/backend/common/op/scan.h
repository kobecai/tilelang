/*!
 * \file tl/backend/common/op/scan.h
 * \brief Shared inclusive scan lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_SCAN_H_
#define TVM_TL_BACKEND_COMMON_OP_SCAN_H_

#include "op/scan.h"
#include "support/check.h"

#include "op/utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

#include <sstream>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

namespace scan {

template <typename ScanOpNode>
Stmt LowerSharedScan(const ScanOpNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, const char *op_name,
                     const char *pretty_name, const char *symbol_prefix,
                     const char *scan_noun) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    LOG(FATAL) << pretty_name
               << " for fragment not implemented, please raise an issue if "
                  "you need this feature.";
  } else if (IsSharedBuffer(op.src)) {
    ICHECK(IsSharedBuffer(op.dst));
    int ndim = op.logical_rank;
    ICHECK(ndim == 1 || ndim == 2)
        << pretty_name << " currently supports only 1D or 2D buffers, got "
        << ndim << "D.";
    auto check_version_axes = [&](const BufferRegion &region) {
      ICHECK_GE(region->region.size(), static_cast<size_t>(ndim))
          << pretty_name << " region rank is smaller than the logical rank of "
          << region->buffer->name;
      for (size_t i = 0; i < region->region.size() - ndim; ++i) {
        ICHECK(analyzer->CanProveEqual(region->region[i]->extent, 1))
            << pretty_name << " requires extent 1 on leading version axis " << i
            << " of " << region->buffer->name;
      }
    };
    check_version_axes(op.srcRegion_);
    check_version_axes(op.dstRegion_);

    std::stringstream ss;
    auto threads = lower_args.thread_bounds->extent;
    auto thread_offset = lower_args.thread_bounds->min;
    Array<PrimExpr> args;

    // Use all physical indices for the pointers, including the stage offsets.
    PrimExpr src_ptr = MakeAccessPtrFromRegion(op.srcRegion_, 1);
    PrimExpr dst_ptr = MakeAccessPtrFromRegion(op.dstRegion_, 2);
    const auto &src_ranges = op.srcRegion_->region;

    if (ndim == 1) {
      ICHECK_EQ(op.dim, 0) << scan_noun
                           << " over a 1D buffer only supports dim = 0.";
      ss << "tl::" << symbol_prefix << "1D<" << threads << ", "
         << (op.reverse ? "true" : "false") << ", " << thread_offset
         << ">::run";
      args = {StringImm(ss.str()), src_ptr, dst_ptr, src_ranges.back()->extent};
    } else {
      ss << "tl::" << symbol_prefix << "2D<" << threads << ", " << op.dim
         << ", " << (op.reverse ? "true" : "false") << ", " << thread_offset
         << ">::run";
      PrimExpr src_stride = op.src->shape[op.src->shape.size() - 1];
      PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];
      args = {StringImm(ss.str()),
              src_ptr,
              dst_ptr,
              src_ranges[src_ranges.size() - 2]->extent,
              src_ranges.back()->extent,
              src_stride,
              dst_stride};
    }
    return Evaluate(Call(op.dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower " << op_name << " for " << op.src.scope()
                  << " and " << op.dst.scope();
  }

  return Stmt();
}

inline Stmt LowerCumSum(const CumSumOpNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  return LowerSharedScan(op, lower_args, analyzer, "cumsum", "CumSum", "CumSum",
                         "Cumulative sum");
}

inline Stmt LowerCumMax(const CumMaxOpNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  return LowerSharedScan(op, lower_args, analyzer, "cummax", "CumMax", "CumMax",
                         "Cumulative maximum");
}

} // namespace scan

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_SCAN_H_
