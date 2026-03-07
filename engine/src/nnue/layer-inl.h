#include "layer_base.h"
#include "mdspan.h"

#if defined(CHEPP_LAYER_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_LAYER_INL_H_
#undef CHEPP_LAYER_INL_H_
#else
#define CHEPP_LAYER_INL_H_
#endif

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        template <typename, auto>
        struct Kernel;
    }

    template <typename Layer, auto cfg>
        requires requires { typename HWY_NAMESPACE::Kernel<Layer, cfg>::sfinae_flag; }
    struct Kernel<HWY_TARGET, Layer, cfg> {
        using type = HWY_NAMESPACE::Kernel<Layer, cfg>;
    };

} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

#endif