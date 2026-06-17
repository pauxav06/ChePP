#include "layer_base.h"
#include "nnue.h"

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
} // namespace chepp::nnue::layers
HWY_AFTER_NAMESPACE();

namespace chepp::nnue::layers {

#if HWY_TARGET == HWY_SCALAR
    template <typename Layer, auto cfg>
        requires (!requires { typename HWY_NAMESPACE::Kernel<Layer, cfg>::sfinae_flag; })
    struct Kernel<HWY_TARGET, Layer, cfg> {
        using type = HWY_NAMESPACE::Kernel<Layer, default_config>;
    };
#endif
    
    template <typename Layer, auto cfg>
        requires requires { typename HWY_NAMESPACE::Kernel<Layer, cfg>::sfinae_flag; }
    struct Kernel<HWY_TARGET, Layer, cfg> {
        using type = HWY_NAMESPACE::Kernel<Layer, cfg>;
    };

}

#endif
