#include "network.h"

#if defined(CHEPP_NETWORK_INL_H_) == defined(HWY_TARGET_TOGGLE) || HWY_IDE
#ifdef CHEPP_NETWORK_INL_H_
#undef CHEPP_NETWORK_INL_H_
#else
#define CHEPP_NETWORK_INL_H_
#endif

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::network {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::network
HWY_AFTER_NAMESPACE();
#endif
