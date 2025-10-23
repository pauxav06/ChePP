#include "meta.h"
#include "network.h"
#include "relu-inl.h"
#include "utils.h"

#if defined(CHEPP_NETWORK_INL_H_) == defined(HWY_TARGET_TOGGLE) || HWY_IDE
#ifdef CHEPP_NETWORK_INL_H_
#undef CHEPP_NETWORK_INL_H_
#else
#define CHEPP_NETWORK_INL_H_
#endif

#include "accumulator-inl.h"
#include "affine-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::network {

#define EXPORT_LAYER(name) using accum_layers = apply_template_t<name## ::layer::HWY_NAMESPACE::Layer, name## ::params>;

    namespace HWY_NAMESPACE {
        using namespace meta;
        using accum_layers = apply_template_t<accumulator::layer::HWY_NAMESPACE::Layer, accumulator::params>;
        EXPORT_LAYER(accumulator);
        using psqt_layers = apply_template_t<psqt::layer::HWY_NAMESPACE::Layer, psqt::params>;
        using act0_layers = apply_template_t<act0::layer::HWY_NAMESPACE::Layer, act0::params>;
        using l1_layers   = apply_template_t<l1::layer::HWY_NAMESPACE::Layer, l1::params>;
        using act1_layers = apply_template_t<act1::layer::HWY_NAMESPACE::Layer, act1::params>;
        using l2_layers   = apply_template_t<l2::layer::HWY_NAMESPACE::Layer, l2::params>;
        using act2_layers = apply_template_t<act2::layer::HWY_NAMESPACE::Layer, act2::params>;
        using l3_layers   = apply_template_t<l3::layer::HWY_NAMESPACE::Layer, l3::params>;

        using networks_t = Cartesian<act0_layers, l1_layers>::type;
        using networks   = MakeNetworks_t<networks_t>;
        inline networks test{};
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue::network
HWY_AFTER_NAMESPACE();
#endif
