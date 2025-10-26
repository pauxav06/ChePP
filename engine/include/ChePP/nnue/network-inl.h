#include "meta.h"
#include "network.h"
#include "utils.h"

#define EXPORT_LAYER(NAMESPACE)                                                                                        \
    namespace NAMESPACE {                                                                                              \
        using LayerPtr = ::std::shared_ptr<const layer_t>;                                                             \
        using IState   = layer_t::IState;                                                                              \
        namespace HWY_NAMESPACE {                                                                                      \
            template <std::size_t I>                                                                                   \
            auto                                                                                                       \
            make_state_templated(const LayerPtr& layer) {                                                              \
                return layer::HWY_NAMESPACE::make_state(layer, std::get<I>(params));                                   \
            }                                                                                                          \
                                                                                                                       \
            constexpr auto                                                                                             \
            make_state_array_builder() {                                                                               \
                constexpr std::size_t N = std::tuple_size_v<decltype(params)>;                                         \
                                                                                                                       \
                return []<std::size_t... Is>(std::index_sequence<Is...>) {                                             \
                    return std::array{+[](const LayerPtr& layer) -> std::unique_ptr<IState> {                          \
                        return make_state_templated<Is>(layer);                                                        \
                    }...};                                                                                             \
                }(std::make_index_sequence<N>{});                                                                      \
            }                                                                                                          \
                                                                                                                       \
            inline constexpr auto make_state_table = make_state_array_builder();                                       \
                                                                                                                       \
            inline std::unique_ptr<IState>                                                                             \
            make_state(std::size_t i, const LayerPtr& layer) {                                                         \
                return make_state_table.at(i)(layer);                                                                  \
            }                                                                                                          \
        };                                                                                                             \
    };

#if defined(CHEPP_NETWORK_INL_H_) == defined(HWY_TARGET_TOGGLE) || HWY_IDE
#ifdef CHEPP_NETWORK_INL_H_
#undef CHEPP_NETWORK_INL_H_
#else
#define CHEPP_NETWORK_INL_H_
#endif

#include "affine-inl.h"
#include "relu-inl.h"
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace chepp::nnue::network {
    EXPORT_LAYER(l1)
    EXPORT_LAYER(act0)
} // namespace chepp::nnue::network
HWY_AFTER_NAMESPACE();

#endif
