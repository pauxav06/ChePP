#include "nnue.h"
#include <hwy/base.h>
#include <hwy/nanobenchmark.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "nnue.cpp"

#include "hwy/foreach_target.h"
#include <hwy/highway.h>

#include "accumulator-inl.h"
#include "affine-inl.h"
#include "relu-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue {
    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;
        template <typename T, auto cfg>
        void
        register_kernel(KernelRegistry& registry) {
            registry.register_kernel<T, HWY_TARGET, cfg>();
        }
    } // namespace HWY_NAMESPACE
} // namespace chepp::nnue

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace chepp::nnue {
    template <typename T, auto cfg>
    void
    register_kernel(KernelRegistry& registery) {
#define VISITOR(TARGET, NAMESPACE) chepp::nnue::NAMESPACE::register_kernel<T, cfg>(registery);
        HWY_VISIT_TARGETS(VISITOR)
#undef VISITOR
    }

    void
    register_all_layers(KernelRegistry& registry) {
        constexpr_for<0, std::tuple_size_v<decltype(ALL_LAYERS)>>([&](auto i) {
            using layer_config_t                    = std::tuple_element_t<i, decltype(ALL_LAYERS)>;
            using layer_t                           = layer_config_t::layer_t;
            static constexpr auto        configs    = layer_config_t::configs;
            static constexpr std::size_t nb_configs = std::tuple_size_v<decltype(configs)>;
            constexpr_for<0, nb_configs>([&](auto j) {
                static constexpr auto config = std::get<j>(configs);
                register_kernel<layer_t, config>(registry);
            });
        });
    }
} // namespace chepp::nnue
// HWY_TEST_MAIN();
#endif // HWY_ONCE