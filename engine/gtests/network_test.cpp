#include "meta.h"
#include "utils.h"

#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "network_test.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#include "network-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers {
    namespace HWY_NAMESPACE {
        namespace {

            namespace hn = hwy::HWY_NAMESPACE;

            void
            test() {
                auto                  layers = Arch::layers;
                static constexpr auto c      = std::make_tuple(0, 0, 0, 0, 0, 0);
                static constexpr auto c1 =
                    std::make_tuple(0, SimdColMajConfig{8, SimdColMajConfig::Operation::SumOfMulQuadAdd}, 0, 0, 0, 0);
                static constexpr auto configs = std::make_tuple(c, c1);

                hwy::AlignedVector<int16_t> inp_mem(2048);
                PaddedSpan<const int16_t>   input{inp_mem};
                hwy::AlignedVector<int32_t> out_mem(100);
                auto                        s = make_state<HWY_TARGET, std::get<0>(configs)>(layers);

                static_for<1>([&](auto I) -> void {
                    static constexpr auto cfg   = std::get<I>(configs);
                    auto                  state = make_state<HWY_TARGET, cfg>(layers);
                    state.init();
                    auto out = state.forward(input, out_mem);
                    std::cout << out[0] << std::endl;
                });
            }

            using namespace hn;
            using namespace chepp::nnue::meta;

        } // namespace
    }     // namespace HWY_NAMESPACE
} // namespace chepp::nnue::layers

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace chepp::nnue::layers {
    namespace {
        HWY_BEFORE_TEST(NetworkTest);
        HWY_EXPORT_AND_TEST_P(NetworkTest, test);
        HWY_AFTER_TEST();
    } // namespace
} // namespace chepp::nnue::layers
// HWY_TEST_MAIN();
#endif // HWY_ONCE