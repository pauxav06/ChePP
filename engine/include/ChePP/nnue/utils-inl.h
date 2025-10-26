#include "utils.h"
#include <experimental/mdarray>
#include <experimental/mdspan>

#if defined(CHEPP_UNROLLER_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_UNROLLER_INL_H_
#undef CHEPP_UNROLLER_INL_H_
#else
#define CHEPP_UNROLLER_INL_H_
#endif

#define REPEAT_1(M) M(0)
#define REPEAT_2(M) REPEAT_1(M) M(1)
#define REPEAT_3(M) REPEAT_2(M) M(2)
#define REPEAT_4(M) REPEAT_3(M) M(3)
#define REPEAT_5(M) REPEAT_4(M) M(4)
#define REPEAT_6(M) REPEAT_5(M) M(5)
#define REPEAT_7(M) REPEAT_6(M) M(6)
#define REPEAT_8(M) REPEAT_7(M) M(7)
#define REPEAT_9(M) REPEAT_8(M) M(8)
#define REPEAT_10(M) REPEAT_9(M) M(9)
#define REPEAT_11(M) REPEAT_10(M) M(10)
#define REPEAT_12(M) REPEAT_11(M) M(11)
#define REPEAT_13(M) REPEAT_12(M) M(12)
#define REPEAT_14(M) REPEAT_13(M) M(13)
#define REPEAT_15(M) REPEAT_14(M) M(14)
#define REPEAT_16(M) REPEAT_15(M) M(15)
#define REPEAT_17(M) REPEAT_16(M) M(16)
#define REPEAT_18(M) REPEAT_17(M) M(17)
#define REPEAT_19(M) REPEAT_18(M) M(18)
#define REPEAT_20(M) REPEAT_19(M) M(19)
#define REPEAT_21(M) REPEAT_20(M) M(20)
#define REPEAT_22(M) REPEAT_21(M) M(21)
#define REPEAT_23(M) REPEAT_22(M) M(22)
#define REPEAT_24(M) REPEAT_23(M) M(23)
#define REPEAT_25(M) REPEAT_24(M) M(24)
#define REPEAT_26(M) REPEAT_25(M) M(25)
#define REPEAT_27(M) REPEAT_26(M) M(26)
#define REPEAT_28(M) REPEAT_27(M) M(27)
#define REPEAT_29(M) REPEAT_28(M) M(28)
#define REPEAT_30(M) REPEAT_29(M) M(29)
#define REPEAT_31(M) REPEAT_30(M) M(30)
#define REPEAT_32(M) REPEAT_31(M) M(31)
#define REPEAT(N, M) REPEAT_##N(M)

#include <hwy/highway.h>
HWY_BEFORE_NAMESPACE();

namespace chepp {
    namespace nnue {
        namespace HWY_NAMESPACE {
            namespace hn = hwy::HWY_NAMESPACE;
            using namespace std::experimental;

            using extent_type = std::size_t;

            // The highway documentation warns us about using STL containers for SIMD vectors because they are sizeless
            // and they might blow the stack. We instead should declare them as local variables.
            // However, this would make it impossible to change the register count based on a constexpr param, and
            // force very repetitive code (declaring acc0, acc1, acc2 ...)
            // This utility soves that problem by declaring 32 registers locally and providing access functions.
            // The unused registers will be optimised out by the compiler, and we get a modular interface that lets us
            // access registers via an index! All accessor functions will also be completly removed, and we essentially
            // get 0 overhead simd register container
            template <size_t N, typename VecT>
            struct RegisterBank {
                template <typename Func, typename Init>
                static HWY_INLINE void
                run(Init&& init, Func&& f) {
                    static_assert(N <= 16, "Unroll factor exceeds 16");
                    using D = hn::DFromV<VecT>;

#define DECL_REG_LOCAL(I)                                                                                              \
    VecT reg##I = [=] {                                                                                                \
        if constexpr ((I) < N)                                                                                         \
            return init(I);                                                                                            \
        else                                                                                                           \
            return hn::Undefined(D());                                                                                 \
    }();

                    REPEAT(32, DECL_REG_LOCAL)
#undef DECL_REG_LOCAL

                    auto get_reg = [&](const size_t idx) -> VecT {
                        if constexpr (N == 0) return hn::Undefined(D());
                        if (idx >= N) return hn::Undefined(D());
                        switch (idx) {
#define SWITCH_GET(I)                                                                                                  \
    case I:                                                                                                            \
        return reg##I;
                            REPEAT(32, SWITCH_GET)
#undef SWITCH_GET
                            default:
                                return hn::Undefined(D());
                        }
                    };

                    auto set_reg = [&](const size_t idx, VecT val) {
                        if constexpr (N == 0) return;
                        if (idx >= N) return;
                        switch (idx) {
#define SWITCH_SET(I)                                                                                                  \
    case I:                                                                                                            \
        reg##I = val;                                                                                                  \
        break;
                            REPEAT(32, SWITCH_SET)
#undef SWITCH_SET
                            default:
                                break;
                        }
                    };

                    f(get_reg, set_reg);
                }
            };

#define IF_ELSE(cond, val, other) _IF_ELSE(cond, val, other)
#define _IF_ELSE(cond, val, other) IF_##cond(val, other)

#define IF_1(val, other) (val)
#define IF_0(val, other) (other)

#define EXTENT_IF(cond, val) (IF_ELSE(cond, val, std::dynamic_extent))
#define EXTENT_IF_LANES_CONSTEXPR(val) EXTENT_IF(HWY_HAVE_CONSTEXPR_LANES, val)

#define STATIC_EXTENT(NAME, val) chepp::nnue::utils::extent_wrapper<val> NAME{.value = val};

#define MAYBE_STATIC_EXTENT(NAME, val, cond)                                                                           \
    chepp::nnue::utils::extent_wrapper<ENTENT_IF(cond, val)> NAME{.value = val};

#define DYNAMIC_EXTENT(NAME, val) chepp::nnue::utils::extent_wrapper<std::dynamic_extent> NAME{.value = val};
        } // namespace HWY_NAMESPACE
    } // namespace nnue
} // namespace chepp

HWY_AFTER_NAMESPACE();
#endif // CHEPP_UNROLLER_INL_H_
