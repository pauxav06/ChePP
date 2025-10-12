#include <experimental/mdspan>
#include <hwy/base.h>
#include <experimental/mdspan>
#include <experimental/mdarray>

#if defined(CHEPP_RELU_INL_H) == defined(HWY_TARGET_TOGGLE)
#ifdef CHEPP_RELU_INL_H
#undef CHEPP_RELU_INL_H
#else
#define CHEPP_RELU_INL_H
#endif

#include <hwy/highway.h>
#include "utils-inl.h"

HWY_BEFORE_NAMESPACE();

namespace chepp::nnue::layers::relu
{
    namespace HWY_NAMESPACE
    {
        namespace hn = hwy::HWY_NAMESPACE;
        namespace nn = chepp::nnue::HWY_NAMESPACE;
        using namespace std::experimental;


        template <typename T>
        struct ClippedReLUParams
        {
            T       clip_min = 0;
            T       clip_max = 127;
            int32_t shift    = 0;
            size_t  Unroll   = 0;
        };

        template <typename InputT, typename OutputT, size_t Size, ClippedReLUParams<InputT> Params_>
        struct ScalarNarrowingClippedReLU
        {
            ScalarNarrowingClippedReLU() = default;

            static void forward(const InputT* input, OutputT* output)
            {
                for (size_t i = 0; i < Size; ++i)
                {
                    InputT val = input[i];
                    if constexpr (Params_.shift != 0)
                        val >>= Params_.shift;
                    val       = std::max(Params_.clip_min, val);
                    val       = std::min(Params_.clip_max, val);
                    output[i] = static_cast<OutputT>(val);
                }
            }
        };

        template <size_t Size, typename InputT, ClippedReLUParams Params_>
        struct SIMDNarrowingClippedReLU
        {
            using Din  = hn::ScalableTag<InputT>;
            using Vin  = hn::VFromD<Din>;
            using Dout = hn::RepartitionToNarrow<Din>;
            using Vout = hn::VFromD<Dout>;
            using OutT = hn::TFromD<Dout>;

            static HWY_LANES_CONSTEXPR size_t InLanes   = hn::Lanes(Din());
            static HWY_LANES_CONSTEXPR size_t OutLanes  = hn::Lanes(Dout());
            static HWY_LANES_CONSTEXPR size_t ChunksIn  = Size / InLanes;
            static HWY_LANES_CONSTEXPR size_t ChunksOut = Size / OutLanes;

            using in_extent_t  = extents<size_t,
                nn::extent_if_constexpr_v<ChunksIn / 2>,
                2,
                nn::extent_if_constexpr_v<InLanes>>;
            using out_extent_t = extents<size_t,
                nn::extent_if_constexpr_v<ChunksOut>,
                nn::extent_if_constexpr_v<OutLanes>>;

            using in_t         = mdspan<const InputT, in_extent_t>;
            using out_t        = mdspan<OutT, out_extent_t>;

            static HWY_LANES_CONSTEXPR in_extent_t  InExtent{ChunksIn / 2, 2, InLanes};
            static HWY_LANES_CONSTEXPR out_extent_t OutExtent{ChunksOut, OutLanes};

            HWY_NOINLINE static void forward(const InputT* HWY_RESTRICT input, OutT* HWY_RESTRICT output)
            {
                in_t  in{input, InExtent};
                out_t out{output, OutExtent};
                HWY_DEFAULT_UNROLL
                for (size_t c = 0; c < ChunksIn / 2; c++)
                {
                    Vin v0 = hn::Load(Din(), &in[c, 0, 0]);
                    Vin v1 = hn::Load(Din(), &in[c, 1, 0]);
                    if constexpr (Params_.shift != 0)
                    {
                        v0 = hn::ShiftLeft<Params_.shift>(v0);
                        v1 = hn::ShiftLeft<Params_.shift>(v1);
                    }

                    Vout v_out = hn::OrderedDemote2To(Dout(), v0, v1);
                    if constexpr (Params_.clip_min > std::numeric_limits<OutT>::min() &&
                                  Params_.clip_min > std::numeric_limits<InputT>::min())
                        v_out = hn::Max(hn::Set(Dout(), Params_.clip_min), v_out);
                    if constexpr (Params_.clip_max < std::numeric_limits<OutT>::max() &&
                                  Params_.clip_max < std::numeric_limits<InputT>::max())
                        v_out = hn::Min(hn::Set(Dout(), Params_.clip_max), v_out);

                    hn::Store(v_out, Dout(), &out[c, 0]);
                }
            }
        };

        template <size_t Size, typename InputT, ClippedReLUParams<InputT> Params_>
        struct SIMDNarrowingX2ClippedReLU
        {
            using Din = hn::ScalableTag<InputT>;
            using Vin = hn::VFromD<Din>;

            using Dmid = hn::RepartitionToNarrow<Din>;
            using Vmid = hn::VFromD<Dmid>;
            using MidT = hn::TFromD<Dmid>;

            using Dout = hn::RepartitionToNarrow<Dmid>;
            using Vout = hn::VFromD<Dout>;
            using OutT = hn::TFromD<Dout>;

            static constexpr size_t Unroll = Params_.Unroll;

            static HWY_LANES_CONSTEXPR size_t InLanes   = hn::Lanes(Din());
            static HWY_LANES_CONSTEXPR size_t MidLanes  = hn::Lanes(Dmid());
            static HWY_LANES_CONSTEXPR size_t OutLanes  = hn::Lanes(Dout());
            static HWY_LANES_CONSTEXPR size_t ChunksIn  = Size / InLanes;
            static HWY_LANES_CONSTEXPR size_t ChunksOut = Size / OutLanes;

            using in_extent_t = extents<
                size_t,
                nn::extent_if_constexpr_v<ChunksIn / 4>,
                4,
                nn::extent_if_constexpr_v<InLanes>>;

            using out_extent_t = extents<
                size_t,
                nn::extent_if_constexpr_v<ChunksOut>,
                nn::extent_if_constexpr_v<OutLanes>>;

            static constexpr in_extent_t  InExtent{ChunksIn / 4, 4, InLanes};
            static constexpr out_extent_t OutExtent{ChunksOut, OutLanes};

            using in_t  = mdspan<const InputT, in_extent_t>;
            using out_t = mdspan<OutT, out_extent_t>;

            HWY_NOINLINE static void forward(const InputT* HWY_RESTRICT input, OutT* HWY_RESTRICT output)
            {
                in_t  in{input, InExtent};
                out_t out{output, OutExtent};

                HWY_DEFAULT_UNROLL
                for (size_t c = 0; c < ChunksIn / 4; ++c)
                {
                    Vin v0 = hn::Load(Din(), &in[c, 0, 0]);
                    Vin v1 = hn::Load(Din(), &in[c, 1, 0]);
                    Vin v2 = hn::Load(Din(), &in[c, 2, 0]);
                    Vin v3 = hn::Load(Din(), &in[c, 3, 0]);

                    if constexpr (Params_.shift != 0)
                    {
                        v0 = hn::ShiftRight<Params_.shift>(v0);
                        v1 = hn::ShiftRight<Params_.shift>(v1);
                        v2 = hn::ShiftRight<Params_.shift>(v2);
                        v3 = hn::ShiftRight<Params_.shift>(v3);
                    }

                    Vmid v01 = hn::OrderedDemote2To(Dmid(), v0, v1);
                    Vmid v23 = hn::OrderedDemote2To(Dmid(), v2, v3);

                    Vout v_out = hn::OrderedDemote2To(Dout(), v01, v23);

                    if constexpr (Params_.clip_min > std::numeric_limits<OutT>::min())
                        v_out = hn::Max(hn::Set(Dout(), Params_.clip_min), v_out);
                    if constexpr (Params_.clip_max < std::numeric_limits<OutT>::max())
                        v_out = hn::Min(hn::Set(Dout(), Params_.clip_max), v_out);

                    hn::Store(v_out, Dout(), &out[c, 0]);
                }
            }
        };

    }; // namespace HWY_NAMESPACE
} // namespace chepp::nnue::activation

HWY_AFTER_NAMESPACE();

#endif // CHEPP_RELU_INL_H
