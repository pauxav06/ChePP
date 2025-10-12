#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

namespace chepp::nnue::layers
{
    namespace affine
    {
        enum class Kernels
        {
            Scalar,
            SIMD
        };
    }
    namespace relu
    {
        enum class Kernels
        {
            Scalar,
            SIMD
        };
    }
}


#endif
