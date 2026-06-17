#ifndef CHEPP_NNUE_MATRIX_H_
#define CHEPP_NNUE_MATRIX_H_

#include <type_traits>
#include <vector>

namespace chepp::nnue::matrix {
    template <typename M>
    concept MatrixViewLike = requires(const std::decay_t<M>& m, size_t r, size_t c) {
        typename std::decay_t<M>::value_type;
        { m(r, c) } -> std::convertible_to<typename std::decay_t<M>::value_type>;
        { m.nRows() } -> std::convertible_to<size_t>;
        { m.nCols() } -> std::convertible_to<size_t>;
    };

    template <typename T>
    struct MatrixView {
        using value_type = T;
        const T* data;
        size_t   rows, cols;

        const T& operator()(size_t r, size_t c) const { return data[r * cols + c]; }
        T&       operator()(size_t r, size_t c) { return data[r * cols + c]; }

        [[nodiscard]] size_t nRows() const { return rows; }
        [[nodiscard]] size_t nCols() const { return cols; }
    };

    template <MatrixViewLike Base>
    struct TransposeView {
        using value_type = Base::value_type;
        Base base;

        [[nodiscard]] size_t nRows() const { return base.nCols(); }
        [[nodiscard]] size_t nCols() const { return base.nRows(); }

        const value_type& operator()(size_t r, size_t c) const { return base(c, r); }
    };

    template <MatrixViewLike Base>
    auto transpose(Base&& base) {
        return TransposeView<std::decay_t<Base>>{std::forward<Base>(base)};
    }

    template <MatrixViewLike Base>
    struct ShuffleRowsView {
        using value_type = Base::value_type;
        Base                base;
        std::vector<size_t> map;

        [[nodiscard]] size_t nRows() const { return base.nRows(); }
        [[nodiscard]] size_t nCols() const { return base.nCols(); }

        const value_type& operator()(size_t r, size_t c) const { return base(map[r], c); }
    };

    template <MatrixViewLike Base>
    auto shuffle_rows(Base&& base, std::vector<size_t> map) {
        return ShuffleRowsView<std::decay_t<Base>>{std::forward<Base>(base), std::move(map)};
    }

    template <MatrixViewLike Base>
    struct ShuffleColsView {
        using value_type = Base::value_type;
        Base                base;
        std::vector<size_t> map;

        [[nodiscard]] size_t nRows() const { return base.nRows(); }
        [[nodiscard]] size_t nCols() const { return base.nCols(); }

        const value_type& operator()(size_t r, size_t c) const { return base(r, map[c]); }
    };

    template <MatrixViewLike Base>
    auto shuffle_cols(Base&& base, std::vector<size_t> map) {
        return ShuffleColsView<std::decay_t<Base>>{std::forward<Base>(base), std::move(map)};
    }

    template <MatrixViewLike Base>
    struct ReshapeView {
        using value_type = Base::value_type;
        Base   base;
        size_t newRows, newCols;

        [[nodiscard]] size_t nRows() const { return newRows; }
        [[nodiscard]] size_t nCols() const { return newCols; }

        const value_type& operator()(size_t r, size_t c) const {
            size_t flat   = r * newCols + c;
            size_t orig_r = flat / base.nCols();
            size_t orig_c = flat % base.nCols();
            return base(orig_r, orig_c);
        }
    };

    template <MatrixViewLike Base>
    auto reshape(Base&& base, size_t rows, size_t cols) {
        return ReshapeView<std::decay_t<Base>>{std::forward<Base>(base), rows, cols};
    }

    template <MatrixViewLike Base>
    struct TileColsView {
        using value_type = Base::value_type;
        Base   base;
        size_t block;

        [[nodiscard]] size_t nRows() const { return base.nCols() / block; }
        [[nodiscard]] size_t nCols() const { return base.nRows() * block; }

        const value_type& operator()(size_t r, size_t c) const {
            size_t orig_row = c / block;
            size_t orig_col = r * block + (c % block);
            return base(orig_row, orig_col);
        }
    };

    template <MatrixViewLike Base>
    auto tile_cols(Base&& base, size_t block) {
        return TileColsView<std::decay_t<Base>>{std::forward<Base>(base), block};
    }

    template <MatrixViewLike Base>
    struct HSplitView {
        using value_type = Base::value_type;
        Base   base;
        size_t parts;

        HSplitView(Base b, size_t p) : base(std::move(b)), parts(p) {
            HWY_ASSERT(base.nCols() % parts == 0);
        }

        [[nodiscard]] size_t nRows() const { return base.nRows() * parts; }
        [[nodiscard]] size_t nCols() const { return base.nCols() / parts; }

        const value_type& operator()(size_t r, size_t c) const {
            size_t orig_r = r % base.nRows();
            size_t block  = r / base.nRows();
            size_t orig_c = c + block * nCols();
            return base(orig_r, orig_c);
        }
    };

    template <MatrixViewLike Base>
    auto hsplit(Base&& base, size_t parts) {
        return HSplitView<std::decay_t<Base>>{std::forward<Base>(base), parts};
    }

    template <MatrixViewLike Base>
    struct PadView {
        using value_type = Base::value_type;
        Base   base;
        size_t extraRows;
        size_t extraCols;

        [[nodiscard]] size_t nRows() const { return base.nRows() + extraRows; }
        [[nodiscard]] size_t nCols() const { return base.nCols() + extraCols; }

        const value_type& operator()(size_t r, size_t c) const {
            static const value_type zero{};
            if (r < base.nRows() && c < base.nCols()) return base(r, c);
            return zero;
        }
    };

    template <MatrixViewLike Base>
    auto pad(Base&& base, size_t extraRows, size_t extraCols) {
        return PadView<std::decay_t<Base>>{std::forward<Base>(base), extraRows, extraCols};
    }

    template <MatrixViewLike M>
    void materialize(const M& m, typename M::value_type* out) {
        const size_t rows = m.nRows();
        const size_t cols = m.nCols();
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) out[r * cols + c] = m(r, c);
    }

} // namespace chepp::nnue::matrix

#endif