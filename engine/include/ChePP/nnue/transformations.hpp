#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>

// Utility for transforming weight matrices of nnue
// Relies on type erasure for ease of use
// Therefore this should not be used in hot paths as it will not be optimised for speed

template <typename T>
struct AnyMatrixView {
    struct Concept {
        virtual ~Concept() = default;
        virtual const T& operator()(size_t r, size_t c) const = 0;
        [[nodiscard]] virtual size_t nRows() const = 0;
        [[nodiscard]] virtual size_t nCols() const = 0;
    };

    template <typename View>
    struct Model final : Concept {
        View view;
        explicit Model(View v) : view(std::move(v)) {}
        const T& operator()(size_t r, size_t c) const override { return view(r,c); }
        [[nodiscard]] size_t nRows() const override { return view.nRows(); }
        [[nodiscard]] size_t nCols() const override { return view.nCols(); }
    };

    std::shared_ptr<const Concept> self;

    template <typename View>
    explicit AnyMatrixView(View v) : self(std::make_shared<Model<View>>(std::move(v))) {}

    const T& operator()(size_t r, size_t c) const { return (*self)(r,c); }
    [[nodiscard]] size_t nRows() const { return self->nRows(); }
    [[nodiscard]] size_t nCols() const { return self->nCols(); }

    void materialize(T* out) const {
        for(size_t r=0; r<nRows(); ++r)
            for(size_t c=0; c<nCols(); ++c)
                out[r*nCols()+c] = (*this)(r,c);
    }

    std::string to_string(const int width = 8, const int precision = 4) {
        std::ostringstream ss;
        for (size_t r = 0; r < nRows(); ++r) {
            for (size_t c = 0; c < nCols(); ++c) {
                ss << std::setw(width) << std::fixed << std::setprecision(precision)
                          << +static_cast<T>((*this)(r, c)) << " ";
            }
            ss << "\n";
        }
        ss << std::flush;
        return ss.str();
    }
};

template <typename T>
struct MatrixView {
    T* data;
    size_t rows, cols, stride;

    MatrixView(T* d, size_t r, size_t c, size_t s)
        : data(d), rows(r), cols(c), stride(s) {}

    const T& operator()(size_t r, size_t c) const { return data[r*stride + c]; }
    T& operator()(size_t r, size_t c) { return data[r*stride + c]; }

    [[nodiscard]] size_t nRows() const { return rows; }
    [[nodiscard]] size_t nCols() const { return cols; }
};

template <typename T>
struct TransposeView {
    AnyMatrixView<T> base;
    explicit TransposeView(AnyMatrixView<T> b) : base(std::move(b)) {}

    [[nodiscard]] size_t nRows() const { return base.nCols(); }
    [[nodiscard]] size_t nCols() const { return base.nRows(); }

    const T& operator()(size_t r, size_t c) const { return base(c,r); }
};

template <typename T>
struct ShuffleRowsView {
    AnyMatrixView<T> base;
    std::vector<size_t> map;
    ShuffleRowsView(AnyMatrixView<T> b, std::vector<size_t> m)
        : base(std::move(b)), map(std::move(m)) {}

    [[nodiscard]] size_t nRows() const { return base.nRows(); }
    [[nodiscard]] size_t nCols() const { return base.nCols(); }

    const T& operator()(size_t r, size_t c) const { return base(map[r],c); }
};

template <typename T>
struct ShuffleColsView {
    AnyMatrixView<T> base;
    std::vector<size_t> map;
    ShuffleColsView(AnyMatrixView<T> b, std::vector<size_t> m)
        : base(std::move(b)), map(std::move(m)) {}

    [[nodiscard]] size_t nRows() const { return base.nRows(); }
    [[nodiscard]] size_t nCols() const { return base.nCols(); }

    const T& operator()(size_t r, size_t c) const { return base(r,map[c]); }
};

template <typename T>
struct ReshapeView {
    AnyMatrixView<T> base;
    size_t newRows, newCols;
    ReshapeView(AnyMatrixView<T> b, const size_t r, const size_t c)
        : base(std::move(b)), newRows(r), newCols(c) {}

    [[nodiscard]] size_t nRows() const { return newRows; }
    [[nodiscard]] size_t nCols() const { return newCols; }

    const T& operator()(const size_t r, const size_t c) const {
        size_t flat = r*newCols + c;
        size_t orig_r = flat / base.nCols();
        size_t orig_c = flat % base.nCols();
        return base(orig_r, orig_c);
    }
};

template <typename T>
struct TileColsView {
    AnyMatrixView<T> base;
    size_t block;
    TileColsView(AnyMatrixView<T> b, size_t blk)
        : base(std::move(b)), block(blk) {}

    [[nodiscard]] size_t nRows() const { return base.nCols()/block; }
    [[nodiscard]] size_t nCols() const { return base.nRows()*block; }

    const T& operator()(size_t r, size_t c) const {
        size_t orig_row = c/block;
        size_t orig_col = r*block + (c % block);
        return base(orig_row, orig_col);
    }
};

template <typename T>
struct HSplitView {
    AnyMatrixView<T> base;
    size_t parts;

    HSplitView(AnyMatrixView<T> b, size_t p) : base(std::move(b)), parts(p) {
        if (base.nCols() % parts != 0)
            throw std::runtime_error("HSplit requires nCols divisible by parts");
    }

    [[nodiscard]] size_t nRows() const { return base.nRows() * parts; }
    [[nodiscard]] size_t nCols() const { return base.nCols() / parts; }

    const T& operator()(size_t r, size_t c) const {
        size_t orig_r = r % base.nRows();
        size_t block  = r / base.nRows();
        size_t orig_c = c + block * nCols();
        return base(orig_r, orig_c);
    }
};


template <typename T>
struct PadView {
    AnyMatrixView<T> base;
    size_t extraCols;
    size_t extraRows;

    PadView(AnyMatrixView<T> b, size_t ec, size_t er)
        : base(std::move(b)), extraCols(ec), extraRows(er) {}

    [[nodiscard]] size_t nRows() const { return base.nRows() + extraRows; }
    [[nodiscard]] size_t nCols() const { return base.nCols() + extraCols; }

    const T& operator()(size_t r, size_t c) const {
        if (r < base.nRows() && c < base.nCols()) {
            return base(r, c);
        } else {
            static const T zero{};
            return zero;
        }
    }
};


template <typename Target, typename Source>
Target safe_cast(const Source& value) {
    if constexpr (std::is_same_v<Target, Source>) {
        return value;
    } else if constexpr (std::is_integral_v<Target> && std::is_integral_v<Source>) {
        if (value < static_cast<Source>(std::numeric_limits<Target>::min()) ||
            value > static_cast<Source>(std::numeric_limits<Target>::max())) {
            throw std::overflow_error("Value out of range for target integral type");
            }
        return static_cast<Target>(value);
    } else if constexpr (std::is_floating_point_v<Target> && std::is_floating_point_v<Source>) {
        if (!std::isfinite(value)) return static_cast<Target>(value);
        if (value < std::numeric_limits<Target>::lowest() ||
            value > std::numeric_limits<Target>::max()) {
            throw std::overflow_error("Value out of range for target floating type");
            }
        return static_cast<Target>(value);
    } else if constexpr (std::is_floating_point_v<Source> && std::is_integral_v<Target>) {
        if (value < static_cast<Source>(std::numeric_limits<Target>::min()) ||
            value > static_cast<Source>(std::numeric_limits<Target>::max())) {
            throw std::overflow_error("Float value out of range for target integral type");
            }
        return static_cast<Target>(value);
    } else if constexpr (std::is_integral_v<Source> && std::is_floating_point_v<Target>) {
        return static_cast<Target>(value);
    } else {
        static_assert(sizeof(Source) == 0, "Unsupported cast");
    }
    return value;
}

template <typename Target, typename Source>
struct CastView {
    AnyMatrixView<Source> base;

    explicit CastView(AnyMatrixView<Source> b) : base(std::move(b)) {}

    [[nodiscard]] size_t nRows() const { return base.nRows(); }
    [[nodiscard]] size_t nCols() const { return base.nCols(); }

    Target operator()(size_t r, size_t c) const {
        return safe_cast<Target>(base(r, c));
    }
};

template <typename T>
AnyMatrixView<T> transpose(const AnyMatrixView<T>& v) {
    return AnyMatrixView<T>(TransposeView<T>(v));
}

template <typename T>
AnyMatrixView<T> shuffle_rows(const AnyMatrixView<T>& v, std::vector<size_t> m) {
    return AnyMatrixView<T>(ShuffleRowsView<T>(v, std::move(m)));
}

template <typename T>
AnyMatrixView<T> shuffle_cols(const AnyMatrixView<T>& v, std::vector<size_t> m) {
    return AnyMatrixView<T>(ShuffleColsView<T>(v, std::move(m)));
}

template <typename T>
AnyMatrixView<T> reshape(const AnyMatrixView<T>& v, size_t r, size_t c) {
    return AnyMatrixView<T>(ReshapeView<T>(v, r, c));
}

template <typename T>
AnyMatrixView<T> tile_cols(const AnyMatrixView<T>& v, size_t b) {
    return AnyMatrixView<T>(TileColsView<T>(v, b));
}

template <typename T>
AnyMatrixView<T> hsplit(const AnyMatrixView<T>& v, size_t p) {
    return AnyMatrixView<T>(HSplitView<T>(v, p));
}

template <typename T>
AnyMatrixView<T> pad(const AnyMatrixView<T>& v, size_t r, size_t c) {
    return AnyMatrixView<T>(PadView<T>(v, r, c));
}

template <typename Target, typename T>
AnyMatrixView<Target> cast(const AnyMatrixView<T>& v) {
    return AnyMatrixView<Target>(CastView<Target, T>(v));
}
