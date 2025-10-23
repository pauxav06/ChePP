#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <expected>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

template <typename T, std::size_t N>
struct ArrayStack {
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    void push_back(const T& v) {
        assert(m_size < N && "ArrayStack overflow");
        m_data[m_size++] = v;
    }

    reference operator[](size_type i) {
        assert(i < m_size);
        return m_data[i];
    }
    const_reference operator[](size_type i) const {
        assert(i < m_size);
        return m_data[i];
    }

    void clear() noexcept { m_size = 0; }
    void shrink(const size_type n) {
        assert(n <= m_size);
        m_size -= n;
    }

    [[nodiscard]] size_type                  size() const noexcept { return m_size; }
    [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
    [[nodiscard]] bool                       empty() const noexcept { return m_size == 0; }

    [[nodiscard]] iterator       begin() noexcept { return m_data.data(); }
    [[nodiscard]] iterator       end() noexcept { return m_data.data() + m_size; }
    [[nodiscard]] const_iterator begin() const noexcept { return m_data.data(); }
    [[nodiscard]] const_iterator end() const noexcept { return m_data.data() + m_size; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return m_data.data(); }
    [[nodiscard]] const_iterator cend() const noexcept { return m_data.data() + m_size; }

    [[nodiscard]] T*       data() noexcept { return m_data.data(); }
    [[nodiscard]] const T* data() const noexcept { return m_data.data(); }

    [[nodiscard]] reference front() {
        assert(!empty());
        return m_data[0];
    }
    [[nodiscard]] reference back() {
        assert(!empty());
        return m_data[m_size - 1];
    }
    [[nodiscard]] const_reference front() const {
        assert(!empty());
        return m_data[0];
    }
    [[nodiscard]] const_reference back() const {
        assert(!empty());
        return m_data[m_size - 1];
    }

  protected:
    std::array<T, N> m_data{};
    size_type        m_size{0};
};

template <typename EnumT>
struct EnumTraits;

template <typename T>
concept StringRepresentableEnum = requires(T t, std::string s) {
    { EnumTraits<T>::to_string(t) } -> std::convertible_to<std::string>;
    { EnumTraits<T>::from_string(s) } -> std::same_as<std::optional<T>>;
};

// a custom enum of consecutive integers 0 ... N and a NONE value N + 1
// values will wrap like integers if they go passed the NONE value
// f eg, for a Color enum with [WHITE, BLACK, NONE], incrementing will result in the following sequence
// [WHITE, BLACK, NONE, WHITE, BLACK ...], and decrementing [WHITE, NONE, BLACK, WHITE, NONE, BLACK ...]
// Loop limits can hence always be the NONE value
template <typename DerivedT,
          typename UnderlyingT,
          std::size_t COUNT,
          bool        EnableInc        = false,
          bool        EnableArithmetic = false>
    requires(std::is_integral_v<UnderlyingT> && std::is_unsigned_v<UnderlyingT>)
struct EnumBase {
    static constexpr bool EnableInc_v        = EnableInc;
    static constexpr bool EnableArithmetic_v = EnableArithmetic;

    static constexpr std::size_t COUNT_V = COUNT;
    static constexpr std::size_t TOTAL_V = COUNT + 1;

    using ValueT  = UnderlyingT;
    using TraitsT = EnumTraits<DerivedT>;
    using IndexT  = std::size_t;

    // we do not use 0 for NONE because packing is easier if useful values occupy lower bits
    static constexpr ValueT NONE_VALUE = static_cast<ValueT>(COUNT);
    explicit constexpr      operator bool() const noexcept { return m_val != NONE_VALUE; }

    constexpr EnumBase() : m_val(NONE_VALUE){};

    template <typename int_type>
        requires std::is_integral_v<int_type>
    constexpr explicit EnumBase(int_type v) : m_val(static_cast<ValueT>(v % TOTAL_V)) {}

    [[nodiscard]] constexpr ValueT             value() const noexcept { return m_val; }
    [[nodiscard]] constexpr IndexT             index() const noexcept { return static_cast<IndexT>(m_val); }
    [[nodiscard]] constexpr bool               is_none() const noexcept { return m_val == NONE_VALUE; }
    [[nodiscard]] static constexpr std::size_t count() noexcept { return COUNT_V; }
    [[nodiscard]] static constexpr std::size_t total() noexcept { return TOTAL_V; }
    [[nodiscard]] static constexpr DerivedT    none() noexcept { return DerivedT(NONE_VALUE); }

    [[nodiscard]] constexpr std::string to_string() const noexcept
        requires(StringRepresentableEnum<DerivedT>)
    {
        return std::string{TraitsT::to_string(static_cast<DerivedT>(value()))};
    }
    [[nodiscard]] static constexpr std::optional<DerivedT> from_string(const std::string& s)
        requires(StringRepresentableEnum<DerivedT>)
    {
        return TraitsT::from_string(s);
    }

    friend std::ostream& operator<<(std::ostream& os, const EnumBase& e)
        requires(StringRepresentableEnum<DerivedT>)
    {
        return os << e.to_string();
    }

    friend constexpr DerivedT operator+(const DerivedT a, const int rhs) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) + rhs));
    }

    friend constexpr DerivedT operator-(const DerivedT a, const int rhs) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) - rhs));
    }

    friend constexpr DerivedT operator+(const DerivedT a, const DerivedT b) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) + static_cast<int>(b.m_val)));
    }

    friend constexpr DerivedT operator-(const DerivedT a, const DerivedT b) noexcept
        requires(EnableArithmetic)
    {
        return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) - static_cast<int>(b.m_val)));
    }

    friend constexpr DerivedT& operator++(DerivedT& d) noexcept
        requires(EnableInc)
    {
        d.m_val = static_cast<ValueT>(d.m_val + 1);
        return d;
    }

    friend constexpr DerivedT& operator--(DerivedT& d) noexcept
        requires(EnableInc)
    {
        d.m_val = static_cast<ValueT>(d.m_val - 1);
        return d;
    }

    friend constexpr DerivedT operator++(DerivedT& d, int) noexcept
        requires(EnableInc)
    {
        DerivedT tmp = d;
        d.m_val      = static_cast<ValueT>(d.m_val + 1);
        return tmp;
    }

    friend constexpr DerivedT operator--(const DerivedT& d, int) noexcept
        requires(EnableInc)
    {
        DerivedT tmp = d;
        d.m_val      = static_cast<ValueT>(d.m_val - 1);
        return tmp;
    }

    friend constexpr bool operator==(const DerivedT a, const DerivedT b) noexcept { return a.m_val == b.m_val; }
    friend constexpr bool operator!=(const DerivedT a, const DerivedT b) noexcept { return a.m_val != b.m_val; }
    friend constexpr bool operator<(const DerivedT a, const DerivedT b) noexcept { return a.m_val < b.m_val; }
    friend constexpr bool operator<=(const DerivedT a, const DerivedT b) noexcept { return a.m_val <= b.m_val; }
    friend constexpr bool operator>(const DerivedT a, const DerivedT b) noexcept { return a.m_val > b.m_val; }
    friend constexpr bool operator>=(const DerivedT a, const DerivedT b) noexcept { return a.m_val >= b.m_val; }

    struct iterator {
        ValueT m_index;

        constexpr explicit iterator(ValueT idx) : m_index(idx) {}

        constexpr DerivedT operator*() const noexcept { return DerivedT(m_index); }

        constexpr iterator& operator++() noexcept {
            ++m_index;
            return *this;
        }

        constexpr iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b) noexcept {
            return a.m_index == b.m_index;
        }

        friend constexpr bool operator!=(const iterator& a, const iterator& b) noexcept { return !(a == b); }
    };

    struct range {
        iterator m_begin, m_end;

        constexpr iterator begin() const noexcept { return m_begin; }
        constexpr iterator end() const noexcept { return m_end; }
    };

    // excluding none
    [[nodiscard]] static constexpr range all() noexcept { return range{iterator(0), iterator(COUNT_V)}; }

    // public to simplify memcpy/templating semantics
    ValueT m_val = NONE_VALUE;
};

// a type safe wrapper around a multi dimesional array
template <typename T, typename... Enums>
struct EnumArray;

template <typename T, typename Enum>
    requires(std::is_base_of_v<
                 EnumBase<Enum, typename Enum::ValueT, Enum::COUNT_V, Enum::EnableInc_v, Enum::EnableArithmetic_v>,
                 Enum> &&
             Enum::count() > 0)
struct EnumArray<T, Enum> {
    static constexpr std::size_t count = Enum::count();

    using ValueT     = T;
    using indexT     = std::size_t;
    using ContainerT = std::array<ValueT, count>;

    ContainerT data;

    // Access
    constexpr ValueT&       operator[](const Enum e) noexcept { return data[e.index()]; }
    constexpr const ValueT& operator[](const Enum e) const noexcept { return data[e.index()]; }

    constexpr ValueT& at(const Enum e) {
        assert(e.index() < count);
        return data[e.index()];
    }
    [[nodiscard]] constexpr const ValueT& at(const Enum e) const {
        assert(e.index() < count);
        return data[e.index()];
    }

    [[nodiscard]] static constexpr indexT size() noexcept { return count; }

    constexpr auto               begin() noexcept { return data.begin(); }
    constexpr auto               end() noexcept { return data.end(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return data.end(); }
    [[nodiscard]] constexpr auto cbegin() const noexcept { return data.cbegin(); }
    [[nodiscard]] constexpr auto cend() const noexcept { return data.cend(); }

    constexpr void fill(const ValueT& v) { data.fill(v); }

    template <typename Func>
    constexpr void fill_pred(Func&& f) {
        for (std::size_t i = 0; i < count; ++i) {
            Enum e(static_cast<typename Enum::ValueT>(i));
            auto val = f(e);
            static_assert(std::is_same_v<decltype(val), ValueT>, "Predicate must return same type as ValueT");
            data[i] = val;
        }
    }
};

template <typename T, typename FirstEnum, typename... RestEnums>
    requires(std::is_base_of_v<EnumBase<FirstEnum,
                                        typename FirstEnum::ValueT,
                                        FirstEnum::COUNT_V,
                                        FirstEnum::EnableInc_v,
                                        FirstEnum::EnableArithmetic_v>,
                               FirstEnum> &&
             FirstEnum::count() > 0)
struct EnumArray<T, FirstEnum, RestEnums...> {
    static constexpr std::size_t count = FirstEnum::count();
    using SubArrayT                    = EnumArray<T, RestEnums...>;
    using ContainerT                   = std::array<SubArrayT, count>;

    ContainerT data;

    SubArrayT&       operator[](const FirstEnum e) noexcept { return data[e.index()]; }
    const SubArrayT& operator[](const FirstEnum e) const noexcept { return data[e.index()]; }

    SubArrayT& at(const FirstEnum e) {
        assert(e.index() < count);
        return data[e.index()];
    }
    const SubArrayT& at(const FirstEnum e) const noexcept {
        assert(e.index() < count);
        return data[e.index()];
    }

    [[nodiscard]] static constexpr std::size_t size() noexcept { return count; }

    constexpr auto begin() noexcept { return data.begin(); }
    constexpr auto end() noexcept { return data.end(); }
    constexpr auto begin() const noexcept { return data.begin(); }
    constexpr auto end() const noexcept { return data.end(); }
    constexpr auto cbegin() const noexcept { return data.cbegin(); }
    constexpr auto cend() const noexcept { return data.cend(); }

    constexpr void fill(const T& v) {
        for (auto& sub : data) sub.fill(v);
    }

    template <typename Func>
    constexpr void fill_pred(Func&& f) {
        for (std::size_t i = 0; i < count; ++i) {
            FirstEnum e(static_cast<FirstEnum::ValueT>(i));
            data[i].fill_pred([&f, e](auto... restEnums) {
                auto val = f(e, restEnums...);
                static_assert(std::is_same_v<decltype(val), T>, "Predicate must return same type as T");
                return val;
            });
        }
    }
};

struct File : EnumBase<File, uint8_t, 8, true, true> {
    using base = EnumBase;
    using base::EnumBase;
};

template <>
struct EnumTraits<File> {
    static constexpr std::string_view repr{"abcdefgh-"};

    [[nodiscard]] static constexpr std::string to_string(const File f) noexcept {
        return std::string{repr.at(f.index())};
    }

    [[nodiscard]] static constexpr std::optional<File> from_string(const std::string& s) noexcept {
        if (s == "-") return File::none();
        if (s.size() == 1 && s[0] >= 'a' && s[0] <= 'h') return File{s[0] - 'a'};
        return {};
    }
};

constexpr File FILE_A{0};
constexpr File FILE_B{1};
constexpr File FILE_C{2};
constexpr File FILE_D{3};
constexpr File FILE_E{4};
constexpr File FILE_F{5};
constexpr File FILE_G{6};
constexpr File FILE_H{7};
constexpr File NO_FILE{8};

struct Rank : EnumBase<Rank, uint8_t, 8, true, true> {
    using base = EnumBase;
    using base::EnumBase;
};

template <>
struct EnumTraits<Rank> {
    static constexpr std::string_view repr{"12345678-"};

    [[nodiscard]] static constexpr std::string to_string(const Rank r) noexcept {
        return std::string{repr.at(r.index())};
    }

    [[nodiscard]] static constexpr std::optional<Rank> from_string(const std::string_view& sv) {
        if (sv == "-") return Rank{};
        if (sv.size() == 1 && sv[0] >= '1' && sv[0] <= '8') return Rank{sv[0] - '1'};
        return {};
    }
};

constexpr Rank RANK_1{0};
constexpr Rank RANK_2{1};
constexpr Rank RANK_3{2};
constexpr Rank RANK_4{3};
constexpr Rank RANK_5{4};
constexpr Rank RANK_6{5};
constexpr Rank RANK_7{6};
constexpr Rank RANK_8{7};
constexpr Rank NO_RANK{8};

using Coordinates = std::pair<File, Rank>;

struct Square : EnumBase<Square, uint8_t, 64, true, true> {
    using base = EnumBase;
    using base::EnumBase;

    constexpr explicit Square(const Coordinates& coordinates)
        : EnumBase{coordinates.first.index() + (coordinates.second.index() << 3)} {}

    constexpr explicit Square(const File file, const Rank rank) : Square{Coordinates{file, rank}} {}

    [[nodiscard]] constexpr File file() const noexcept { return File{m_val & 7}; }

    [[nodiscard]] constexpr Rank rank() const noexcept { return Rank{m_val >> 3}; }

    [[nodiscard]] constexpr Coordinates coordinates() const noexcept { return {file(), rank()}; }

    [[nodiscard]] constexpr Square flipped_horizontally() const noexcept { return Square{file(), RANK_8 - rank()}; }

    [[nodiscard]] constexpr Square flipped_vertically() const noexcept { return Square{FILE_H - file(), rank()}; }
};

template <>
struct EnumTraits<Square> {
    [[nodiscard]] static constexpr std::string to_string(const Square sq) noexcept {
        if (!sq) return "-";
        return std::string{sq.file().to_string() + sq.rank().to_string()};
    }

    [[nodiscard]] static constexpr std::optional<Square> from_string(const std::string& s) {
        if (s.size() == 1 && s[0] == '-') return Square::none();
        if (s.size() != 2) return {};
        const auto file = File::from_string(s.substr(0, 1));
        const auto rank = Rank::from_string(s.substr(1, 1));
        if (!file || !rank) return std::nullopt;
        return Square{*file, *rank};
    }
};

constexpr Square A1{FILE_A, RANK_1};
constexpr Square A2{FILE_A, RANK_2};
constexpr Square A3{FILE_A, RANK_3};
constexpr Square A4{FILE_A, RANK_4};
constexpr Square A5{FILE_A, RANK_5};
constexpr Square A6{FILE_A, RANK_6};
constexpr Square A7{FILE_A, RANK_7};
constexpr Square A8{FILE_A, RANK_8};

constexpr Square B1{FILE_B, RANK_1};
constexpr Square B2{FILE_B, RANK_2};
constexpr Square B3{FILE_B, RANK_3};
constexpr Square B4{FILE_B, RANK_4};
constexpr Square B5{FILE_B, RANK_5};
constexpr Square B6{FILE_B, RANK_6};
constexpr Square B7{FILE_B, RANK_7};
constexpr Square B8{FILE_B, RANK_8};

constexpr Square C1{FILE_C, RANK_1};
constexpr Square C2{FILE_C, RANK_2};
constexpr Square C3{FILE_C, RANK_3};
constexpr Square C4{FILE_C, RANK_4};
constexpr Square C5{FILE_C, RANK_5};
constexpr Square C6{FILE_C, RANK_6};
constexpr Square C7{FILE_C, RANK_7};
constexpr Square C8{FILE_C, RANK_8};

constexpr Square D1{FILE_D, RANK_1};
constexpr Square D2{FILE_D, RANK_2};
constexpr Square D3{FILE_D, RANK_3};
constexpr Square D4{FILE_D, RANK_4};
constexpr Square D5{FILE_D, RANK_5};
constexpr Square D6{FILE_D, RANK_6};
constexpr Square D7{FILE_D, RANK_7};
constexpr Square D8{FILE_D, RANK_8};

constexpr Square E1{FILE_E, RANK_1};
constexpr Square E2{FILE_E, RANK_2};
constexpr Square E3{FILE_E, RANK_3};
constexpr Square E4{FILE_E, RANK_4};
constexpr Square E5{FILE_E, RANK_5};
constexpr Square E6{FILE_E, RANK_6};
constexpr Square E7{FILE_E, RANK_7};
constexpr Square E8{FILE_E, RANK_8};

constexpr Square F1{FILE_F, RANK_1};
constexpr Square F2{FILE_F, RANK_2};
constexpr Square F3{FILE_F, RANK_3};
constexpr Square F4{FILE_F, RANK_4};
constexpr Square F5{FILE_F, RANK_5};
constexpr Square F6{FILE_F, RANK_6};
constexpr Square F7{FILE_F, RANK_7};
constexpr Square F8{FILE_F, RANK_8};

constexpr Square G1{FILE_G, RANK_1};
constexpr Square G2{FILE_G, RANK_2};
constexpr Square G3{FILE_G, RANK_3};
constexpr Square G4{FILE_G, RANK_4};
constexpr Square G5{FILE_G, RANK_5};
constexpr Square G6{FILE_G, RANK_6};
constexpr Square G7{FILE_G, RANK_7};
constexpr Square G8{FILE_G, RANK_8};

constexpr Square H1{FILE_H, RANK_1};
constexpr Square H2{FILE_H, RANK_2};
constexpr Square H3{FILE_H, RANK_3};
constexpr Square H4{FILE_H, RANK_4};
constexpr Square H5{FILE_H, RANK_5};
constexpr Square H6{FILE_H, RANK_6};
constexpr Square H7{FILE_H, RANK_7};
constexpr Square H8{FILE_H, RANK_8};

constexpr Square NO_SQUARE{64};

struct PieceType : EnumBase<PieceType, uint8_t, 6, true, true> {
    using base = EnumBase;
    using base::EnumBase;

    using PieceValueT = int;
    [[nodiscard]] constexpr PieceValueT piece_value() const;
};

template <>
struct EnumTraits<PieceType> {
    static constexpr std::string_view repr{"pnbrqk-"};

    [[nodiscard]] static constexpr std::string to_string(const PieceType pt) noexcept {
        return std::string{repr.at(pt.index())};
    }

    [[nodiscard]] static constexpr std::optional<PieceType> from_string(const std::string& s) noexcept {
        if (s.size() != 1) return std::nullopt;
        const auto it = repr.find(s[0]);
        if (it == std::string::npos) return std::nullopt;
        return PieceType{it};
    }
};

constexpr PieceType PAWN{0};
constexpr PieceType KNIGHT{1};
constexpr PieceType BISHOP{2};
constexpr PieceType ROOK{3};
constexpr PieceType QUEEN{4};
constexpr PieceType KING{5};
constexpr PieceType NO_PIECE_TYPE{6};

inline EnumArray<PieceType::PieceValueT, PieceType> piece_type_value{100, 300, 325, 500, 900, 20000};

[[nodiscard]] constexpr PieceType::PieceValueT PieceType::piece_value() const {
    return piece_type_value.at(*this);
}

struct Color : EnumBase<Color, uint8_t, 2> {
    using base = EnumBase;
    using base::EnumBase;

    constexpr Color operator~() const noexcept {
        assert(!is_none());
        return Color{m_val ^ 1};
    }
    [[nodiscard]] constexpr Color opposite() const noexcept { return ~(*this); }
    // Prevent misuse of boolean context or !
    constexpr bool     operator!() const     = delete;
    explicit constexpr operator bool() const = delete;
};

template <>
struct EnumTraits<Color> {
    static constexpr std::string_view repr = {"wb-"};

    [[nodiscard]] static constexpr std::string to_string(const Color c) noexcept {
        return std::string{repr.at(c.index())};
    }

    [[nodiscard]] static constexpr std::optional<Color> from_string(const std::string& s) noexcept {
        if (s.size() != 1) return std::nullopt;
        const auto it = repr.find(s[0]);
        if (it == std::string::npos) return std::nullopt;
        return Color{it};
    }
};

constexpr Color WHITE{0};
constexpr Color BLACK{1};
constexpr Color NO_COLOR{2};

struct Piece : EnumBase<Piece, uint8_t, 12, true, true> {
    using base = EnumBase;
    using base::EnumBase;

    explicit constexpr Piece(const Color c, const PieceType pt) : EnumBase{c.value() + (pt.value() << 1)} {}

    [[nodiscard]] constexpr PieceType type() const noexcept { return PieceType{m_val >> 1}; }

    [[nodiscard]] constexpr Color color() const noexcept { return Color{m_val & 1}; }

    using PieceValueT = int;
    [[nodiscard]] constexpr PieceValueT piece_value() const { return piece_type_value.at(this->type()); }
};

template <>
struct EnumTraits<Piece> {
    static constexpr std::string_view          repr{"PpNnBbRrQqKk-"};
    [[nodiscard]] static constexpr std::string to_string(const Piece pt) noexcept {
        return std::string{repr.at(pt.index())};
    }

    [[nodiscard]] static constexpr std::optional<Piece> from_string(const std::string& s) noexcept {
        if (s.size() != 1) return std::nullopt;
        const auto it = repr.find(s[0]);
        if (it == std::string::npos) return std::nullopt;
        return Piece{it};
    }
};

constexpr Piece W_PAWN{WHITE, PAWN};
constexpr Piece W_KNIGHT{WHITE, KNIGHT};
constexpr Piece W_BISHOP{WHITE, BISHOP};
constexpr Piece W_ROOK{WHITE, ROOK};
constexpr Piece W_QUEEN{WHITE, QUEEN};
constexpr Piece W_KING{WHITE, KING};
constexpr Piece B_PAWN{BLACK, PAWN};
constexpr Piece B_KNIGHT{BLACK, KNIGHT};
constexpr Piece B_BISHOP{BLACK, BISHOP};
constexpr Piece B_ROOK{BLACK, ROOK};
constexpr Piece B_QUEEN{BLACK, QUEEN};
constexpr Piece B_KING{BLACK, KING};
constexpr Piece NO_PIECE{12};

// add a to a square value to get a shift in the associated direction
// shift a bitboard by the value to get a shift in the associated direction
// be careful, this on its own wraps around the board edges
enum Direction {
    NORTH        = 8,
    EAST         = 1,
    SOUTH        = -NORTH,
    WEST         = -EAST,
    NORTH_EAST   = NORTH + EAST,
    NORTH_WEST   = NORTH + WEST,
    SOUTH_EAST   = SOUTH + EAST,
    SOUTH_WEST   = SOUTH + WEST,
    NO_DIRECTION = 0
};

constexpr Direction direction_from(const Square a, const Square b) {
    assert(a && b);
    constexpr std::array dir_table{
        SOUTH_WEST, SOUTH, SOUTH_EAST, WEST, NO_DIRECTION, EAST, NORTH_WEST, NORTH, NORTH_EAST};

    const int dr = b.rank().value() - a.rank().value();
    const int df = b.file().value() - a.file().value();
    const int nr = (dr > 0) - (dr < 0);
    const int nf = (df > 0) - (df < 0);

    return dir_table.at((nr + 1) * 3 + (nf + 1));
}

constexpr Square operator+(const Square s, const Direction d) noexcept {
    return Square{s.value() + d};
}
constexpr Square operator-(const Square s, const Direction d) noexcept {
    return Square{s.value() - d};
}

template <Direction d>
constexpr auto inverse_dir = static_cast<Direction>(-d);

template <Color c, Direction d>
constexpr Direction relative_dir = c == WHITE ? d : inverse_dir<d>;

template <Color c, Rank r>
constexpr Rank relative_rank = c == WHITE ? r : RANK_8 - r;

template <Color c, File f>
constexpr File relative_file = f;

template <Color c, Square sq>
constexpr auto relative_square = Square{relative_file<c, sq.file()>, relative_rank<c, sq.rank()>};

enum CastlingSide : uint8_t {
    KINGSIDE  = 0,
    QUEENSIDE = 1,
};

struct CastlingType : EnumBase<CastlingType, uint8_t, 4> {
    using base = EnumBase;
    using base::EnumBase;

    constexpr explicit CastlingType(const Color c, const CastlingSide side) : CastlingType((c.value() << 1) + side) {}

    [[nodiscard]] constexpr Color color() const noexcept {
        assert(!is_none());
        return Color{m_val >> 1};
    }
    [[nodiscard]] constexpr CastlingSide side() const noexcept {
        assert(!is_none());
        return static_cast<CastlingSide>(m_val & 1);
    }

    [[nodiscard]] constexpr ValueT mask() const noexcept {
        assert(!is_none());
        return 1 << m_val;
    }

    [[nodiscard]] constexpr std::pair<Square, Square> king_move() const {
        assert(!is_none());
        constexpr EnumArray<std::pair<Square, Square>, CastlingType> king_moves{
            std::pair{E1, G1}, {E1, C1}, {E8, G8}, {E8, C8}};
        return king_moves.at(*this);
    }
    [[nodiscard]] constexpr std::pair<Square, Square> rook_move() const {
        assert(!is_none());
        constexpr EnumArray<std::pair<Square, Square>, CastlingType> rook_moves{
            std::pair{H1, F1}, {A1, D1}, {H8, F8}, {A8, D8}};
        return rook_moves.at(*this);
    }
};

template <>
struct EnumTraits<CastlingType> {
    static constexpr std::string_view          repr{"KQkq-"};
    [[nodiscard]] static constexpr std::string to_string(const CastlingType pt) noexcept {
        return std::string{repr.at(pt.index())};
    }

    [[nodiscard]] static constexpr std::optional<CastlingType> from_string(const std::string& s) noexcept {
        if (s.size() != 1) return std::nullopt;
        const auto it = repr.find(s[0]);
        if (it == std::string::npos) return std::nullopt;
        return CastlingType{it};
    }
};

constexpr CastlingType WHITE_KINGSIDE{WHITE, KINGSIDE};
constexpr CastlingType BLACK_KINGSIDE{BLACK, KINGSIDE};
constexpr CastlingType WHITE_QUEENSIDE{WHITE, QUEENSIDE};
constexpr CastlingType BLACK_QUEENSIDE{BLACK, QUEENSIDE};
constexpr CastlingType NO_CASTLING_TYPE{4};

struct Move;

struct CastlingRights {
    using MaskT                        = uint8_t;
    static constexpr std::size_t NComb = 16;

    constexpr CastlingRights() : m_mask(0) {}

    template <typename int_type, std::enable_if_t<std::is_integral_v<int_type>, int> = 0>
    constexpr explicit CastlingRights(const int_type mask) noexcept : m_mask(mask & 0b1111) {}

    constexpr CastlingRights(const std::initializer_list<CastlingType> types) noexcept : m_mask(0) {
        for (auto t : types) m_mask |= t.mask();
    }

    constexpr explicit CastlingRights(const Color c) noexcept
        : CastlingRights{CastlingType{c, KINGSIDE}, CastlingType{c, QUEENSIDE}} {}

    static constexpr std::array<std::string_view, NComb> repr = {
        "-", "K", "Q", "KQ", "k", "Kk", "Qk", "KQk", "q", "Kq", "Qq", "KQq", "kq", "Kkq", "Qkq", "KQkq"};

    [[nodiscard]] std::string_view to_string() const noexcept { return repr.at(m_mask); }

    friend std::ostream& operator<<(std::ostream& os, const CastlingRights& cr) {
        os << cr.to_string();
        return os;
    }

    static constexpr CastlingRights all() { return CastlingRights{0b1111}; }
    static constexpr CastlingRights none() { return CastlingRights{0}; }

    friend constexpr bool operator==(const CastlingRights& cr1, const CastlingRights& cr2) noexcept {
        return cr1.m_mask == cr2.m_mask;
    }
    friend constexpr bool operator!=(const CastlingRights& cr1, const CastlingRights& cr2) noexcept {
        return cr1.m_mask != cr2.m_mask;
    }

    [[nodiscard]] constexpr bool has(const CastlingType t) const noexcept { return m_mask & t.mask(); }

    [[nodiscard]] constexpr bool has_any() const noexcept { return m_mask; }
    [[nodiscard]] constexpr bool has_any_color(const Color c) const noexcept {
        return m_mask & CastlingRights(c).m_mask;
    }

    constexpr void add(const CastlingType t) { m_mask |= t.mask(); }
    constexpr void remove(const CastlingType t) { m_mask &= ~t.mask(); }

    constexpr void remove(const CastlingRights other) { m_mask &= ~other.m_mask; }
    constexpr void keep(const CastlingRights other) { m_mask &= other.m_mask; }

    [[nodiscard]] constexpr bool empty() const { return m_mask == 0; }

    [[nodiscard]] constexpr MaskT mask() const { return m_mask; }

    static const EnumArray<CastlingRights, Square> lost_table;

    [[nodiscard]] constexpr CastlingRights lost_from_move(Move move) const;

    [[nodiscard]] static constexpr std::optional<CastlingRights> from_string(const std::string_view& sv) {
        const auto it = std::ranges::find(repr, sv);
        return it == repr.end() ? std::nullopt : std::optional{CastlingRights(std::distance(repr.begin(), it))};
    }

    MaskT m_mask;
};

constexpr EnumArray<CastlingRights, Square> CastlingRights::lost_table = [] {
    EnumArray<CastlingRights, Square> t{};
    t.fill_pred([](const Square sq) {
        return sq == E1   ? CastlingRights{WHITE_KINGSIDE, WHITE_QUEENSIDE}
               : sq == H1 ? CastlingRights{WHITE_KINGSIDE}
               : sq == A1 ? CastlingRights{WHITE_QUEENSIDE}
               : sq == E8 ? CastlingRights{BLACK_KINGSIDE, BLACK_QUEENSIDE}
               : sq == H8 ? CastlingRights{BLACK_KINGSIDE}
               : sq == A8 ? CastlingRights{BLACK_QUEENSIDE}
                          : CastlingRights{};
    });
    return t;
}();

constexpr CastlingRights CASTLING_NONE{0};

constexpr CastlingRights CASTLING_K{WHITE_KINGSIDE};
constexpr CastlingRights CASTLING_Q{WHITE_QUEENSIDE};
constexpr CastlingRights CASTLING_k{BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_q{BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQ{WHITE_KINGSIDE, WHITE_QUEENSIDE};
constexpr CastlingRights CASTLING_Kk{WHITE_KINGSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_Kq{WHITE_KINGSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Qk{WHITE_QUEENSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_Qq{WHITE_QUEENSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_kq{BLACK_KINGSIDE, BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQk{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE};
constexpr CastlingRights CASTLING_KQq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Kkq{WHITE_KINGSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};
constexpr CastlingRights CASTLING_Qkq{WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};

constexpr CastlingRights CASTLING_KQkq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};

struct Result : EnumBase<Result, uint8_t, 3> {
    using base = EnumBase;
    using base::EnumBase;

    explicit constexpr Result(const Color c) : EnumBase(c.value()) {}
};

template <>
struct EnumTraits<Result> {
    static constexpr std::array<std::string_view, Result::total()> repr{"1-0", "0-1", "1/2-1/2", "*"};
    [[nodiscard]] static constexpr std::string                     to_string(const Result r) noexcept {
        return std::string{repr.at(r.index())};
    }

    [[nodiscard]] static constexpr std::optional<Result> from_string(const std::string& s) noexcept {
        const auto it = std::ranges::find(repr, s);
        if (it == repr.end()) return std::nullopt;
        return Result{std::distance(repr.begin(), it)};
    }
};

constexpr Result WIN_WHITE{0};
constexpr Result WIN_BLACK{1};
constexpr Result DRAW{2};
constexpr Result NO_RESULT{3};

namespace bit {
    template <std::unsigned_integral T>
    constexpr int popcount(const T x) noexcept {
        return std::popcount(x);
    }

    template <std::unsigned_integral T>
    constexpr int get_lsb(const T bb) noexcept {
        return std::countr_zero(bb);
    }

    template <std::unsigned_integral T>
    constexpr int get_msb(const T bb) noexcept {
        return std::numeric_limits<T>::digits - 1 - std::countl_zero(bb);
    }

    template <std::unsigned_integral T>
    constexpr int pop_lsb(T& bb) noexcept {
        int n = std::countr_zero(bb);
        bb &= ~(static_cast<T>(1) << n);
        return n;
    }

    template <std::unsigned_integral T>
    constexpr int pop_msb(T& bb) noexcept {
        int n = std::numeric_limits<T>::digits - 1 - std::countl_zero(bb);
        bb &= ~(static_cast<T>(1) << n);
        return n;
    }

    template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
    constexpr T shift_left(const T value, const unsigned shift) {
        // assert(shift < sizeof(T) * 8 && "shift exceeds its bit width");
        return value << shift;
    }

    template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
    constexpr T shift_right(const T value, const unsigned shift) {
        // assert(shift < sizeof(T) * 8 && "shift exceeds its bit width");
        return value >> shift;
    }

} // namespace bit

enum move_type_t : uint16_t { NORMAL = 0, PROMOTION = 1 << 14, EN_PASSANT = 2 << 14, CASTLING = 3 << 14 };

// a move is encoded as an 16 bit unsigned int
// 0-5 bit : to square (square 0 to 63)
// 6-11 bit : from square (square 0 to 63)
// 12-13 bit : promotion piece type (shifted by KNIGHT which is the lowest promotion to fit) or
// castle type 14-15: promotion (1), en passant (2), castling (3)
struct Move {
  public:
    Move() : m_data(0) {}
    constexpr explicit Move(const std::uint16_t d) : m_data(d) {}

    constexpr Move(const Square from, const Square to) : m_data((from.value() << 6) + to.value()) {}

    // to build a move if you already know the type of move
    template <move_type_t T>
    static constexpr Move make(const Square from, const Square to, const PieceType pt = KNIGHT) {
        assert(T != CASTLING);
        return Move{static_cast<uint16_t>(T + ((pt - KNIGHT).value() << 12) + (from.value() << 6) + to.value())};
    }

    template <move_type_t T>
    static constexpr Move make(const Square from, const Square to, const CastlingType c) {
        assert(T == CASTLING);
        return Move(static_cast<uint16_t>(T + (c.value() << 12) + (from.value() << 6) + to.value()));
    }

    // for sanity check
    [[nodiscard]] constexpr bool is_ok() const { return none().m_data != m_data && null().m_data != m_data; }

    // for these two moves from and to are the same
    static constexpr Move null() { return Move(65); }
    static constexpr Move none() { return Move(0); }

    constexpr bool operator==(const Move& m) const { return m_data == m.m_data; }
    constexpr bool operator!=(const Move& m) const { return m_data != m.m_data; }

    constexpr explicit operator bool() const { return m_data != 0; }

    [[nodiscard]] constexpr std::uint16_t raw() const { return m_data; }

    [[nodiscard]] constexpr Square from_sq() const {
        assert(is_ok());
        return Square{(m_data >> 6) & 0x3F};
    }

    [[nodiscard]] constexpr Square to_sq() const {
        assert(is_ok());
        return Square{m_data & 0x3F};
    }

    [[nodiscard]] constexpr move_type_t type_of() const { return static_cast<move_type_t>(m_data & 0b11 << 14); }

    [[nodiscard]] constexpr PieceType promotion_type() const { return PieceType{(m_data >> 12 & 0b11)} + KNIGHT; }

    [[nodiscard]] constexpr CastlingType castling_type() const {
        return static_cast<CastlingType>(m_data >> 12 & 0b11);
    }

    [[nodiscard]] std::string to_string() const {
        std::string s{};
        s.reserve(5);

        s.append(from_sq().to_string());
        s.append(to_sq().to_string());
        if (type_of() == PROMOTION) {
            s.append(promotion_type().to_string());
        }
        return s;
    }

    struct UciInfo {
        const EnumArray<Piece, Square>& pieces;
        Square                          ep_square;
        CastlingRights                  castling_rights;
    };

    static constexpr std::optional<Move> from_uci(const std::string& s, const UciInfo& info);

    friend std::ostream& operator<<(std::ostream& os, const Move mv) { return os << mv.to_string(); }

    struct AlgebraicInfo {
        Piece piece;
        bool  needs_rank{};
        bool  needs_file{};
        bool  is_capture{};
        bool  is_check{};
        bool  is_mate{};
    };

    [[nodiscard]] std::string to_algebraic(const AlgebraicInfo& info) const {
        if (*this == none() || *this == null()) return "--";

        if (type_of() == CASTLING) {
            return castling_type().side() == KINGSIDE ? "O-O" : "O-O-O";
        }

        std::ostringstream oss;

        if (info.piece.type() != PAWN) oss << info.piece.type();

        if (info.needs_file) oss << from_sq().file();
        if (info.needs_rank) oss << from_sq().rank();

        if (info.is_capture) {
            if (info.piece.type() == PAWN && !info.needs_file) oss << from_sq().file();
            oss << "x";
        }

        oss << to_sq();

        if (type_of() == PROMOTION) {
            oss << "=" << Piece{info.piece.color(), promotion_type()}.type();
        }

        if (info.is_check) {
            oss << (info.is_mate ? "#" : "+");
        }

        return oss.str();
    }

    std::uint16_t m_data;
};

template <>
struct std::hash<Move> {
    std::size_t operator()(const Move& m) const noexcept { return std::hash<std::uint16_t>()(m.raw()); }
};

constexpr std::optional<Move> Move::from_uci(const std::string& s, const UciInfo& info) {
    if (!(s.size() == 4 || s.size() == 5)) return std::nullopt;

    const auto from = Square::from_string(s.substr(0, 2));
    const auto to   = Square::from_string(s.substr(2, 2));

    if (!from || !to) return std::nullopt;

    if (s.size() == 5) {
        const auto pt = PieceType::from_string(s.substr(4, 1));
        if (!pt) return std::nullopt;
        return make<PROMOTION>(*from, *to, *pt);
    }

    if (info.pieces.at(*from).type() == PAWN && info.ep_square == *to) {
        return make<EN_PASSANT>(*from, *to);
    }

    if (info.pieces.at(*from).type() == KING) {
        CastlingRights copy = info.castling_rights;
        while (!copy.empty()) {
            auto type = CastlingType{bit::get_lsb(copy.mask())};
            copy.remove(type);
            const auto [k_from, k_to] = type.king_move();
            const auto [r_from, _]    = type.rook_move();
            if (info.pieces.at(r_from) != Piece{type.color(), ROOK}) continue;
            if (info.pieces.at(*from).color() != type.color() || *from != k_from || *to != k_to) continue;
            return make<CASTLING>(k_from, k_to, type);
        }
    }

    return make<NORMAL>(*from, *to);
}

[[nodiscard]] constexpr CastlingRights CastlingRights::lost_from_move(Move move) const {
    return CastlingRights{(lost_table[move.from_sq()].m_mask | lost_table[move.to_sq()].m_mask) & m_mask};
}

constexpr int MAX_PLY = 255;

constexpr int MATE_SCORE    = 32000;
constexpr int INF_SCORE     = 32001;
constexpr int INVALID_SCORE = 32002;

constexpr int MAX_MOVES = 256;

enum Score : int {
    WIN_TB           = 10000,
    LOSS_TB          = -WIN_TB,
    MATE             = 32000,
    MATED            = -MATE,
    MATE_IN_MAX_PLY  = MATE - MAX_PLY,
    MATED_IN_MAX_PLY = -MATE_IN_MAX_PLY,
    INF              = 32001,
    INVALID          = 32002
};

constexpr int mate_in(const int ply) noexcept {
    // better to mate quick -> ply is small
    return MATE - ply;
}

constexpr int mated_in(const int ply) noexcept {
    // better if mate slow -> ply big
    return MATED + ply;
}

constexpr int absolute_eval(const int eval, const Color side) noexcept {
    return side == WHITE ? eval : -eval;
}

constexpr int relative_eval(const int eval, const Color side) noexcept {
    return side == WHITE ? eval : -eval;
}

struct Date {
    int y, m, d;

    [[nodiscard]] std::string to_string() const {
        char buf[11];
        std::sprintf(buf, "%04d.%02d.%02d", y, m, d);
        return buf;
    }
    static bool from_string(const std::string& s, Date& out) {
        if (s.size() != 10) return false;
        int yy, mm, dd;
        if (std::sscanf(s.c_str(), "%d.%d.%d", &yy, &mm, &dd) != 3) return false;

        if (mm < 1 || mm > 12 || dd < 1 || dd > 31) return false;

        out = Date{yy, mm, dd};
        return true;
    }

    static std::optional<Date> from_string(const std::string& s) {
        Date d{};
        if (!from_string(s, d)) return std::nullopt;
        return d;
    }
};

inline constexpr auto start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct MoveList : ArrayStack<Move, MAX_MOVES> {
    using Base = ArrayStack;
    using Base::push_back;
    using Base::operator[];
    using Base::begin;
    using Base::end;
    using Base::size;
};

using MoveScoreT = int;

struct ScoredMove {
    Move       move{};
    MoveScoreT score{};
};

struct ScoredMoveList : ArrayStack<ScoredMove, MoveList::capacity()> {
    using Base = ArrayStack;
    using Base::push_back;
    using Base::operator[];
    using Base::begin;
    using Base::end;
    using Base::size;

    template <typename ScoringFn, std::ranges::range Range>
        requires std::is_invocable_r_v<MoveScoreT, ScoringFn, Move>
    ScoredMoveList(Range&& move_list, ScoringFn&& scoring_fn) {
        for (auto&& move : move_list) {
            push_back({move, scoring_fn(move)});
        }
    }
};

template <typename T>
struct VectorHandle {
    std::vector<T>* ptr;
    uint32_t        index;

    VectorHandle(std::vector<T>* p, const uint32_t i) : ptr(p), index(i) {}
    VectorHandle() : ptr(nullptr), index(0) {}

    T& operator*() const {
        MoveList list;
        return ptr->at(index);
    }

    T* operator->() const { return &ptr->at(index); }

    T& operator()() const { return ptr->at(index); }

    explicit operator bool() const { return ptr != nullptr && index < ptr->size(); }

    bool operator==(const VectorHandle& other) const { return ptr == other.ptr && index == other.index; }
    bool operator!=(const VectorHandle& other) const { return !(*this == other); }
};

template <typename Pred1, typename Pred2>
auto or_predicate(Pred1 a, Pred2 b) {
    return [=](auto&& x) { return a(x) || b(x); };
}

#endif
