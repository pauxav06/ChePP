#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include "format.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>

#include <vector>

#include "expected.h"

namespace chepp {
    using namespace std::literals;

    template <typename T, std::size_t N>
    struct ArrayStack {
        using value_type      = T;
        using size_type       = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference       = T&;
        using const_reference = const T&;
        using iterator        = T*;
        using const_iterator  = const T*;

        void
        push_back(const T& v) {
            assert(m_size < N && "ArrayStack overflow");
            m_data[m_size++] = v;
        }

        reference
        operator[](size_type i) {
            assert(i < m_size);
            return m_data[i];
        }
        const_reference
        operator[](size_type i) const {
            assert(i < m_size);
            return m_data[i];
        }

        void
        clear() noexcept {
            m_size = 0;
        }
        void
        shrink(const size_type n) {
            assert(n <= m_size);
            m_size -= n;
        }

        [[nodiscard]] size_type
        size() const noexcept {
            return m_size;
        }
        [[nodiscard]] static constexpr size_type
        capacity() noexcept {
            return N;
        }
        [[nodiscard]] bool
        empty() const noexcept {
            return m_size == 0;
        }

        [[nodiscard]] iterator
        begin() noexcept {
            return m_data.data();
        }
        [[nodiscard]] iterator
        end() noexcept {
            return m_data.data() + m_size;
        }
        [[nodiscard]] const_iterator
        begin() const noexcept {
            return m_data.data();
        }
        [[nodiscard]] const_iterator
        end() const noexcept {
            return m_data.data() + m_size;
        }
        [[nodiscard]] const_iterator
        cbegin() const noexcept {
            return m_data.data();
        }
        [[nodiscard]] const_iterator
        cend() const noexcept {
            return m_data.data() + m_size;
        }

        [[nodiscard]] T*
        data() noexcept {
            return m_data.data();
        }
        [[nodiscard]] const T*
        data() const noexcept {
            return m_data.data();
        }

        [[nodiscard]] reference
        front() {
            assert(!empty());
            return m_data[0];
        }
        [[nodiscard]] reference
        back() {
            assert(!empty());
            return m_data[m_size - 1];
        }
        [[nodiscard]] const_reference
        front() const {
            assert(!empty());
            return m_data[0];
        }
        [[nodiscard]] const_reference
        back() const {
            assert(!empty());
            return m_data[m_size - 1];
        }

      protected:
        std::array<T, N> m_data{};
        size_type        m_size{0};
    };

    template <typename Derived>
    struct Printable {
        friend std::ostream&
        operator<<(std::ostream& os, const Derived& e) {
            return os << e.to_string();
        }
    };

    template <typename Derived>
    struct Parsable {
        friend std::istream&
        operator>>(std::istream& is, Derived& obj) {
            std::string buffer;

            if (!(is >> buffer)) {
                return is;
            }

            auto result = Derived::parse(buffer);

            if (result) {
                obj = std::move(*result);
            } else {
                is.setstate(std::ios::failbit);
            }

            return is;
        }
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
    struct EnumBase : Printable<DerivedT>, Parsable<DerivedT> {
        static constexpr bool EnableInc_v        = EnableInc;
        static constexpr bool EnableArithmetic_v = EnableArithmetic;

        static constexpr std::size_t COUNT_V = COUNT;
        static constexpr std::size_t TOTAL_V = COUNT + 1;

        using ValueT = UnderlyingT;
        using IndexT = std::size_t;

        // we do not use 0 for NONE because packing is easier if useful values occupy lower bits
        static constexpr ValueT NONE_VALUE = static_cast<ValueT>(COUNT);
        explicit constexpr operator bool() const noexcept {
            return m_val != NONE_VALUE;
        }

        constexpr EnumBase() : m_val(NONE_VALUE){};

        template <typename int_type>
            requires std::is_integral_v<int_type>
        constexpr explicit EnumBase(int_type v) : m_val(static_cast<ValueT>(v) % TOTAL_V) {
        }

        [[nodiscard]] constexpr ValueT
        value() const noexcept {
            return m_val;
        }
        [[nodiscard]] constexpr IndexT
        index() const noexcept {
            return static_cast<IndexT>(m_val);
        }
        [[nodiscard]] constexpr bool
        is_none() const noexcept {
            return m_val == NONE_VALUE;
        }
        [[nodiscard]] static constexpr std::size_t
        count() noexcept {
            return COUNT_V;
        }
        [[nodiscard]] static constexpr std::size_t
        total() noexcept {
            return TOTAL_V;
        }
        [[nodiscard]] static constexpr DerivedT
        none() noexcept {
            return DerivedT(NONE_VALUE);
        }

        friend constexpr DerivedT
        operator+(const DerivedT a, const int rhs) noexcept
            requires(EnableArithmetic)
        {
            return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) + rhs));
        }

        friend constexpr DerivedT
        operator-(const DerivedT a, const int rhs) noexcept
            requires(EnableArithmetic)
        {
            return DerivedT(static_cast<ValueT>(static_cast<int>(a.m_val) - rhs));
        }

        friend constexpr DerivedT
        operator+(const DerivedT a, const DerivedT b) noexcept
            requires(EnableArithmetic)
        {
            return DerivedT(static_cast<ValueT>(a.m_val + b.m_val));
        }

        friend constexpr DerivedT
        operator-(const DerivedT a, const DerivedT b) noexcept
            requires(EnableArithmetic)
        {
            return DerivedT(static_cast<ValueT>(a.m_val - b.m_val));
        }

        friend constexpr DerivedT&
        operator++(DerivedT& d) noexcept
            requires(EnableInc)
        {
            d.m_val = static_cast<ValueT>(d.m_val + 1);
            return d;
        }

        friend constexpr DerivedT&
        operator--(DerivedT& d) noexcept
            requires(EnableInc)
        {
            d.m_val = static_cast<ValueT>(d.m_val - 1);
            return d;
        }

        friend constexpr DerivedT
        operator++(DerivedT& d, int) noexcept
            requires(EnableInc)
        {
            DerivedT tmp = d;
            d.m_val      = static_cast<ValueT>(d.m_val + 1);
            return tmp;
        }

        friend constexpr DerivedT
        operator--(const DerivedT& d, int) noexcept
            requires(EnableInc)
        {
            DerivedT tmp = d;
            d.m_val      = static_cast<ValueT>(d.m_val - 1);
            return tmp;
        }

        friend constexpr bool
        operator==(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val == b.m_val;
        }
        friend constexpr bool
        operator!=(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val != b.m_val;
        }
        friend constexpr bool
        operator<(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val < b.m_val;
        }
        friend constexpr bool
        operator<=(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val <= b.m_val;
        }
        friend constexpr bool
        operator>(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val > b.m_val;
        }
        friend constexpr bool
        operator>=(const DerivedT a, const DerivedT b) noexcept {
            return a.m_val >= b.m_val;
        }

        struct iterator {
            ValueT m_index;

            constexpr explicit iterator(ValueT idx) : m_index(idx) {
            }

            constexpr DerivedT
            operator*() const noexcept {
                return DerivedT(m_index);
            }

            constexpr iterator&
            operator++() noexcept {
                ++m_index;
                return *this;
            }

            constexpr iterator
            operator++(int) noexcept {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend constexpr bool
            operator==(const iterator& a, const iterator& b) noexcept {
                return a.m_index == b.m_index;
            }

            friend constexpr bool
            operator!=(const iterator& a, const iterator& b) noexcept {
                return !(a == b);
            }
        };

        struct range {
            iterator m_begin, m_end;

            constexpr iterator
            begin() const noexcept {
                return m_begin;
            }
            constexpr iterator
            end() const noexcept {
                return m_end;
            }
        };

        // excluding none
        [[nodiscard]] static constexpr range
        all() noexcept {
            return range{iterator(0), iterator(COUNT_V)};
        }

        template <typename F>
        static constexpr void
        constepr_for(F&& f) {
            constepr_for<0, COUNT_V>([&](auto i) { return std::forward<F>(f)(DerivedT(i)); });
        }

        // public to simplify memcpy/templating semantics
        ValueT m_val = NONE_VALUE;
    };
} // namespace chepp

template <typename T>
    requires std::is_base_of_v<chepp::Printable<T>, T>
struct fmt::formatter<T> : fmt::formatter<std::string> {
    auto
    format(T value, fmt::format_context& ctx) const {
        return std::ranges::copy(std::move(value).to_string(), ctx.out()).out;
    }
};

struct constexpr_in_place_t {};
inline constexpr constexpr_in_place_t constexpr_in_place{};

namespace chepp {
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

        ContainerT data{};

        constexpr EnumArray()                     = default;
        constexpr EnumArray(const EnumArray&)     = default;
        constexpr EnumArray(EnumArray&&) noexcept = default;
        constexpr EnumArray&
        operator=(const EnumArray&) = default;
        constexpr EnumArray&
        operator=(EnumArray&&) noexcept = default;

        constexpr EnumArray(std::initializer_list<T> init) {
            assert(init.size() == count && "Initializer list must match EnumArray size");
            std::copy(init.begin(), init.end(), data.begin());
        }

        template <typename F>
            requires std::is_invocable_r_v<T, F, Enum>
        constexpr explicit EnumArray(std::in_place_t, F&& f) {
            for (std::size_t i = 0; i < count; ++i) {
                Enum e(static_cast<typename Enum::ValueT>(i));
                std::construct_at(&data[i], f(e));
            }
        }

        template <typename F>
            requires std::is_invocable_r_v<T, F, std::integral_constant<Enum, Enum{0}>>
        constexpr explicit EnumArray(constexpr_in_place_t, F&& f) {
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((std::construct_at(&data[Is], f(std::integral_constant<Enum, Enum{Is}>{}))), ...);
            }(std::make_index_sequence<count>{});
        }

        constexpr ValueT&
        operator[](const Enum e) noexcept {
            return data[e.index()];
        }
        constexpr const ValueT&
        operator[](const Enum e) const noexcept {
            return data[e.index()];
        }

        constexpr ValueT&
        at(const Enum e) {
            assert(e.index() < count);
            return data[e.index()];
        }
        [[nodiscard]] constexpr const ValueT&
        at(const Enum e) const {
            assert(e.index() < count);
            return data[e.index()];
        }

        [[nodiscard]] static constexpr indexT
        size() noexcept {
            return count;
        }

        constexpr auto
        begin() noexcept {
            return data.begin();
        }
        constexpr auto
        end() noexcept {
            return data.end();
        }
        [[nodiscard]] constexpr auto
        begin() const noexcept {
            return data.begin();
        }
        [[nodiscard]] constexpr auto
        end() const noexcept {
            return data.end();
        }
        [[nodiscard]] constexpr auto
        cbegin() const noexcept {
            return data.cbegin();
        }
        [[nodiscard]] constexpr auto
        cend() const noexcept {
            return data.cend();
        }

        constexpr void
        fill(const ValueT& v) {
            data.fill(v);
        }

        template <typename Func>
        constexpr void
        fill_pred(Func&& f) {
            for (std::size_t i = 0; i < count; ++i) {
                Enum e(static_cast<Enum::ValueT>(i));
                std::construct_at(&data[i], f(e));
            }
        }

        template <std::size_t... Is, typename F>
        static constexpr EnumArray
        make_impl(std::index_sequence<Is...>, F&& f) {
            return {(std::forward<F>(f)(Enum(Is)))...};
        }

        template <typename F>
        static constexpr EnumArray
        make(F&& f) {
            return make_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
        }
    };
} // namespace chepp

template <typename T, typename Enum>
inline constexpr bool std::ranges::enable_borrowed_range<chepp::EnumArray<T, Enum>> = true;

namespace chepp {
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

        constexpr EnumArray()                     = default;
        constexpr EnumArray(const EnumArray&)     = default;
        constexpr EnumArray(EnumArray&&) noexcept = default;
        constexpr EnumArray&
        operator=(const EnumArray&) = default;
        constexpr EnumArray&
        operator=(EnumArray&&) noexcept = default;

        constexpr EnumArray(std::initializer_list<SubArrayT> init) {
            assert(init.size() == count && "Initializer list must match EnumArray size");
            std::copy(init.begin(), init.end(), data.begin());
        }

        template <typename F>
        constexpr explicit EnumArray(std::in_place_t, F&& f) {
            for (std::size_t i = 0; i < count; ++i) {
                FirstEnum e(static_cast<typename FirstEnum::ValueT>(i));
                std::construct_at(
                    &data[i], std::in_place, [&, e](auto... restEnums) { return std::forward<F>(f)(e, restEnums...); });
            }
        }

        template <typename F>
        constexpr explicit EnumArray(constexpr_in_place_t, F&& f) {
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((std::construct_at(&data[Is],
                                    SubArrayT(constexpr_in_place,
                                              [&](auto... restEnums) {
                                                  return f(std::integral_constant<FirstEnum, FirstEnum{Is}>{},
                                                           restEnums...);
                                              }))),
                 ...);
            }(std::make_index_sequence<count>{});
        }

        constexpr SubArrayT&
        operator[](const FirstEnum e) noexcept {
            return data[e.index()];
        }
        constexpr const SubArrayT&
        operator[](const FirstEnum e) const noexcept {
            return data[e.index()];
        }

        constexpr SubArrayT&
        at(const FirstEnum e) {
            assert(e.index() < count);
            return data[e.index()];
        }
        [[nodiscard]] constexpr const SubArrayT&
        at(const FirstEnum e) const noexcept {
            assert(e.index() < count);
            return data[e.index()];
        }

        [[nodiscard]] static constexpr std::size_t
        size() noexcept {
            return count;
        }

        constexpr auto
        begin() noexcept {
            return data.begin();
        }
        constexpr auto
        end() noexcept {
            return data.end();
        }
        [[nodiscard]] constexpr auto
        begin() const noexcept {
            return data.begin();
        }
        [[nodiscard]] constexpr auto
        end() const noexcept {
            return data.end();
        }
        [[nodiscard]] constexpr auto
        cbegin() const noexcept {
            return data.cbegin();
        }
        [[nodiscard]] constexpr auto
        cend() const noexcept {
            return data.cend();
        }

        constexpr void
        fill(const T& v) {
            for (auto& sub : data) sub.fill(v);
        }

        template <typename Func>
        constexpr void
        fill_pred(Func&& f) {
            for (std::size_t i = 0; i < count; ++i) {
                FirstEnum e(static_cast<FirstEnum::ValueT>(i));
                data[i].fill_pred([&f, e](auto... restEnums) {
                    auto val = f(e, restEnums...);
                    static_assert(std::is_same_v<decltype(val), T>, "Predicate must return same type as T");
                    return val;
                });
            }
        }

        template <std::size_t... Is, typename F>
        static constexpr EnumArray
        make_impl(std::index_sequence<Is...>, F&& f) {
            return EnumArray{
                (SubArrayT::make([&, e = FirstEnum(Is)](auto... restEnums) { return f(e, restEnums...); }))...};
        }

        template <typename F>
        static constexpr EnumArray
        make(F&& f) {
            return make_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
        }
    };

    struct File : EnumBase<File, uint8_t, 8, true, true> {
        using base = EnumBase;
        using base::EnumBase;

        static constexpr std::string_view repr{"abcdefgh-"};

        [[nodiscard]] std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<File, std::string>
        from_string(const std::string_view s) noexcept {
            if (s == "-") {
                return File::none();
            }
            if (s.size() == 1 && s[0] >= 'a' && s[0] <= 'h') {
                return File{s[0] - 'a'};
            }
            return tl::unexpected{fmt::format("unknown file {}", s)};
        }
    };

    inline constexpr File FILE_A{0};
    inline constexpr File FILE_B{1};
    inline constexpr File FILE_C{2};
    inline constexpr File FILE_D{3};
    inline constexpr File FILE_E{4};
    inline constexpr File FILE_F{5};
    inline constexpr File FILE_G{6};
    inline constexpr File FILE_H{7};
    inline constexpr File NO_FILE{8};

    struct Rank : EnumBase<Rank, uint8_t, 8, true, true> {
        using base = EnumBase;
        using base::EnumBase;

        static constexpr std::string_view repr{"12345678-"};

        [[nodiscard]] constexpr std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<Rank, std::string>
        from_string(const std::string_view sv) {
            if (sv == "-") {
                return Rank{};
            }
            if (sv.size() == 1 && sv[0] >= '1' && sv[0] <= '8') {
                return Rank{sv[0] - '1'};
            }
            return tl::unexpected{fmt::format("unknown rank {}", sv)};
        }
    };

    inline constexpr Rank RANK_1{0};
    inline constexpr Rank RANK_2{1};
    inline constexpr Rank RANK_3{2};
    inline constexpr Rank RANK_4{3};
    inline constexpr Rank RANK_5{4};
    inline constexpr Rank RANK_6{5};
    inline constexpr Rank RANK_7{6};
    inline constexpr Rank RANK_8{7};
    inline constexpr Rank NO_RANK{8};

    using Coordinates = std::pair<File, Rank>;

    struct Square : EnumBase<Square, uint8_t, 64, true, true> {
        using base = EnumBase;
        using base::EnumBase;

        constexpr explicit Square(const Coordinates& coordinates)
            : EnumBase{coordinates.first.index() + (coordinates.second.index() << 3)} {
        }

        constexpr explicit Square(const File file, const Rank rank) : Square{Coordinates{file, rank}} {
        }

        [[nodiscard]] constexpr File
        file() const noexcept {
            return File{m_val & 7};
        }

        [[nodiscard]] constexpr Rank
        rank() const noexcept {
            return Rank{m_val >> 3};
        }

        [[nodiscard]] constexpr Coordinates
        coordinates() const noexcept {
            return {file(), rank()};
        }

        [[nodiscard]] constexpr Square
        flipped_horizontally() const noexcept {
            return Square{file(), RANK_8 - rank()};
        }

        [[nodiscard]] constexpr Square
        flipped_vertically() const noexcept {
            return Square{FILE_H - file(), rank()};
        }

        [[nodiscard]] constexpr std::string
        to_string() const noexcept {
            if (is_none()) {
                return "-";
            }
            return file().to_string() + rank().to_string();
        }

        using parse_t = tl::expected<Square, std::string>;
        [[nodiscard]] static parse_t
        from_string(const std::string_view s) {
            if (s.size() == 1 && s[0] == '-') {
                return parse_t{tl::in_place};
            }
            if (s.size() != 2) {
                return parse_t{tl::unexpect, fmt::format("unknown square {}", s)};
            }
            const auto file = File::from_string(s.substr(0, 1));
            const auto rank = Rank::from_string(s.substr(1, 1));
            if (!file) {
                return parse_t{tl::unexpect, fmt::format("could not build square: {}", file.error())};
            } else if (!rank) {
                return parse_t{tl::unexpect, fmt::format("could not build square: {}", rank.error())};
            }
            return parse_t{tl::in_place, *file, *rank};
        }
    };

    inline constexpr Square A1{FILE_A, RANK_1};
    inline constexpr Square A2{FILE_A, RANK_2};
    inline constexpr Square A3{FILE_A, RANK_3};
    inline constexpr Square A4{FILE_A, RANK_4};
    inline constexpr Square A5{FILE_A, RANK_5};
    inline constexpr Square A6{FILE_A, RANK_6};
    inline constexpr Square A7{FILE_A, RANK_7};
    inline constexpr Square A8{FILE_A, RANK_8};

    inline constexpr Square B1{FILE_B, RANK_1};
    inline constexpr Square B2{FILE_B, RANK_2};
    inline constexpr Square B3{FILE_B, RANK_3};
    inline constexpr Square B4{FILE_B, RANK_4};
    inline constexpr Square B5{FILE_B, RANK_5};
    inline constexpr Square B6{FILE_B, RANK_6};
    inline constexpr Square B7{FILE_B, RANK_7};
    inline constexpr Square B8{FILE_B, RANK_8};

    inline constexpr Square C1{FILE_C, RANK_1};
    inline constexpr Square C2{FILE_C, RANK_2};
    inline constexpr Square C3{FILE_C, RANK_3};
    inline constexpr Square C4{FILE_C, RANK_4};
    inline constexpr Square C5{FILE_C, RANK_5};
    inline constexpr Square C6{FILE_C, RANK_6};
    inline constexpr Square C7{FILE_C, RANK_7};
    inline constexpr Square C8{FILE_C, RANK_8};

    inline constexpr Square D1{FILE_D, RANK_1};
    inline constexpr Square D2{FILE_D, RANK_2};
    inline constexpr Square D3{FILE_D, RANK_3};
    inline constexpr Square D4{FILE_D, RANK_4};
    inline constexpr Square D5{FILE_D, RANK_5};
    inline constexpr Square D6{FILE_D, RANK_6};
    inline constexpr Square D7{FILE_D, RANK_7};
    inline constexpr Square D8{FILE_D, RANK_8};

    inline constexpr Square E1{FILE_E, RANK_1};
    inline constexpr Square E2{FILE_E, RANK_2};
    inline constexpr Square E3{FILE_E, RANK_3};
    inline constexpr Square E4{FILE_E, RANK_4};
    inline constexpr Square E5{FILE_E, RANK_5};
    inline constexpr Square E6{FILE_E, RANK_6};
    inline constexpr Square E7{FILE_E, RANK_7};
    inline constexpr Square E8{FILE_E, RANK_8};

    inline constexpr Square F1{FILE_F, RANK_1};
    inline constexpr Square F2{FILE_F, RANK_2};
    inline constexpr Square F3{FILE_F, RANK_3};
    inline constexpr Square F4{FILE_F, RANK_4};
    inline constexpr Square F5{FILE_F, RANK_5};
    inline constexpr Square F6{FILE_F, RANK_6};
    inline constexpr Square F7{FILE_F, RANK_7};
    inline constexpr Square F8{FILE_F, RANK_8};

    inline constexpr Square G1{FILE_G, RANK_1};
    inline constexpr Square G2{FILE_G, RANK_2};
    inline constexpr Square G3{FILE_G, RANK_3};
    inline constexpr Square G4{FILE_G, RANK_4};
    inline constexpr Square G5{FILE_G, RANK_5};
    inline constexpr Square G6{FILE_G, RANK_6};
    inline constexpr Square G7{FILE_G, RANK_7};
    inline constexpr Square G8{FILE_G, RANK_8};

    inline constexpr Square H1{FILE_H, RANK_1};
    inline constexpr Square H2{FILE_H, RANK_2};
    inline constexpr Square H3{FILE_H, RANK_3};
    inline constexpr Square H4{FILE_H, RANK_4};
    inline constexpr Square H5{FILE_H, RANK_5};
    inline constexpr Square H6{FILE_H, RANK_6};
    inline constexpr Square H7{FILE_H, RANK_7};
    inline constexpr Square H8{FILE_H, RANK_8};
    inline constexpr Square NO_SQUARE{64};

    struct PieceType : EnumBase<PieceType, uint8_t, 6, true, true> {
        using base = EnumBase;
        using base::EnumBase;

        using PieceValueT = int;
        [[nodiscard]] constexpr PieceValueT
        piece_value() const;

        static constexpr std::string_view repr{"pnbrqk-"};

        [[nodiscard]] std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<PieceType, std::string>
        from_string(const std::string_view s) noexcept {
            if (s.size() != 1) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            };
            const auto it = repr.find(s[0]);
            if (it == std::string::npos) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            }
            return PieceType{it};
        }
    };

    inline constexpr PieceType PAWN{0};
    inline constexpr PieceType KNIGHT{1};
    inline constexpr PieceType BISHOP{2};
    inline constexpr PieceType ROOK{3};
    inline constexpr PieceType QUEEN{4};
    inline constexpr PieceType KING{5};
    inline constexpr PieceType NO_PIECE_TYPE{6};

    inline constexpr EnumArray<PieceType::PieceValueT, PieceType> piece_type_value{100, 300, 325, 500, 900, 20000};

    [[nodiscard]] constexpr PieceType::PieceValueT
    PieceType::piece_value() const {
        return piece_type_value.at(*this);
    }

    struct Color : EnumBase<Color, uint8_t, 2> {
        using base = EnumBase;
        using base::EnumBase;

        constexpr Color
        operator~() const noexcept {
            assert(!is_none());
            return Color{m_val ^ 1};
        }
        [[nodiscard]] constexpr Color
        opposite() const noexcept {
            return ~(*this);
        }
        // Prevent misuse of boolean context or !
        constexpr bool
        operator!() const                        = delete;
        explicit constexpr operator bool() const = delete;

        static constexpr std::string_view repr = {"wb-"};

        [[nodiscard]] constexpr std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<Color, std::string>
        from_string(const std::string_view s) noexcept {
            if (s.size() != 1) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            }
            const auto it = repr.find(s[0]);
            if (it == std::string::npos) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            }
            return Color{it};
        }
    };

    inline constexpr Color WHITE{0};
    inline constexpr Color BLACK{1};
    inline constexpr Color NO_COLOR{2};

    struct Piece : EnumBase<Piece, uint8_t, 12, true, true> {
        using base = EnumBase;
        using base::EnumBase;

        explicit constexpr Piece(const Color c, const PieceType pt) : EnumBase{c.value() + (pt.value() << 1)} {
        }

        [[nodiscard]] constexpr PieceType
        type() const noexcept {
            return PieceType{m_val >> 1};
        }

        [[nodiscard]] constexpr Color
        color() const noexcept {
            return Color{m_val & 1};
        }

        using PieceValueT = int;
        [[nodiscard]] constexpr PieceValueT
        piece_value() const {
            return piece_type_value.at(this->type());
        }

        static constexpr std::string_view repr{"PpNnBbRrQqKk-"};
        [[nodiscard]] std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<Piece, std::string>
        from_string(const std::string_view s) noexcept {
            if (s.size() != 1) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            }
            if (const auto it = repr.find(s[0]); it != std::string::npos) {
                return Piece{it};
            }
            return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
        }
    };

    inline constexpr Piece W_PAWN{WHITE, PAWN};
    inline constexpr Piece W_KNIGHT{WHITE, KNIGHT};
    inline constexpr Piece W_BISHOP{WHITE, BISHOP};
    inline constexpr Piece W_ROOK{WHITE, ROOK};
    inline constexpr Piece W_QUEEN{WHITE, QUEEN};
    inline constexpr Piece W_KING{WHITE, KING};
    inline constexpr Piece B_PAWN{BLACK, PAWN};
    inline constexpr Piece B_KNIGHT{BLACK, KNIGHT};
    inline constexpr Piece B_BISHOP{BLACK, BISHOP};
    inline constexpr Piece B_ROOK{BLACK, ROOK};
    inline constexpr Piece B_QUEEN{BLACK, QUEEN};
    inline constexpr Piece B_KING{BLACK, KING};
    inline constexpr Piece NO_PIECE{12};

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

    inline constexpr std::array dir_table{
        SOUTH_WEST, SOUTH, SOUTH_EAST, WEST, NO_DIRECTION, EAST, NORTH_WEST, NORTH, NORTH_EAST};
    constexpr Direction
    direction_from(const Square a, const Square b) {
        assert(a && b);

        const auto dr = b.rank().value() - a.rank().value();
        const auto df = b.file().value() - a.file().value();
        const auto nr = (dr > 0) - (dr < 0);
        const auto nf = (df > 0) - (df < 0);

        const auto idx = (nr + 1) * 3 + (nf + 1);
        assert(idx >= 0);
        return dir_table.at(static_cast<std::size_t>(idx));
    }

    constexpr Square
    operator+(const Square s, const Direction d) noexcept {
        return Square{s.value() + d};
    }
    constexpr Square
    operator-(const Square s, const Direction d) noexcept {
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

        constexpr explicit CastlingType(const Color c, const CastlingSide side)
            : CastlingType((c.value() << 1) + side) {
        }

        [[nodiscard]] constexpr Color
        color() const noexcept {
            assert(!is_none());
            return Color{m_val >> 1};
        }
        [[nodiscard]] constexpr CastlingSide
        side() const noexcept {
            assert(!is_none());
            return static_cast<CastlingSide>(m_val & 1);
        }

        [[nodiscard]] constexpr ValueT
        mask() const noexcept {
            assert(!is_none());
            return static_cast<ValueT>(1 << m_val);
        }

        static const EnumArray<std::pair<Square, Square>, CastlingType> king_moves;
        [[nodiscard]] constexpr std::pair<Square, Square>
        king_move() const {
            assert(!is_none());
            return king_moves.at(*this);
        }
        static const EnumArray<std::pair<Square, Square>, CastlingType> rook_moves;
        [[nodiscard]] constexpr std::pair<Square, Square>
        rook_move() const {
            assert(!is_none());
            return rook_moves.at(*this);
        }

        static constexpr std::string_view repr{"KQkq-"};

        [[nodiscard]] std::string
        to_string() const noexcept {
            return std::string{repr.at(index())};
        }

        [[nodiscard]] static tl::expected<CastlingType, std::string>
        from_string(const std::string_view s) noexcept {
            if (s.size() != 1) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            }
            const auto it = repr.find(s[0]);
            if (it == std::string::npos) {
                return tl::unexpected{fmt::format("expected char in {} but found {}", repr, s)};
            };
            return CastlingType{it};
        }
    };

    inline constexpr EnumArray<std::pair<Square, Square>, CastlingType> CastlingType::king_moves{
        std::pair{E1, G1}, {E1, C1}, {E8, G8}, {E8, C8}};
    inline constexpr EnumArray<std::pair<Square, Square>, CastlingType> CastlingType::rook_moves{
        std::pair{H1, F1}, {A1, D1}, {H8, F8}, {A8, D8}};

    inline constexpr CastlingType WHITE_KINGSIDE{WHITE, KINGSIDE};
    inline constexpr CastlingType BLACK_KINGSIDE{BLACK, KINGSIDE};
    inline constexpr CastlingType WHITE_QUEENSIDE{WHITE, QUEENSIDE};
    inline constexpr CastlingType BLACK_QUEENSIDE{BLACK, QUEENSIDE};
    inline constexpr CastlingType NO_CASTLING_TYPE{4};

    struct Move;

    struct CastlingRights : Printable<CastlingRights>, Parsable<CastlingRights> {
        using MaskT                        = uint8_t;
        static constexpr std::size_t NComb = 16;

        constexpr CastlingRights() : m_mask(0) {
        }

        template <typename int_type, std::enable_if_t<std::is_integral_v<int_type>, int> = 0>
        constexpr explicit CastlingRights(const int_type mask) noexcept : m_mask(mask & 0b1111) {
        }

        constexpr CastlingRights(const std::initializer_list<CastlingType> types) noexcept : m_mask(0) {
            for (auto t : types) m_mask |= t.mask();
        }

        constexpr explicit CastlingRights(const Color c) noexcept
            : CastlingRights{CastlingType{c, KINGSIDE}, CastlingType{c, QUEENSIDE}} {
        }

        static constexpr CastlingRights
        all() {
            return CastlingRights{0b1111};
        }
        static constexpr CastlingRights
        none() {
            return CastlingRights{0};
        }

        friend constexpr bool
        operator==(const CastlingRights& cr1, const CastlingRights& cr2) noexcept {
            return cr1.m_mask == cr2.m_mask;
        }
        friend constexpr bool
        operator!=(const CastlingRights& cr1, const CastlingRights& cr2) noexcept {
            return cr1.m_mask != cr2.m_mask;
        }

        [[nodiscard]] constexpr bool
        has(const CastlingType t) const noexcept {
            return m_mask & t.mask();
        }

        [[nodiscard]] constexpr bool
        has_any() const noexcept {
            return m_mask;
        }
        [[nodiscard]] constexpr bool
        has_any_color(const Color c) const noexcept {
            return m_mask & CastlingRights(c).m_mask;
        }

        constexpr void
        add(const CastlingType t) {
            m_mask |= t.mask();
        }
        constexpr void
        remove(const CastlingType t) {
            m_mask &= static_cast<MaskT>(~t.mask());
        }

        constexpr void
        remove(const CastlingRights other) {
            m_mask &= ~other.m_mask;
        }
        constexpr void
        keep(const CastlingRights other) {
            m_mask &= other.m_mask;
        }

        [[nodiscard]] constexpr bool
        empty() const {
            return m_mask == 0;
        }

        [[nodiscard]] constexpr MaskT
        mask() const {
            return m_mask;
        }

        static const EnumArray<CastlingRights, Square> lost_table;

        [[nodiscard]] constexpr CastlingRights
        lost_from_move(Move move) const;

        static constexpr std::array<std::string_view, NComb> repr = {
            "-", "K", "Q", "KQ", "k", "Kk", "Qk", "KQk", "q", "Kq", "Qq", "KQq", "kq", "Kkq", "Qkq", "KQkq"};

        [[nodiscard]] std::string_view
        to_string() const noexcept {
            return repr.at(m_mask);
        }

        [[nodiscard]] static tl::expected<CastlingRights, std::string>
        from_string(const std::string_view sv) {
            const auto it = std::ranges::find(repr, sv);
            if (it == repr.end()) {
                return tl::unexpected{fmt::format("unexpected string {}", sv)};
            }
            return CastlingRights(std::distance(repr.begin(), it));
        }

        MaskT m_mask;
    };

    inline constexpr EnumArray<CastlingRights, Square> CastlingRights::lost_table{
        std::in_place, [](const Square sq) {
            return sq == E1   ? CastlingRights{WHITE_KINGSIDE, WHITE_QUEENSIDE}
                   : sq == H1 ? CastlingRights{WHITE_KINGSIDE}
                   : sq == A1 ? CastlingRights{WHITE_QUEENSIDE}
                   : sq == E8 ? CastlingRights{BLACK_KINGSIDE, BLACK_QUEENSIDE}
                   : sq == H8 ? CastlingRights{BLACK_KINGSIDE}
                   : sq == A8 ? CastlingRights{BLACK_QUEENSIDE}
                              : CastlingRights{};
        }};

    inline constexpr CastlingRights CASTLING_NONE{0};

    inline constexpr CastlingRights CASTLING_K{WHITE_KINGSIDE};
    inline constexpr CastlingRights CASTLING_Q{WHITE_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_k{BLACK_KINGSIDE};
    inline constexpr CastlingRights CASTLING_q{BLACK_QUEENSIDE};

    inline constexpr CastlingRights CASTLING_KQ{WHITE_KINGSIDE, WHITE_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_Kk{WHITE_KINGSIDE, BLACK_KINGSIDE};
    inline constexpr CastlingRights CASTLING_Kq{WHITE_KINGSIDE, BLACK_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_Qk{WHITE_QUEENSIDE, BLACK_KINGSIDE};
    inline constexpr CastlingRights CASTLING_Qq{WHITE_QUEENSIDE, BLACK_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_kq{BLACK_KINGSIDE, BLACK_QUEENSIDE};

    inline constexpr CastlingRights CASTLING_KQk{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE};
    inline constexpr CastlingRights CASTLING_KQq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_Kkq{WHITE_KINGSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};
    inline constexpr CastlingRights CASTLING_Qkq{WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};

    inline constexpr CastlingRights CASTLING_KQkq{WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE};

    namespace bit {
        template <std::unsigned_integral T>
        constexpr int
        popcount(const T x) noexcept {
            return std::popcount(x);
        }

        template <std::unsigned_integral T>
        constexpr int
        get_lsb(const T bb) noexcept {
            return std::countr_zero(bb);
        }

        template <std::unsigned_integral T>
        constexpr int
        get_msb(const T bb) noexcept {
            return std::numeric_limits<T>::digits - 1 - std::countl_zero(bb);
        }

        template <std::unsigned_integral T>
        constexpr int
        pop_lsb(T& bb) noexcept {
            int n = std::countr_zero(bb);
            bb &= ~(static_cast<T>(1) << n);
            return n;
        }

        template <std::unsigned_integral T>
        constexpr int
        pop_msb(T& bb) noexcept {
            int n = std::numeric_limits<T>::digits - 1 - std::countl_zero(bb);
            bb &= ~(static_cast<T>(1) << n);
            return n;
        }

        template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
        constexpr T
        shift_left(const T value, const unsigned shift) {
            // assert(shift < sizeof(T) * 8 && "shift exceeds its bit width");
            return value << shift;
        }

        template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
        constexpr T
        shift_right(const T value, const unsigned shift) {
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
    struct Move : Printable<Move>, Parsable<Move> {
      public:
        Move() : m_data(0) {
        }
        constexpr explicit Move(const std::uint16_t d) noexcept : m_data(d) {
        }

        constexpr Move(const Square from, const Square to) noexcept
            : m_data(static_cast<uint16_t>((from.value() << 6) + to.value())) {
        }

        // to build a move if you already know the type of move
        template <move_type_t T>
        static constexpr Move
        make(const Square from, const Square to, const PieceType pt = KNIGHT) noexcept {
            assert(T != CASTLING);
            return Move{static_cast<uint16_t>(T + ((pt - KNIGHT).value() << 12) + (from.value() << 6) + to.value())};
        }

        template <move_type_t T>
        static constexpr Move
        make(const Square from, const Square to, const CastlingType c) noexcept {
            assert(T == CASTLING);
            return Move(static_cast<uint16_t>(T + (c.value() << 12) + (from.value() << 6) + to.value()));
        }

        // for sanity check
        [[nodiscard]] constexpr bool
        is_ok() const noexcept {
            return none().m_data != m_data && null().m_data != m_data;
        }

        // for these two moves from and to are the same
        static constexpr Move
        null() noexcept {
            return Move(65);
        }
        static constexpr Move
        none() noexcept {
            return Move(0);
        }

        constexpr bool
        operator==(const Move& m) const noexcept {
            return m_data == m.m_data;
        }
        constexpr bool
        operator!=(const Move& m) const noexcept {
            return m_data != m.m_data;
        }

        constexpr explicit operator bool() const noexcept {
            return m_data != 0;
        }

        [[nodiscard]] constexpr std::uint16_t
        raw() const noexcept {
            return m_data;
        }

        [[nodiscard]] constexpr Square
        from_sq() const noexcept {
            assert(is_ok());
            return Square{(m_data >> 6) & 0x3F};
        }

        [[nodiscard]] constexpr Square
        to_sq() const noexcept {
            assert(is_ok());
            return Square{m_data & 0x3F};
        }

        [[nodiscard]] constexpr move_type_t
        type_of() const noexcept {
            return static_cast<move_type_t>(m_data & 0b11 << 14);
        }

        [[nodiscard]] constexpr PieceType
        promotion_type() const noexcept {
            return PieceType{(m_data >> 12 & 0b11)} + KNIGHT;
        }

        [[nodiscard]] constexpr CastlingType
        castling_type() const noexcept {
            return static_cast<CastlingType>(m_data >> 12 & 0b11);
        }

        [[nodiscard]] std::string
        to_string() const noexcept {
            if (type_of() != PROMOTION) {
                return fmt::format("{}{}", from_sq(), to_sq());
            } else {
                return fmt::format("{}{}{}", from_sq(), to_sq(), promotion_type());
            }
        }

        struct UCICtx {
            const EnumArray<Piece, Square>& pieces;
            const Square                    ep_square;
            const CastlingRights            castling_rights;
        };

        static tl::expected<Move, std::string>
        from_uci(std::string_view, const UCICtx&);

        [[nodiscard]] static Move
        make_with_uci_ctx(const Move move, const UCICtx& ctx) {
            const auto from = move.from_sq();
            const auto to   = move.to_sq();
            if (ctx.pieces.at(from).type() == PAWN && ctx.ep_square == to) {
                return make<EN_PASSANT>(from, to);
            }

            if (ctx.pieces.at(from).type() == KING) {
                CastlingRights copy = ctx.castling_rights;
                while (!copy.empty()) {
                    auto type = CastlingType{bit::get_lsb(copy.mask())};
                    copy.remove(type);
                    const auto [k_from, k_to] = type.king_move();
                    const auto [r_from, _]    = type.rook_move();
                    if (ctx.pieces.at(r_from) != Piece{type.color(), ROOK}) continue;
                    if (ctx.pieces.at(from).color() != type.color() || from != k_from || to != k_to) continue;
                    return make<CASTLING>(k_from, k_to, type);
                }
            }

            return move;
        }

        std::uint16_t m_data;
    };
} // namespace chepp

template <>
struct std::hash<chepp::Move> {
    std::size_t
    operator()(const chepp::Move& m) const noexcept {
        return std::hash<std::uint16_t>()(m.raw());
    }
};

namespace chepp {

    inline tl::expected<Move, std::string>
    Move::from_uci(const std::string_view s, const UCICtx& info) {
        auto err = [](const std::string& msg) {
            return tl::unexpected{fmt::format("error while parsing uci move: {}", msg)};
        };
        if (!(s.size() == 4 || s.size() == 5)) {
            return err("invalid string size");
        }
        const auto from = Square::from_string(s.substr(0, 2));
        const auto to   = Square::from_string(s.substr(2, 2));

        if (!from) {
            return err(from.error());
        }
        if (!to) {
            return err(to.error());
        }

        if (s.size() == 5) {
            const auto pt = PieceType::from_string(s.substr(4, 1));
            if (!pt) {
                return err(pt.error());
            }
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
                if (info.pieces.at(r_from) != Piece{type.color(), ROOK}) {
                    continue;
                }
                if (info.pieces.at(*from).color() != type.color() || *from != k_from || *to != k_to) {
                    continue;
                }
                return make<CASTLING>(k_from, k_to, type);
            }
        }

        return make<NORMAL>(*from, *to);
    }

    [[nodiscard]] constexpr CastlingRights
    CastlingRights::lost_from_move(Move move) const {
        return CastlingRights{(lost_table[move.from_sq()].m_mask | lost_table[move.to_sq()].m_mask) & m_mask};
    }

    struct Fen {
        EnumArray<Piece, Square> pieces{};
        Color                    color{};
        CastlingRights           crs{};
        Square                   ep_square{};
        uint16_t                 halfmove{}, fullmove{};

        [[nodiscard]] static tl::expected<Fen, std::string>
        from_string(const std::string& s) noexcept {
            auto err = [](const std::string& msg) {
                return tl::unexpected{fmt::format("error while parsing fen: {}", msg)};
            };
            Fen                fen{};
            std::istringstream iss{s};
            std::string        board_str, color_str, castling_str, ep_str, halfmove_str, fullmove_str;
            if (!(iss >> board_str >> color_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str)) {
                return err("FEN parse error: expected 6 fields");
            }

            File file = FILE_A;
            Rank rank = RANK_8;

            for (const char c : board_str) {
                if (c == '/') {
                    if (file) {
                        return err("Each rank must have exactly 8 squares");
                    }
                    --rank;
                    file = FILE_A;
                } else if (std::isdigit(static_cast<unsigned char>(c))) {
                    if (c < '1' || c > '8') {
                        return err("Invalid digit in FEN rank");
                    }
                    file = file + (c - '0');
                } else {
                    const auto pc = Piece::from_string({&c, 1});
                    if (!pc) {
                        return err(pc.error());
                    }
                    if (!file) {
                        return err("FEN error: file overflow when placing piece");
                    }

                    fen.pieces.at(Square{file, rank}) = pc.value();
                    ++file;
                }
            }

            if (rank != RANK_1) {
                return err("Incorrect number of ranks/squares");
            }
            const auto color = Color::from_string(color_str);
            if (!color) {
                return err(color.error());
            }
            fen.color = color.value();

            const auto crs = CastlingRights::from_string(castling_str);
            if (!crs) {
                return err(crs.error());
            }
            fen.crs = crs.value();

            const auto ep_square = Square::from_string(ep_str);
            if (!ep_square) {
                return err(fmt::format("Invalid en-passant square: {}", ep_square.error()));
            }
            fen.ep_square = ep_square.value();

            if (halfmove_str.size() > 3 || fullmove_str.size() > 3)
                return tl::unexpected(fmt::format("Halfmove/fullmove field too long"));

            try {
                fen.halfmove = static_cast<uint16_t>(std::stoi(halfmove_str));
                fen.fullmove = static_cast<uint16_t>(std::stoi(fullmove_str));
            } catch (const std::exception& e) {
                return err(fmt::format("Invalid move counters in FEN: {}", e.what()));
            }
            if (fen.fullmove < 1) {
                return err("Invalid move counters in FEN: ");
            }
            return fen;
        }

        [[nodiscard]] std::string
        to_string() const {
            std::ostringstream oss;

            for (auto r = RANK_1; r <= RANK_8; ++r) {
                const auto rank  = RANK_8 - r;
                int        empty = 0;
                for (auto file = FILE_A; file <= FILE_H; ++file) {
                    const Square sq{file, rank};

                    if (Piece pc = pieces.at(sq); pc == NO_PIECE) {
                        ++empty;
                    } else {
                        if (empty > 0) {
                            fmt::print(oss, "{}", empty);
                            empty = 0;
                        }
                        fmt::print(oss, "{}", pc);
                    }
                }
                if (empty > 0) {
                    fmt::print(oss, "{}", empty);
                }

                if (rank > RANK_1) {
                    fmt::print(oss, "/");
                }
            }

            fmt::print(oss, " {} {} {} {} {}", color, crs, ep_square, halfmove, fullmove);
            return oss.str();
        }

        friend auto&
        operator<<(std::ostream& os, const Fen& fen) {
            os << fen.to_string();
            return os;
        }
    };

    inline constexpr int MAX_PLY = 255;

    inline constexpr int MATE_SCORE    = 32000;
    inline constexpr int INF_SCORE     = 32001;
    inline constexpr int INVALID_SCORE = 32002;

    inline constexpr int MAX_MOVES = 256;

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

    constexpr int
    mate_in(const int ply) noexcept {
        // better to mate quick -> ply is small
        return MATE - ply;
    }

    constexpr int
    mated_in(const int ply) noexcept {
        // better if mate slow -> ply big
        return MATED + ply;
    }

    constexpr int
    absolute_eval(const int eval, const Color side) noexcept {
        return side == WHITE ? eval : -eval;
    }

    constexpr int
    relative_eval(const int eval, const Color side) noexcept {
        return side == WHITE ? eval : -eval;
    }

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

        VectorHandle(std::vector<T>* p, const uint32_t i) : ptr(p), index(i) {
        }
        VectorHandle() : ptr(nullptr), index(0) {
        }

        T&
        operator*() const {
            return ptr->at(index);
        }

        T*
        operator->() const {
            return &ptr->at(index);
        }

        T&
        operator()() const {
            return ptr->at(index);
        }

        explicit operator bool() const {
            return ptr != nullptr && index < ptr->size();
        }

        bool
        operator==(const VectorHandle& other) const {
            return ptr == other.ptr && index == other.index;
        }
        bool
        operator!=(const VectorHandle& other) const {
            return !(*this == other);
        }
    };

    template <typename Pred1, typename Pred2>
    auto
    or_predicate(Pred1 a, Pred2 b) {
        return [=](auto&& x) { return a(x) || b(x); };
    }

    template <std::size_t N, typename F, std::size_t... I>
    constexpr auto
    create_array_impl(F&& func, std::index_sequence<I...>) {
        return std::array<std::invoke_result_t<F, std::size_t>, N>{{func(I)...}};
    }

    template <std::size_t N, typename F>
    constexpr auto
    create_array(F&& func) {
        return create_array_impl<N>(std::forward<F>(func), std::make_index_sequence<N>{});
    }
} // namespace chepp
#endif
