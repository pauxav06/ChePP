//
// Created by paul on 9/10/25.
//

#ifndef CHEPP_FORMATTING_H
#define CHEPP_FORMATTING_H

#include "ChePP/engine/movegen.h"
#include "ChePP/engine/position.h"

#include <regex>

namespace PGN
{
    template <std::size_t N>
    struct fixed_string
    {
        char value[N]{};

        constexpr fixed_string(const char (&str)[N]) { std::copy_n(str, N, value); }

        constexpr operator std::string_view() const { return {value, N - 1}; }
    };

    template <fixed_string Name, typename T>
    struct Field
    {
        using value_type                       = T;
        static constexpr std::string_view name = Name;

        std::optional<T> value;

        Field() = default;
        Field(const T& v) : value(v) {}
        Field(T&& v) : value(std::move(v)) {}

        Field& operator=(const T& v)
        {
            value = v;
            return *this;
        }
        operator std::optional<T>&() { return value; }
        operator const std::optional<T>&() const { return value; }
    };

    using Event  = Field<"Event", std::string>;
    using Site   = Field<"Site", std::string>;
    using Round  = Field<"Round", int>;
    using Date   = Field<"Date", std::string>;
    using White  = Field<"White", std::string>;
    using Black  = Field<"Black", std::string>;
    using Result = Field<"Result", std::string>;

    struct GenericField
    {
        std::string name;
        std::string value;
    };

    inline void format_field(std::ostringstream& oss, const GenericField& f)
    {
        oss << "[" << f.name << " \"" << f.value << "\"]\n";
    }

    template <typename Field>
    void format_field(std::ostringstream& oss, const Field& f)
    {
        if (f.value)
        {
            oss << "[" << Field::name << " \"" << *f.value << "\"]\n";
        }
    }

    inline std::vector<GenericField> parse_tags(std::istream& in)
    {
        std::vector<GenericField> tags;
        std::string               line;
        static const std::regex   line_re(R"PGN(^\s*\[([^\s\]]+)\s+"([^"]*)"\s*\]\s*$)PGN");

        while (std::getline(in, line))
        {
            if (line.empty())
                break;

            if (std::smatch match; std::regex_match(line, match, line_re))
            {
                tags.push_back({match[1].str(), match[2].str()});
            }
        }
        return tags;
    }

    template <typename... Fs>
    struct Fields
    {
        std::tuple<Fs...> fields;

        Fields() = default;

        Fields(typename Fs::value_type... vals) : fields(Fs{vals}...) {}

        Fields(Fs... f) : fields(std::move(f)...) {}
    };

    template <typename... Fs>
    std::string format_pgn_tags(const Fields<Fs...>& tags)
    {
        std::ostringstream oss;
        std::apply([&](auto&&... f) { (format_field(oss, f), ...); }, tags.fields);
        oss << "\n";
        return oss.str();
    }

    inline Move::AlgebraicInfo get_algebraic_info(const Position& prev, const Position& next)
    {
        const auto     move     = next.move();
        const auto     piece    = prev.piece_at(move.from_sq());
        const bool     is_check = next.check_mask(~piece.color()) != Bitboard::empty();
        const Bitboard candidates =
            attacks(piece.type(), move.to_sq(), prev.occupancy(), ~piece.color()) &
            prev.occupancy(piece.type(), piece.color()) & ~Bitboard(move.from_sq());

        const bool is_capture = next.captured() != NO_PIECE;
        bool needs_file = false, needs_rank = false;
        if (piece.type() != PAWN || is_capture)
        {
            candidates.for_each_square([&] (const Square sq)
            {
                if (sq.file() == move.to_sq().file())
                {
                    needs_rank = true;
                }
                if (sq.rank() == move.to_sq().rank())
                {
                    needs_file = true;
                }
            });
        }
        bool is_mate = false;
        if (is_check)
        {
            if (gen_legal(next).empty())
            {
                is_mate = true;
            }
        }
        return {
            piece, needs_rank, needs_file, is_capture, is_check, is_mate
        };
    }

    template <typename... Fields>
    std::string to_pgn(const std::span<Position> positions, const PGN::Fields<Fields...>& tags)
    {
        std::ostringstream oss;
        oss << format_pgn_tags(tags);

        for (size_t i = 0; i < positions.size() - 1; ++i)
        {
            if (i % 2 == 0)
            {
                oss << (i / 2 + 1) << ". ";
            }
            oss << positions[i + 1].move().to_algebraic(get_algebraic_info(positions[i], positions[i + 1])) << " ";
        }

        return oss.str();
    }

} // namespace PGN

#endif // CHEPP_FORMATTING_H
