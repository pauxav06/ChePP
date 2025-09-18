//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_GRAPHEUS_CONVERTER_H
#define CHEPP_GRAPHEUS_CONVERTER_H

#include "../../../engine/include/ChePP/engine/types.h"
#include "../binpack/nnue_training_data_formats.h"

#include <cstdint>




namespace GrapheusData {

struct Header {
    uint64_t entry_count {};
    char     label_1[128] {};
    char     label_2[128] {};
    char     label_3[1024] {};
};


struct Position {

    struct PieceList {
        static constexpr uint64_t piece_equivalence_tb[] {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13};

        static PieceList          make_piece_list(const chess::Position& pos) {
            auto      pieces = pos.piecesBB();
            PieceList list {};
            int       idx = 0;
            while (!pieces.isEmpty()) {
                const chess::Square sq {std::countr_zero(pieces.bits())};
                const uint64_t      pc     = piece_equivalence_tb[chess::ordinal(pos.pieceAt(sq))];
                const int           block  = idx / (64 / 4);
                const int           offset = idx % (64 / 4);
                list.bb[block] &= ~(0xFULL << (offset * 4));
                list.bb[block] |= (pc << (offset * 4));
                pieces.unset(sq);
                idx++;
            }
            return list;
        }

        uint64_t bb[2] {0xCCCCCCCCCCCCCCCC, 0xCCCCCCCCCCCCCCCC};
    };

    struct PositionMeta {
        uint8_t m_move_count {};
        uint8_t m_fifty_move_rule {};
        uint8_t m_castling_and_active_player {};
        uint8_t m_en_passant_square {64};
    };

    struct Result {
        uint16_t score {};
        uint8_t  wdl {};
    };


    static Position from_binpack_entry(const binpack::TrainingDataEntry& entry) {
        PositionMeta meta {};
        meta.m_castling_and_active_player =
            (chess::ordinal(entry.pos.sideToMove()) << 7)
            | static_cast<uint8_t>(entry.pos.castlingRights());
        meta.m_fifty_move_rule   = entry.pos.rule50Counter();
        meta.m_move_count        = entry.pos.fullMove();
        meta.m_en_passant_square = chess::ordinal(entry.pos.epSquare());

        Result res {};
        res.score = entry.score;
        res.wdl   = entry.result;

        return Position {
            PieceList::make_piece_list(entry.pos),
            entry.pos.piecesBB().bits(),
            meta,
            res};
    }


    PieceList    list;
    uint64_t     occupancy {};
    PositionMeta meta;
    Result       res;
};


static Header make_header(const std::size_t size) {
    Header header {};
    header.entry_count = size;
    return header;
}

}    // namespace GrapheusData


#endif // CHEPP_GRAPHEUS_CONVERTER_H
