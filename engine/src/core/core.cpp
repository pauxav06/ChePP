#include "bitboard.h"
#include "position.h"
#include "tbconfig.h"
#include "types.h"
#include <filesystem>

#if CHEPP_USE_TB
#include "tbprobe.c"
#include "tbprobe.h"
#endif

namespace chepp {
    bool
    init_tb(const std::string_view path) {
#if CHEPP_USE_TB
        if (!std::filesystem::exists(path)) {
            std::cerr << "Tablebase path does not exist: " << path << "\n";
            return false;
        }

        if (tb_init(path.data())) {
            return true;
        }
        std::cerr << "Tablebase init failed: " << path << "\n";
        return false;
#else
        (void)path;
        throw std::runtime_error("TB are not active");
#endif
    }

    unsigned
    Position::wdl_probe() const {
#if CHEPP_USE_TB
        const unsigned ep_sq = ep_square() == NO_SQUARE ? 0 : ep_square().value() + 1;
        return tb_probe_wdl(occupancy(WHITE).value(),
                            occupancy(BLACK).value(),
                            occupancy(KING).value(),
                            occupancy(QUEEN).value(),
                            occupancy(ROOK).value(),
                            occupancy(BISHOP).value(),
                            occupancy(KNIGHT).value(),
                            occupancy(PAWN).value(),
                            halfmove_clock(),
                            castling_rights().mask(),
                            ep_sq,
                            side_to_move() == WHITE);
#else
        throw std::runtime_error("TB are not active");
#endif
    }

    unsigned
    Position::dtz_probe() const {
#if CHEPP_USE_TB
        const unsigned ep_sq = ep_square() == NO_SQUARE ? 0 : ep_square().value() + 1;
        return tb_probe_root(occupancy(WHITE).value(),
                             occupancy(BLACK).value(),
                             occupancy(KING).value(),
                             occupancy(QUEEN).value(),
                             occupancy(ROOK).value(),
                             occupancy(BISHOP).value(),
                             occupancy(KNIGHT).value(),
                             occupancy(PAWN).value(),
                             halfmove_clock(),
                             castling_rights().mask(),
                             ep_sq,
                             side_to_move() == WHITE,
                             nullptr);
#else
        throw std::runtime_error("TB are not active");
#endif
    }
} // namespace chepp