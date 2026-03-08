#include "format.h"
#include "position.h"

#include <filesystem>

#if CHEPP_USE_TB
#include "tbconfig.h"
#include "tbprobe.c"
#endif

namespace chepp {
    bool
    init_tb(const std::string_view path) {
#if CHEPP_USE_TB
        if (!std::filesystem::exists(path)) {
            fmt::println(stderr, "Tablebase path does not exist: {}", path);
            return false;
        }

        if (tb_init(path.data())) {
            return true;
        }
        fmt::println(stderr, "Failed to initialize TB: {}", path);
        return false;
#else
        (void)path;
        throw std::runtime_error("TB are not enabled");
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
        throw std::runtime_error("TB are not enabled");
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
        throw std::runtime_error("TB are not enabled");
#endif
    }
} // namespace chepp