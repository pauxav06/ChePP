//
// Created by paul on 7/30/25.
//

#ifndef TB_H
#define TB_H

#if CHEPP_USE_TB
#include <src/tbprobe.h>
#endif
#include <filesystem>
#include <iostream>

inline bool
init_tb(const std::string_view path) {
#if CHEPP_USE_TB
    if (!std::filesystem::exists(path)) {
        std::cerr << "Tablebase path does not exist: " << path << "\n";
        return false;
    }

    if (tb_init(path.begin())) {
        return true;
    }
    std::cerr << "Tablebase init failed: " << path << "\n";
    return false;
#else
    (void)path;
    throw std::runtime_error("Tablebases are not enabled!");
#endif
}

#endif // TB_H
