//
// Created by paul on 7/30/25.
//

#ifndef TB_H
#define TB_H

#include <src/tbprobe.h>

#include <filesystem>
#include <iostream>

inline bool init_tb(const std::string_view path)
{
    if (!std::filesystem::exists(path)) {
        std::cerr << "Tablebase path does not exist: " << path << "\n";
        return false;
    }

    if (tb_init(path.begin()))
    {
        return true;
    }
    std::cerr << "Tablebase init failed: " << path << "\n";
    return false;

}

#endif //TB_H
