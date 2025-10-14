//
// Created by paul on 9/30/25.
//

#ifndef CHEPP_INIT_H
#define CHEPP_INIT_H

#include "bitboard.h"
#include "nnue.h"
#include "zobrist.h"

// initialisation order does not matter
inline Zobrist::Initialiser  zobrist_initialiser;
inline Bitboard::Initialiser bitboard_initialiser;
inline Movegen::Initialiser  movegen_initialiser;
inline nnue::Initialiser     nnue_initialiser;

#endif // CHEPP_INIT_H
