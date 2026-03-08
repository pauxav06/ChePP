#ifndef CHEPP_FORMAT_H
#define CHEPP_FORMAT_H

#ifndef USE_STD_FMT
#include <cstdio>
#include <fmt/base.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#else
#include <cstdio>
#include <format>
#include <ostream>
#include <print>
namespace fmt = std;
#endif

#endif // CHEPP_FORMAT_H
