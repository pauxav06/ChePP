#ifndef COMPAT_H
#define COMPAT_H

#if defined(USE_STD_EXPEXTED)
#include <expected>
namespace tl = std;
#else
#include <tl/expected.hpp>
#endif

#endif