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

struct fclose_deleter {
    void
    operator()(FILE* f) const noexcept {
        if (f) {
            fclose(f);
        }
    }
};

#include <memory>
using file_ptr = std::unique_ptr<FILE, fclose_deleter>;

namespace chepp {
    using ::file_ptr;
}

#endif // CHEPP_FORMAT_H
