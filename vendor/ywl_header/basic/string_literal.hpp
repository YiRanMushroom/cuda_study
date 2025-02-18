#pragma once

#include <algorithm>

namespace ywl::basic {
    template<std::size_t N>
    struct string_literal {
        char data[N];

        constexpr string_literal(const char (&str)[N]) : data{} { // NOLINT
            std::copy_n(str, N, data);
        }
    };
}
