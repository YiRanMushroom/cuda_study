#pragma once

#include <array>

namespace _impl {
    struct array_offset {
        size_t offset;
        operator size_t() const { return offset; }

        template<typename T>
        T *operator[](T *ptr) const { return ptr + offset; }

        template<typename T>
        const T *operator[](const T *ptr) const { return ptr + offset; }

        size_t get_offset() const { return offset; }
    };

    template<std::size_t N>
    struct multi_dim_accessor {
        std::array<std::size_t, N> dims;
        std::array<std::size_t, N> strides;

        multi_dim_accessor(const std::array<std::size_t, N> &dims) : dims(dims) {
            strides[N - 1] = 1;
            for (std::size_t i = N - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * dims[i];
            }
        }

        array_offset operator()(const auto &... indices) const {
            static_assert(sizeof...(indices) == N, "Number of arguments must match number of dimensions");
            // use index sequence to expand the variadic template arguments
            auto seq = std::index_sequence_for<decltype(indices)...>{};
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return array_offset{((indices * strides[I]) + ...)};
            }(seq);
        }
    };
}

auto make_md_accessor(const auto &&... dims) {
    return _impl::multi_dim_accessor<sizeof...(dims)>{
        std::array<size_t, sizeof...(dims)>{static_cast<std::size_t>(dims)...}
    };
}
