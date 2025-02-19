#pragma once

#include <assert.h>
#include <fenv.h>
#include <ywl_header/all.hpp>

namespace cuda {
    namespace _impl {
        template<typename T>
        struct CudaPtrHolderHint {
            struct voidptr_ull {
                void *ptr;
                size_t byte_size;

                bool operator==(const voidptr_ull &other) const = default;
            };

            using value_type = voidptr_ull;
            using data_type = T;
            constexpr static value_type default_value = {nullptr, 0};

            static void destroy_value(value_type &&value) {
                auto &&[ptr, byte_size] = value;
                if (ptr) {
                    cudaFree(ptr);
                } else {
                    assert(byte_size == 0);
                }

                std::cout << "destroyed cuda ptr of type " << typeid(T).name() << " with length "
                        << byte_size / sizeof(T)
                        << std::endl;

                ptr = nullptr;
                byte_size = 0;
            }

            static value_type move_value(value_type &value) {
                auto &&[ptr, byte_size] = value;
                value_type ret = {ptr, byte_size};
                ptr = nullptr;
                byte_size = 0;
                return ret;
            }

            static value_type create(size_t byte_size) {
                void *ptr = nullptr;
                cudaMalloc(&ptr, byte_size);
                return {ptr, byte_size};
            }
        };

        // static_assert(ywl::miscellaneous::is_unique_resource_holder_identity_hint_type<CudaPtrHolderHint>);

        template<typename T>
        using CudaPtrHolder = ywl::miscellaneous::unique_holder<CudaPtrHolderHint<T> >;
    }

    template<typename T> requires std::is_trivial_v<T>
    class CudaArray {
    private:
        constexpr static size_t element_size = sizeof(T);
        _impl::CudaPtrHolder<T> m_holder;

        CudaArray(_impl::CudaPtrHolder<T> holder) : m_holder(std::move(holder)) {}

    public:
        using value_type = T;

        CudaArray() = default;

        std::optional<T *> get() {
            return std::optional<T *>(static_cast<T *>(m_holder.get()));
        }

        T *get_as_buffer() {
            return static_cast<T *>(m_holder.get().ptr);
        }

        const T *get_as_buffer() const {
            return static_cast<const T *>(m_holder.get().ptr);
        }

        size_t length() const {
            if (!m_holder) {
                return 0;
            }
            assert(m_holder.get().byte_size % element_size == 0);
            return m_holder.get().byte_size / element_size;
        }

        template<typename Ret = int>
        Ret byte_size() const {
            if (!m_holder) {
                return 0;
            }
            return static_cast<Ret>(m_holder.get().byte_size);
        }

        template<typename Iter> requires (
            std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>
            && std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>
        )
        static auto create_from_host(Iter begin, Iter end) -> CudaArray<T> {
            size_t size = std::distance(begin, end);
            size_t byte_size = size * element_size;
            auto holder = _impl::CudaPtrHolder<T>::create(byte_size);
            cudaMemcpy(holder.get().ptr, &(*begin), byte_size, cudaMemcpyHostToDevice);
            return CudaArray<T>{std::move(holder)};
        }

        template<typename Iter> requires (
            std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>
            && std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>
        )
        void yield_non_block(Iter begin, Iter end) {
            size_t length = std::distance(begin, end);

            assert(length >= this->length());

            cudaMemcpy(&(*begin), this->get_as_buffer(), byte_size(), cudaMemcpyDeviceToHost);
        }

        template<typename Iter> requires (
            std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>
            && std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>
        )
        void yield_block(Iter begin, Iter end) {
            size_t length = std::distance(begin, end);

            assert(length >= this->length());
            cudaDeviceSynchronize();

            cudaMemcpy(&(*begin), this->get_as_buffer(), byte_size(), cudaMemcpyDeviceToHost);
        }

        static auto copy_from(const CudaArray<T> &other) -> CudaArray<T> {
            auto holder = _impl::CudaPtrHolder<T>::create(other.byte_size());
            cudaMemcpy(holder.get().ptr, other.get_as_buffer(), other.byte_size(), cudaMemcpyDeviceToDevice);
            return CudaArray<T>{std::move(holder)};
        }

        static auto with_length(size_t size) -> CudaArray<T> {
            size_t byte_size = size * element_size;
            auto holder = _impl::CudaPtrHolder<T>::create(byte_size);
            return CudaArray<T>{std::move(holder)};
        }

        template<typename Iter> requires (
            std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>
            && std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>
        )
        static auto same_size_as(Iter begin, Iter end) -> CudaArray<T> {
            size_t size = std::distance(begin, end);
            return with_length(size);
        }

        static auto same_size_as(const CudaArray<T> &other) -> CudaArray<T> {
            return with_length(other.length());
        }
    };

    namespace _impl {
        template<typename T>
        struct Wait_Result_T {
        private:
            T result;

        public:
            Wait_Result_T(T result) : result(std::move(result)) {}

            T get() {
                return std::move(result);
            }

            T wait() {
                // sync cuda
                cudaDeviceSynchronize();
                return std::move(result);
            }
        };

        template<typename T>
        struct Wait_For_Size_Then_Invoke_Return_Struct;

        template<typename F, int block_size, typename ReturnType, bool forward_size, typename... Args>
        struct Invoke_Same_Return_Struct {
        private:
            int array_length;
            F *func;
            std::tuple<const Args &...> arrays;

        public:
            friend struct Wait_For_Size_Then_Invoke_Return_Struct<Invoke_Same_Return_Struct>;
            // reference to the arrays
            Invoke_Same_Return_Struct(int array_length, F *func,
                                      const Args &... arrays) : array_length(array_length),
                                                                func(func), arrays(arrays...) {}


            auto invoke_with(auto &&... args) {
                auto ret = CudaArray<ReturnType>::with_length(array_length);

                if constexpr (forward_size) {
                    std::apply([&](const auto &... array) {
                        func<<<(array_length + block_size - 1) / block_size, block_size>>>(array.get_as_buffer()...,
                            ret.get_as_buffer(),
                            ret.byte_size(),
                            std::forward<decltype(args)>(args)...);
                    }, arrays);
                } else {
                    std::apply([&](const auto &... array) {
                        func<<<(array_length + block_size - 1) / block_size, block_size>>>(array.get_as_buffer()...,
                            ret.get_as_buffer(),
                            std::forward<decltype(args)>(args)...);
                    }, arrays);
                }

                return Wait_Result_T<decltype(ret)>(std::move(ret));
            }

            auto invoke() {
                return invoke_with();
            }
        };

        template<typename T>
        struct Wait_For_Size_Then_Invoke_Return_Struct {
        private:
            T invoke_return_struct;

        public:
            Wait_For_Size_Then_Invoke_Return_Struct(T invoke_return_struct) : invoke_return_struct(
                std::move(invoke_return_struct)) {}

            T output_length(const auto &length) {
                invoke_return_struct.array_length = length;
                return std::move(invoke_return_struct);
            }
        };
    }

    template<typename ReturnType = void, bool forward_size = false, int block_size = 256, typename F, typename First,
        typename
        ... Rest>
    // func must be global function
    auto invoke_any_size(F *func, const First &first, const Rest &... rest) {
        if constexpr (!std::is_same_v<ReturnType, void>) {
            return _impl::Wait_For_Size_Then_Invoke_Return_Struct{
                _impl::Invoke_Same_Return_Struct<F, block_size, ReturnType, forward_size, const First &, const Rest &
                    ...>(
                    0, func, first, rest...)
            };
        } else {
            using ActualReturnType = typename std::remove_cvref_t<decltype(first)>::value_type;
            return _impl::Wait_For_Size_Then_Invoke_Return_Struct{
                _impl::Invoke_Same_Return_Struct<F, block_size, ActualReturnType, forward_size, const First &,
                    const Rest &...>(
                    0, func, first, rest...)
            };
        }
    };

    /// F is the function, expect it to take some (or 1) T*, which is the device data pointer
    /// any number of input arguments, and then one output T* pointer
    /// followed by the length of the data (not byte size)
    /// with any number of additional arguments
    /// F must be a function pointer
    /// for the size of the data, we expect all T* has the same size
    /// otherwise the smallest size will be used
    template<typename ReturnType = void, bool forward_size = true, int block_size = 256, typename F, typename First,
        typename
        ... Rest>
    // func must be global function
    auto invoke_same_size(F *func, const First &first, const Rest &... rest) {
        // assert all cuda_arrays have the same length
        int array_length = std::min({first.length(), rest.length()...});
        // this function returns a struct that overloads operator()
        if constexpr (!std::is_same_v<ReturnType, void>) {
            return _impl::Wait_For_Size_Then_Invoke_Return_Struct{
                _impl::Invoke_Same_Return_Struct<F, block_size, ReturnType, forward_size, const First &, const Rest &
                    ...>(
                    array_length, func, first, rest...)
            };
        } else {
            using ActualReturnType = typename std::remove_cvref_t<decltype(first)>::value_type;
            return _impl::Wait_For_Size_Then_Invoke_Return_Struct{
                _impl::Invoke_Same_Return_Struct<F, block_size, ActualReturnType, forward_size, const First &, const
                    Rest &...>(
                    array_length, func, first, rest...)
            };
        }
    }
}
