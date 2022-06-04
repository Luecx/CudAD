/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUDATEST1_SRC_DATA_DATA_H_
#define CUDATEST1_SRC_DATA_DATA_H_

#include "../assert/Assert.h"
#include "../assert/GPUAssert.h"
#include "align.h"
#include "mode.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <new>
#include <ostream>
#include <random>

template<typename Type = float>
class SArray {

    protected:
    // use an unaligned and aligned data access
    Type* raw_cpu   = nullptr;

    bool  clean_cpu = false;
    bool  clean_gpu = false;

    public:
    uint32_t size {};
    Type*    cpu_values = nullptr;
    Type*    gpu_values = nullptr;

    public:
    explicit SArray(uint32_t p_size) : size {p_size} {}
    SArray(const SArray<Type>& other) {
        this->size = other.size;
        if (other.cpu_is_allocated()) {
            malloc_cpu();
            this->template copy<HOST>(other);
        }
        if (other.gpu_is_allocated()) {
            malloc_gpu();
            this->template copy<DEVICE>(other);
        }
    }
    SArray(SArray<Type>&& other) noexcept {
        this->size       = other.size;
        this->raw_cpu    = other.raw_cpu;
        this->clean_cpu  = other.clean_cpu;
        this->clean_gpu  = other.clean_gpu;
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
        other.clean_gpu  = false;
        other.clean_cpu  = false;
    }
    SArray(const SArray<Type>& other, int offset, int count) {
        this->raw_cpu    = other.raw_cpu;
        this->cpu_values = &(other.cpu_values[offset]);
        this->gpu_values = &(other.gpu_values[offset]);
        this->size       = count;
    }
    virtual ~SArray() {
        free_cpu();
        free_gpu();
    }
    SArray<Type>& operator=(const SArray<Type>& other) {
        free_cpu();
        free_gpu();
        this->size = other.size;
        if (other.cpu_is_allocated()) {
            malloc_cpu();
            this->template copy<HOST>(other);
        }
        if (other.gpu_is_allocated()) {
            malloc_gpu();
            this->template copy<DEVICE>(other);
        }
        return (*this);
    }
    SArray<Type>& operator=(SArray<Type>&& other) noexcept {
        free_cpu();
        free_gpu();
        this->size       = other.size;
        this->raw_cpu    = other.raw_cpu;
        this->clean_cpu  = other.clean_cpu;
        this->clean_gpu  = other.clean_gpu;
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
        other.clean_gpu  = false;
        other.clean_cpu  = false;
        return (*this);
    }

    void malloc_cpu() {
        if (cpu_is_allocated())
            free_cpu();
        // make sure to clean the data once we are done here
        clean_cpu = true;
        // allocate raw unaligned data
        raw_cpu = new Type[size + ALIGNED_BYTES] {};

        // create a void pointer which will be aligned via the std::align function
        void* data_ptr = raw_cpu;
        // also track how much the aligned data can hold
        uint64_t reduced_size = size + ALIGNED_BYTES;

        // align the pointer and reduce the maximum size
        std::align(ALIGNED_BYTES, size, data_ptr, reduced_size);

        // store the final pointer
        cpu_values = (Type*) data_ptr;
    }
    void malloc_gpu() {
        if (gpu_is_allocated())
            free_gpu();
        // make sure to clean once we are done
        clean_gpu = true;
        CUDA_ASSERT(cudaMalloc(&gpu_values, size * sizeof(Type)));
    }
    void free_cpu() {
        if (cpu_is_allocated()) {
            if (clean_cpu) {
                delete[] raw_cpu;
            }
            raw_cpu    = nullptr;
            cpu_values = nullptr;

            clean_cpu  = false;
        }
    }
    void free_gpu() {
        if (gpu_is_allocated()) {
            if (clean_gpu) {
                CUDA_ASSERT(cudaFree(gpu_values));
            }
            gpu_values = nullptr;
            clean_gpu  = false;
        }
    }
    bool cpu_is_allocated() const { return cpu_values != nullptr; }
    bool gpu_is_allocated() const { return gpu_values != nullptr; }
    void gpu_upload() {
        if (!cpu_is_allocated() || !gpu_is_allocated())
            return;
        CUDA_ASSERT(cudaMemcpy(gpu_values, cpu_values, size * sizeof(Type), cudaMemcpyHostToDevice));
    }
    void gpu_download() {
        if (!cpu_is_allocated() || !gpu_is_allocated())
            return;
        CUDA_ASSERT(cudaMemcpy(cpu_values, gpu_values, size * sizeof(Type), cudaMemcpyDeviceToHost));
    }

    Type& get(int height) const {
        ASSERT(cpu_is_allocated());
        ASSERT(height < size);
        ASSERT(height >= 0);
        return cpu_values[height];
    }
    Type               operator()(int height) const { return get(height); }
    Type&              operator()(int height) { return get(height); }

    [[nodiscard]] Type min() const {
        if (cpu_values == nullptr)
            return 0;
        Type m = cpu_values[0];
        for (int i = 0; i < size; i++) {
            m = std::min(m, cpu_values[i]);
        }
        return m;
    };
    [[nodiscard]] Type max() const {
        if (cpu_values == nullptr)
            return 0;
        Type m = cpu_values[0];
        for (int i = 0; i < size; i++) {
            m = std::max(m, cpu_values[i]);
        }
        return m;
    }
    void sort() const {
        if (cpu_values == nullptr)
            return;
        std::sort(cpu_values, cpu_values + size, std::greater<Type>());
    };

    template<Mode mode = HOST>
    void clear() const {
        if (mode == HOST) {
            if (cpu_values == nullptr)
                return;
            std::memset(cpu_values, 0, sizeof(Type) * size);
        }
        if (mode == DEVICE) {
            if (gpu_values == nullptr)
                return;
            cudaMemset(gpu_values, 0, sizeof(Type) * size);
        }
    }
    template<Mode mode = HOST>
    void copy(const SArray& other) {
        if (mode == HOST) {
            memcpy(cpu_values, other.cpu_values, size * sizeof(Type));
        } else if (mode == DEVICE) {
            cudaMemcpy(gpu_values, other.gpu_values, size * sizeof(Type), cudaMemcpyDeviceToDevice);
        }
    }
    template<Mode mode = HOST>
    void copy(const Type* data, const int count, const int offset) {
        if (mode == HOST) {
            memcpy(&cpu_values[offset], data, count * sizeof(Type));
        } else if (mode == DEVICE) {
            cudaMemcpy(&cpu_values[offset], data, count * sizeof(Type), cudaMemcpyDefault);
        }
    }
    template<Mode mode = HOST>
    void assign(const Type* data) {
        if (mode == HOST) {
            free_cpu();
            this->cpu_values = data;
        } else if (mode == DEVICE) {
            free_gpu();
            this->gpu_values = data;
        }
    }
    template<Mode mode = HOST>
    void assign(const SArray& other) {
        if (mode == HOST) {
            free_cpu();
            this->cpu_values = other.cpu_values;
        } else if (mode == DEVICE) {
            free_gpu();
            this->gpu_values = other.gpu_values;
        }
    }

    void randomise(Type lower = 0, Type upper = 1) {
        if (cpu_values == nullptr)
            return;
        for (int i = 0; i < size; i++) {
            this->cpu_values[i] = static_cast<Type>(rand()) / RAND_MAX * (upper - lower) + lower;
        }
    }
    void randomiseGaussian(Type mean, Type deviation) {
        if (cpu_values == nullptr)
            return;
        std::default_random_engine     generator;
        std::normal_distribution<Type> distribution(mean, deviation);
        for (int i = 0; i < size; i++) {
            this->cpu_values[i] = distribution(generator);
        }
    }

    [[nodiscard]] SArray<Type> newInstance() const { return SArray<Type> {size}; }

    friend std::ostream&       operator<<(std::ostream& os, const SArray& data) {
        os << "size:       " << data.size << "\n"
           << "gpu_values: " << data.gpu_values << "    cleanUp = [" << data.clean_gpu << "]\n"
           << "cpu_values: " << data.cpu_values << "    cleanUp = [" << data.clean_cpu << "]\n";
        if (!data.cpu_is_allocated())
            return os;
        for (int n = 0; n < data.size; n++) {
            os << std::setw(11) << (double) data.get(n);
        }
        os << "\n";
        return os;
    }
};

#endif    // CUDATEST1_SRC_DATA_DATA_H_
