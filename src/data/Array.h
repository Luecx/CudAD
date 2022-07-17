
//
// Created by Luecx on 20.06.2022.
//

#ifndef CUDAD_SRC_DATA_ARRAY_H_
#define CUDAD_SRC_DATA_ARRAY_H_

#include "../assert/Assert.h"
#include "../assert/GPUAssert.h"
#include "Align.h"
#include "Mode.h"

#include <cstdint>

using ArraySizeType = uint32_t;

template<typename Type>
struct Array {
    ArraySizeType m_size = 0;
    Type*         m_data = nullptr;

    explicit Array(ArraySizeType p_size) : m_size(p_size) {}

    ArraySizeType size() const { return m_size; }
};

template<typename Type>
struct CPUArray : Array<Type> {

    explicit CPUArray(ArraySizeType p_size) : Array<Type>(p_size) {
        this->m_data = new Type[p_size] {};
    }
    virtual ~CPUArray() { delete[] this->m_data; }

    Type& operator()(ArraySizeType idx) { return this->m_data[idx]; }
    Type  operator()(ArraySizeType idx) const { return this->m_data[idx]; }
    Type& operator[](ArraySizeType idx) { return this->m_data[idx]; }
    Type  operator[](ArraySizeType idx) const { return this->m_data[idx]; }

    void  copyFrom(const CPUArray<Type>& other) {
        ASSERT(other.size() == this->size());
        memcpy(this->m_data, other.m_data, this->size() * sizeof(Type));
    }
    void clear() { memset(this->m_data, 0, sizeof(Type) * this->size()); }
};

template<typename Type>
struct GPUArray : Array<Type> {
    explicit GPUArray(ArraySizeType p_size) : Array<Type>(p_size) {
        CUDA_ASSERT(cudaMalloc(&this->m_data, this->size() * sizeof(Type)));
    }
    virtual ~GPUArray() { CUDA_ASSERT(cudaFree(this->m_data)); }

    void upload(CPUArray<Type>& cpu_array) {
        CUDA_ASSERT(cudaMemcpy(this->m_data,
                               cpu_array.m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyHostToDevice));
    }
    void download(CPUArray<Type>& cpu_array) {
        CUDA_ASSERT(cudaMemcpy(cpu_array.m_data,
                               this->m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyDeviceToHost));
    }

    void copyFrom(const GPUArray<Type>& other) {
        ASSERT(other.size() == this->size());
        cudaMemcpy(this->m_data, other.m_data, this->size() * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    void clear() { cudaMemset(this->m_data, 0, sizeof(Type) * this->size()); }
};

#endif    // CUDAD_SRC_DATA_ARRAY_H_
