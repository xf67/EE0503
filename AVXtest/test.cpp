#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <cstdint>

// Initialize matrix with random integer values
void initializeMatrix(int8_t* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = rand() % 256 - 128;  // Random values between -128 and 127
    }
}

// Print matrix
void printMatrix(const char* name, int32_t* matrix, int size) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Matrix multiplication using AVX-512 for int8_t matrices
void matrixMultiplyAVX512(int8_t* a, int8_t* b, int32_t* c, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m512i sum = _mm512_setzero_si512();
            for (int k = 0; k < size; k += 64) {  // Process 64 elements per iteration
                __m512i vecA = _mm512_loadu_si512(reinterpret_cast<__m512i*>(a + i * size + k));
                __m512i vecB = _mm512_set1_epi8(b[k * size + j]);
                sum = _mm512_add_epi32(sum, _mm512_maddubs_epi16(vecA, vecB));  // Multiply and add packed 8-bit integers
            }
            c[i * size + j] = _mm512_reduce_add_epi32(sum);
        }
    }
}

// Matrix multiplication using AVX2 for int8_t matrices

int32_t _mm256_reduce_add_epi32(__m256i vec) {
    __m256i hsum = _mm256_hadd_epi32(vec, vec);
    hsum = _mm256_hadd_epi32(hsum, hsum);
    __m128i sum_high = _mm256_extracti128_si256(hsum, 1);
    __m128i sum_low = _mm256_castsi256_si128(hsum);
    __m128i sum128 = _mm_add_epi32(sum_high, sum_low);
    return _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);
}
void matrixMultiplyAVX2(int8_t* a, int8_t* b, int32_t* c, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m256i sum = _mm256_setzero_si256();
            for (int k = 0; k < size; k += 32) {  // Process 32 elements per iteration
                __m256i vecA = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * size + k));
                __m256i vecB = _mm256_set1_epi8(b[k * size + j]);
                sum = _mm256_add_epi32(sum, _mm256_maddubs_epi16(vecA, vecB));  // Multiply and add packed 8-bit integers
            }
            c[i * size + j] = _mm256_reduce_add_epi32(sum);
        }
    }
}

int main() {
    const int size = 32;  // 32x32 matrix
    int8_t a[size*size], b[size*size];
    int32_t c[size*size] = {0};

    initializeMatrix(a, size);
    initializeMatrix(b, size);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++){
        // matrixMultiplyAVX512(a, b, c, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "AVX-512 took " << diff.count() << " seconds." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; i++){
        matrixMultiplyAVX2(a, b, c, size);
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "AVX-2 took " << diff.count() << " seconds." << std::endl;

    return 0;
}
