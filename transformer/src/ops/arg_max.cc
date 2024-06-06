
// #include "ops/arg_max.h"

// #include <cassert>

// void arg_max_dim2(Matrix3D<float> &input, Matrix3D<int> &output) {
//     int bz = input.m_dim_x;
//     int sqlen = input.m_dim_y;
//     int voc_size = input.m_dim_z;

//     assert(sqlen == output.m_dim_z);
//     assert(bz == output.m_dim_x);

//     for (int b = 0; b < bz; b++) {
//         for (int i = 0; i < sqlen; i++) {
//             float max = FLOAT_MIN;
//             int max_idx = -1;
//             for (int j = 0; j < voc_size; j++) {
//                 float v = input(b, i, j);
//                 if (max < v) {
//                     max = v;
//                     max_idx = j;
//                 }
//             }
//             output(b, 0, i) = max_idx;
//         }
//     }
// }




#include "ops/arg_max.h"

#include <immintrin.h>  // For SIMD intrinsics

#include <cassert>
#include <cfloat>

void arg_max_dim2(Matrix3D<float> &input, Matrix3D<int> &output) {
    int bz = input.m_dim_x;
    int sqlen = input.m_dim_y;
    int voc_size = input.m_dim_z;

    assert(sqlen == output.m_dim_z);
    assert(bz == output.m_dim_x);

    const int unroll_factor = 8;  // 循环展开因子，根据实际情况调整
    //  fprintf(stderr,"111");
    for (int b = 0; b < bz; b++) {
        for (int i = 0; i < sqlen; i++) {
            float max_values[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
            int max_indices[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

            // 处理可被展开因子整除的部分
            for (int j = 0; j < voc_size - voc_size % unroll_factor; j += unroll_factor) {
                __m256 max_values_vec = _mm256_set1_ps(-FLT_MAX);
                __m256i max_indices_vec = _mm256_setzero_si256();

                for (int k = 0; k < unroll_factor; ++k) {
                    __m256 input_vec = _mm256_loadu_ps(&input(b, i, j + k));

                    __m256i candidate_indices = _mm256_set1_epi32(j + k);
                    __m256i current_max_indices = _mm256_loadu_si256((__m256i *)&max_indices[k]);

                    __m256 mask = _mm256_cmp_ps(input_vec, max_values_vec, _CMP_GT_OS);
                    max_values_vec = _mm256_max_ps(input_vec, max_values_vec);
                    max_indices_vec =
                        _mm256_blendv_epi8(current_max_indices, candidate_indices, _mm256_castps_si256(mask));
                }

                _mm256_storeu_ps(&max_values[0], max_values_vec);
                _mm256_storeu_si256((__m256i *)&max_indices[0], max_indices_vec);

                for (int k = 0; k < unroll_factor; ++k) {
                    if (max_values[k] > max_values[0]) {
                        max_values[0] = max_values[k];
                        max_indices[0] = max_indices[k];
                    }
                }
            }

            // 处理不可被展开因子整除的部分
            for (int j = voc_size - voc_size % unroll_factor; j < voc_size; ++j) {
                float v = input(b, i, j);
                if (v > max_values[0]) {
                    max_values[0] = v;
                    max_indices[0] = j;
                }
            }

            output(b, 0, i) = max_indices[0];
        }
    }
}
