/*
#include "ops/arg_max.h"

#include <cassert>

void arg_max_dim2(Matrix3D<float> &input, Matrix3D<int> &output) {
    int bz = input.m_dim_x;
    int sqlen = input.m_dim_y;
    int voc_size = input.m_dim_z;

    assert(sqlen == output.m_dim_z);
    assert(bz == output.m_dim_x);

    for (int b = 0; b < bz; b++) {
        for (int i = 0; i < sqlen; i++) {
            float max = FLOAT_MIN;
            int max_idx = -1;
            for (int j = 0; j < voc_size; j++) {
                float v = input(b, i, j);
                if (max < v) {
                    max = v;
                    max_idx = j;
                }
            }
            output(b, 0, i) = max_idx;
        }
    }
}
*/
#include "ops/arg_max.h"
#include <cassert>
#include <cfloat> // For FLT_MIN

void arg_max_dim2(Matrix3D<float> &input, Matrix3D<int> &output) {
    int bz = input.m_dim_x;
    int sqlen = input.m_dim_y;
    int voc_size = input.m_dim_z;

    assert(sqlen == output.m_dim_z);
    assert(bz == output.m_dim_x);

    const int unroll_factor = 4; // 循环展开因子

    for (int b = 0; b < bz; b++) {
        for (int i = 0; i < sqlen; i++) {
            float max = -FLT_MAX; // 使用FLT_MAX作为初始最小值
            int max_idx = -1;

            // 处理可被展开因子整除的部分
            for (int j = 0; j < voc_size - voc_size % unroll_factor; j += unroll_factor) {
                float v0 = input(b, i, j);
                float v1 = input(b, i, j + 1);
                float v2 = input(b, i, j + 2);
                float v3 = input(b, i, j + 3);

                if (v0 > max) { max = v0; max_idx = j; }
                if (v1 > max) { max = v1; max_idx = j + 1; }
                if (v2 > max) { max = v2; max_idx = j + 2; }
                if (v3 > max) { max = v3; max_idx = j + 3; }
            }

            for (int j = voc_size - voc_size % unroll_factor; j < voc_size; ++j) {
                float v = input(b, i, j);
                if (v > max) {
                    max = v;
                    max_idx = j;
                }
            }

            output(b, 0, i) = max_idx;
        }
    }
}