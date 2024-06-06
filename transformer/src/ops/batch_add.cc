/*
#include "operators.h"

void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2,Matrix3D<float> &output) {
    PROFILE_START("batch_Add");
    assert(input.m_dim_y == input2.m_dim_y);
    assert(input.m_dim_z == input2.m_dim_z);
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_y);
    assert(input.m_dim_z == output.m_dim_z);

    if (input.m_dim_x != input2.m_dim_x && input2.m_dim_x == 1) {
        // Find the maximum value in the input array
        for (int i = 0; i < input.m_dim_x; i++) {
            for (int j = 0; j < input.m_dim_y; j++) {
                for (int k = 0; k < input.m_dim_z; k++){
                    output(i, j, k) = input(i, j, k) + input2(0, j, k);
                }
            }
        }
    } else {
        throw("Unsupported dimension for softmax");
    }
    PROFILE_END("batch_Add");
}
*/

#include "operators.h"

void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output) {
    PROFILE_START("batch_Add");
    assert(input.m_dim_y == input2.m_dim_y);
    assert(input.m_dim_z == input2.m_dim_z);
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_y);
    assert(input.m_dim_z == output.m_dim_z);

    const int x_dim = input.m_dim_x;
    const int y_dim = input.m_dim_y;
    const int z_dim = input.m_dim_z;
    const float *input2_data = &input2(0, 0, 0);

    for (int i = 0; i < x_dim; i += 4) {
        for (int j = 0; j < y_dim; ++j) {
            for (int k = 0; k < z_dim; ++k) {
                output(i + 0, j, k) = input(i + 0, j, k) + input2_data[j * z_dim + k];
                output(i + 1, j, k) = input(i + 1, j, k) + input2_data[j * z_dim + k];
                output(i + 2, j, k) = input(i + 2, j, k) + input2_data[j * z_dim + k];
                output(i + 3, j, k) = input(i + 3, j, k) + input2_data[j * z_dim + k];
            }
        }
    }
    for (int i = x_dim - x_dim % 4; i < x_dim; ++i) {
        for (int j = 0; j < y_dim; ++j) {
            for (int k = 0; k < z_dim; ++k) {
                output(i, j, k) = input(i, j, k) + input2(0, j, k);
            }
        }
    }

    PROFILE_END("batch_Add");
}
