//#include <cstring>
//
//#include "operators.h"
//#include "utils.h"
//
//void load_Embedding_params(Embedding& op, std::string prefix) {
//    op.lookup.load((prefix + "/weight.bin").c_str());
//    // read_to_array((prefix + "/weight.bin").c_str(), op.lookup.m_data, op.lookup.length());
//}
//
//void Embedding::forward(Matrix3D<int> input_id, Matrix3D<float> output) {
//    PROFILE_START(profile_name);
//    assert(input_id.m_dim_x == 1);
//    assert(input_id.m_dim_y == 1);
//    assert(input_id.m_dim_z == output.m_dim_y);
//    assert(output.m_dim_z == this->embed_dim);
//
//    for (int i = 0; i < input_id.m_dim_z; i++) {
//        int token_id = input_id(0, 0, i);
//        float* output_sample_ptr = &output.m_data[i * this->embed_dim];
//        float* target_embed = &this->lookup.m_data[token_id * this->embed_dim];
//        memcpy(output_sample_ptr, target_embed, sizeof(float) * this->embed_dim);
//    }
//    PROFILE_END(profile_name);
//}
#include <fstream>
#include <cstring>

#include "operators.h"
#include "utils.h"
#include <time.h>

std::string my_prefix;

void load_Embedding_params(Embedding& op, std::string prefix) {
    // 不再加载整个嵌入表进内存
    my_prefix=prefix;
}

void Embedding::forward(Matrix3D<int> input_id, Matrix3D<float> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    std::ifstream embedding_file(my_prefix+"/weight.bin", std::ios::binary);

    if (!embedding_file) {
        throw std::runtime_error("Failed to open embedding file.");
    }

    for (int i = 0; i < input_id.m_dim_z; i++) {
        int token_id = input_id(0, 0, i);
        float* output_sample_ptr = &output.m_data[i * this->embed_dim];

        std::streampos offset = token_id * this->embed_dim * sizeof(float);

        // set timer
        // struct timespec start, end;
        // clock_gettime(CLOCK_MONOTONIC, &start);

        // 移动文件指针到正确位置并读取嵌入向量
        embedding_file.seekg(offset, std::ios::beg);
        embedding_file.read(reinterpret_cast<char*>(output_sample_ptr), sizeof(float) * this->embed_dim);

        // end timer
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        // fprintf(stderr, "Read embedding %d in %fs\n", token_id, time);


        if (embedding_file.fail()) {
            throw std::runtime_error("Failed to read embedding from file.");
        }
    }

    embedding_file.close();
    PROFILE_END(profile_name);
}
