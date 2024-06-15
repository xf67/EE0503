#include <unordered_map>
#include <deque>
#include <vector>
#include <fstream>
#include <cstring>
#include <ctime>
#include <iostream>
#include <cassert>
#include <algorithm> 
#include "operators.h"
#include "utils.h"

std::string my_prefix;


class EmbeddingCache {
   public:
    EmbeddingCache(size_t max_memory) : max_memory_(max_memory), current_memory_(0) {}

    bool get(int token_id, float* embedding, size_t embedding_size) {
        auto it = cache_.find(token_id);
        if (it != cache_.end()) {
            memcpy(embedding, it->second.data(), embedding_size * sizeof(float));
            update_access_order(token_id);
            return true;
        }
        return false;
    }

    void put(int token_id, const float* embedding, size_t embedding_size) {
        fprintf(stderr, "\nSize: %d | Max: %d\n", current_memory_ + embedding_size * sizeof(float), max_memory_);
        if (current_memory_ + embedding_size * sizeof(float) > max_memory_) {
            evict();
        }

        std::vector<float> embedding_vec(embedding, embedding + embedding_size);
        cache_[token_id] = embedding_vec;
        access_order_.push_back(token_id);
        current_memory_ += embedding_size * sizeof(float);
    }


   private:
    void evict() {
        // 如果evit，显示在命令行
        
        if (access_order_.empty()) return;

        int token_id = access_order_.front();
        access_order_.pop_front();
        fprintf(stderr, "\nEvicting token ID %d\n",token_id);
        current_memory_ -= cache_[token_id].size() * sizeof(float);
        
        cache_.erase(token_id);
        shrinkCache();
        // std::vector<float>().swap(cache_[token_id]);
    }

    void update_access_order(int token_id) {
        access_order_.erase(std::remove(access_order_.begin(), access_order_.end(), token_id), access_order_.end());
        access_order_.push_back(token_id);
    }

    void shrinkCache() {
        std::unordered_map<int, std::vector<float>> temp(cache_);
        cache_.swap(temp);
    }

    size_t max_memory_;
    size_t current_memory_;
    std::unordered_map<int, std::vector<float>> cache_;
    std::deque<int> access_order_;
};

static const size_t MAX_MEMORY = 1024 * 1024 * 1; // 1M
static EmbeddingCache global_cache(MAX_MEMORY);

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

        if (!global_cache.get(token_id, output_sample_ptr, this->embed_dim)) {
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

            global_cache.put(token_id, output_sample_ptr, this->embed_dim);
        }
    }

    embedding_file.close();
    PROFILE_END(profile_name);
}

void load_Embedding_params(Embedding &op, std::string prefix) {
    // 不再加载整个嵌入表进内存
    my_prefix = prefix;
}