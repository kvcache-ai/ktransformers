#pragma once
#include <vector>

struct SFT_MoEForwardCache {
    // 每个 token 按 expert 分块保存
    std::vector<std::vector<float>> gate_u;   // u = W_gate x
    std::vector<std::vector<float>> up_v;     // v = W_up   x
    // 若希望反向直接用 z = σ(u)⊙v，则再加一份
    // std::vector<std::vector<float>> z;
    void init(int k, int inter_size) {
        /* ---- 只增不减：capacity 不够时才增，永不缩小，避免多线程情况下的use-after-free ---- */
       if (k > (int)gate_u.size()) {
            gate_u.resize(k);
            up_v  .resize(k);
            // z     .resize(k);
        }

        for (int i = 0; i < k; ++i) {
            if ((int)gate_u[i].capacity() < inter_size)
                gate_u[i].reserve(inter_size);   // 只增 capacity
            if ((int)up_v[i].capacity()   < inter_size)
                up_v[i].reserve(inter_size);
            // if ((int)z[i].capacity()      < inter_size)
            //     z[i].reserve(inter_size);

            // size() 更新为 inter_size 以便直接下标写入
            gate_u[i].resize(inter_size);
            up_v[i]  .resize(inter_size);
            // z[i]     .resize(inter_size);
        }
	}
};