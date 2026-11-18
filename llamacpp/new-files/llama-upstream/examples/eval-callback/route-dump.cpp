// route-dump: 逐 token 导出 MoE 路由概率 (ffn_moe_probs) 与 teacher-forcing NLL。
//
// 目的：E-R2 路由一致性诊断。对同一份文本、同一 tokenizer，在不同量化配置下
// 记录每层 top-k 专家集合与完整路由分布，用于 Jaccard / 精确匹配 / JS 散度，
// 以及"路由翻转 token 的 NLL 增量"归因。
//
// 用法:
//   llama-route-dump -m model.gguf -c 4096 -b 512 -ngl 99 \
//       --texts texts.txt --out out.bin --tag iq2xxs
//   texts.txt 用一行 "%%__TEXT_BOUNDARY__%%" 分隔多段文本。
//
// 输出 (小端):
//   per text:
//     int32 magic       = 0x52544431 ('RTD1')
//     int32 text_idx
//     int32 n_tokens
//     int32 n_vocab
//     int32 n_layers
//     int32 tokens[n_tokens]
//     float nll[n_tokens]        // nll[i] = -log p(tokens[i+1] | tokens[0..i]); 末位 0
//     for each layer (按层号升序):
//       int32 layer
//       int32 ne0 (n_expert)
//       int32 ne1 (n_tokens)
//       float probs[ne0*ne1]     // ggml 行主序, ne0 连续
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const char * SEP = "%%__TEXT_BOUNDARY__%%";

struct dump_state {
    FILE * f = nullptr;
    int text_idx = 0;
    int n_layers = 0;
    int n_expert = 0;
    int n_tokens = 0;
    int n_vocab = 0;
    std::vector<int32_t> layer_ids;
    std::vector<std::vector<float>> layer_probs;
};

static bool route_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<dump_state *>(user_data);
    static const char * PFX = "ffn_moe_probs-";
    const size_t plen = strlen(PFX);
    if (ask) {
        return strncmp(t->name, PFX, plen) == 0;
    }
    if (strncmp(t->name, PFX, plen) != 0) {
        return true;
    }
    const int il = atoi(t->name + plen);
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t n = ne0 * ne1;
    std::vector<float> buf((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t) n);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) {
            buf[(size_t) i] = ggml_fp16_to_fp32(tmp[(size_t) i]);
        }
    } else {
        LOG_ERR("%s: unsupported probs type %s\n", __func__, ggml_type_name(t->type));
        return true;
    }
    st->layer_ids.push_back(il);
    st->layer_probs.push_back(std::move(buf));
    st->n_expert = (int) ne0;
    st->n_tokens = (int) ne1;
    return true;
}

static void wr(FILE * f, const void * p, size_t n) {
    if (fwrite(p, 1, n, f) != n) {
        LOG_ERR("fwrite failed\n");
        exit(2);
    }
}

static void wr_i32(FILE * f, int32_t v) { wr(f, &v, sizeof(v)); }

static std::vector<std::string> load_texts(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERR("cannot open texts: %s\n", path.c_str());
        exit(2);
    }
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= all.size()) {
        size_t hit = all.find(SEP, pos);
        if (hit == std::string::npos) {
            if (pos < all.size()) {
                out.push_back(all.substr(pos));
            }
            break;
        }
        out.push_back(all.substr(pos, hit - pos));
        pos = hit + strlen(SEP);
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string texts_path;
    std::string out_path;
    std::string tag;

    // 先摘出自定义参数，其余交给 common_params_parse
    std::vector<char *> args;
    args.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto take = [&](std::string & dst) {
            if (i + 1 >= argc) {
                LOG_ERR("%s needs a value\n", a.c_str());
                exit(2);
            }
            dst = argv[++i];
        };
        if (a == "--texts")      { take(texts_path); }
        else if (a == "--out")   { take(out_path); }
        else if (a == "--tag")   { take(tag); }
        else                     { args.push_back(argv[i]); }
    }
    if (texts_path.empty() || out_path.empty()) {
        LOG_ERR("usage: llama-route-dump -m model.gguf --texts FILE --out FILE [--tag NAME]\n");
        return 2;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.warmup = false;
    params.load_mtp = false;   // 只走主模型图
    // 路由概率由 llama_context 内部通过 LLAMA_MOE_DUMP 导出（避免 cb_eval 的逐节点同步开销）；
    // 本工具只负责 teacher-forcing NLL 与 token 序列。
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;

    dump_state st;
    params.cb_eval_user_data = &st;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    st.n_layers = (int) llama_model_n_layer(model);
    st.n_vocab  = (int) llama_vocab_n_tokens(vocab);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    llama_memory_t mem = llama_get_memory(ctx);

    auto texts = load_texts(texts_path);
    LOG_INF("%s: %zu texts, n_layers=%d, n_vocab=%d\n", __func__, texts.size(), st.n_layers, st.n_vocab);
    st.f = fopen(out_path.c_str(), "wb");
    if (!st.f) {
        LOG_ERR("cannot open out: %s\n", out_path.c_str());
        return 2;
    }

    const int n_max = (int) params.n_batch;
    for (size_t ti = 0; ti < texts.size(); ti++) {
        std::vector<llama_token> tokens = common_tokenize(ctx, texts[ti], add_bos, true);
        if (tokens.empty()) {
            LOG_WRN("text %zu empty, skip\n", ti);
            continue;
        }
        if ((int) tokens.size() > n_max) {
            LOG_WRN("text %zu len %zu > n_batch %d, truncate\n", ti, tokens.size(), n_max);
            tokens.resize(n_max);
        }
        const int n = (int) tokens.size();

        st.text_idx = (int) ti;
        st.layer_ids.clear();
        st.layer_probs.clear();
        st.n_expert = 0;
        llama_memory_clear(mem, true);

        common_batch batch(ctx);
        for (int p = 0; p < n; p++) {
            batch.add(tokens[p], p, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("decode failed at text %zu\n", ti);
            return 1;
        }

        // teacher-forcing NLL
        std::vector<float> nll((size_t) n, 0.0f);
        for (int p = 0; p + 1 < n; p++) {
            const float * lg = llama_get_logits_ith(ctx, p);
            if (!lg) { continue; }
            float mx = -INFINITY;
            for (int v = 0; v < st.n_vocab; v++) { mx = std::max(mx, lg[v]); }
            double se = 0.0;
            for (int v = 0; v < st.n_vocab; v++) { se += std::exp((double) lg[v] - mx); }
            const double lse = mx + std::log(se);
            nll[(size_t) p] = (float) (lse - (double) lg[tokens[p + 1]]);
        }

        // 写记录
        int32_t magic = 0x52544431;
        wr_i32(st.f, magic);
        wr_i32(st.f, (int32_t) ti);
        wr_i32(st.f, (int32_t) n);
        wr_i32(st.f, (int32_t) st.n_vocab);
        wr_i32(st.f, (int32_t) st.n_layers);
        wr(st.f, tokens.data(), tokens.size() * sizeof(llama_token));
        wr(st.f, nll.data(), nll.size() * sizeof(float));
        for (size_t li = 0; li < st.layer_ids.size(); li++) {
            const int ne0 = st.n_expert;
            const int ne1 = (int) (st.layer_probs[li].size() / (size_t) ne0);
            wr_i32(st.f, st.layer_ids[li]);
            wr_i32(st.f, ne0);
            wr_i32(st.f, ne1);
            wr(st.f, st.layer_probs[li].data(), st.layer_probs[li].size() * sizeof(float));
        }
        fflush(st.f);
        if ((ti + 1) % 8 == 0 || ti + 1 == texts.size()) {
            LOG_INF("[%s] %zu/%zu texts, last n_tokens=%d, layers=%zu\n",
                    tag.c_str(), ti + 1, texts.size(), n, st.layer_ids.size());
        }
    }
    fclose(st.f);
    LOG_INF("[%s] wrote %s\n", tag.c_str(), out_path.c_str());
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
