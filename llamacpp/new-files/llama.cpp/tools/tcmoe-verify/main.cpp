// Dumps patched expert weights from a tcmoe GGUF so they can be compared bit-for-bit
// against the reference quantizer (coserving/moe_lib.quantize_int4).

#include "llama.h"
#include "llama-tcmoe.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct sample {
    int    layer;
    int    proj;
    int    expert;
};

static int parse_proj(const char * s) {
    if (std::strcmp(s, "gate")    == 0) return TCMOE_PROJ_GATE;
    if (std::strcmp(s, "up")      == 0) return TCMOE_PROJ_UP;
    if (std::strcmp(s, "down")    == 0) return TCMOE_PROJ_DOWN;
    if (std::strcmp(s, "gate_up") == 0) return TCMOE_PROJ_GATE_UP;
    return -1;
}

static void usage(const char * argv0) {
    std::printf(
        "usage: %s -m model.gguf --sample sample.txt [options]\n"
        "       %s -m model.gguf --nll --text text.txt [--domains 0,1] [--alt 100]\n"
        "\n"
        "  --sample FILE    text file with 'layer proj expert' lines (proj: gate|up|down|gate_up)\n"
        "  --domain NAME    domain to read patches from (default: first domain)\n"
        "  -o FILE          write dequantized patch weights as f32 (default: out.bin)\n"
        "  --manifest FILE  write the layout manifest (default: out.bin.manifest)\n"
        "  --info           print the extension metadata and exit\n"
        "\n"
        "  --nll            score a text per domain (teacher forcing, NLL per token)\n"
        "  --nll-prefix     score each position from one decode of its prompt prefix, as llama-server does\n"
        "  --text FILE      text to score (required by --nll)\n"
        "  --domains LIST   comma separated domain ids, -1 is the base experts only (default: all)\n"
        "  --alt N          run N extra passes, alternating the domains, and compare with the first pass\n"
        "  --n-tokens N     use at most N tokens of the text (default 64)\n"
        "\n"
        "  --bench          time mul_mat_id on the expert shapes, 25%% of the slots patched (no model needed)\n"
        "  --bench-iters N  timed iterations per case (default 50)\n"
        "  --bench-repeats N take the best of N timed runs per case (default 3)\n"
        "  --bench-tokens N tokens per pass (default 1, decode shape)\n"
        "  --bench-used N   expert slots per token (default 8)\n"
        "  --bench-threads N threads for the bench (default: hardware concurrency)\n",
        argv0, argv0);
}

struct nll_pass {
    double              nll = 0.0;   // mean NLL over the predicted tokens
    std::vector<double> nll_tok;     // NLL per predicted token, encodes the full softmax
    double              ms  = 0.0;
};

// the logits of a 1-token decode are the distribution for the next token, so the NLL of
// the next gold token is the log-sum-exp of the logits minus the logit of that token
static bool nll_pass_run(llama_context * ctx, const std::vector<llama_token> & tok, int domain, nll_pass & out) {
    const auto t0 = std::chrono::steady_clock::now();

    llama_context_tcmoe_set_domain(ctx, domain);
    llama_memory_clear(llama_get_memory(ctx), true);

    const llama_vocab * vocab   = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    out.nll_tok.assign(tok.empty() ? 0 : tok.size() - 1, 0.0);
    out.nll = 0.0;

    for (size_t i = 0; i + 1 < tok.size(); ++i) {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tok[i]), 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: llama_decode failed at token %zu\n", i);
            return false;
        }

        const float * logits = llama_get_logits(ctx);

        double mx = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            mx = std::max(mx, (double) logits[v]);
        }
        double sum = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            sum += std::exp((double) logits[v] - mx);
        }
        out.nll_tok[i] = mx + std::log(sum) - (double) logits[tok[i + 1]];
        out.nll += out.nll_tok[i];
    }

    if (!out.nll_tok.empty()) {
        out.nll /= (double) out.nll_tok.size();
    }

    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// score every position from a single decode of the prompt prefix, the shape the server uses
static bool nll_prefix_run(llama_context * ctx, const std::vector<llama_token> & tok, int domain, nll_pass & out) {
    const auto t0 = std::chrono::steady_clock::now();

    llama_context_tcmoe_set_domain(ctx, domain);

    const llama_vocab * vocab   = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    out.nll_tok.assign(tok.empty() ? 0 : tok.size() - 1, 0.0);
    out.nll = 0.0;

    for (size_t n = 1; n < tok.size(); ++n) {
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            const int32_t k = (int32_t) i;
            batch.token[k]    = tok[i];
            batch.pos[k]      = (llama_pos) i;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k]   = i + 1 == n;
        }
        batch.n_tokens = (int32_t) n;

        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: llama_decode failed at prefix %zu\n", n);
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);

        const float * logits = llama_get_logits_ith(ctx, (int32_t) n - 1);

        double mx = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            mx = std::max(mx, (double) logits[v]);
        }
        double sum = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            sum += std::exp((double) logits[v] - mx);
        }
        out.nll_tok[n - 1] = mx + std::log(sum) - (double) logits[tok[n]];
        out.nll += out.nll_tok[n - 1];
    }

    if (!out.nll_tok.empty()) {
        out.nll /= (double) out.nll_tok.size();
    }

    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

static bool read_text(const char * path, std::string & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open text file %s\n", path);
        return false;
    }
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    std::fclose(f);
    return true;
}

static std::vector<int> parse_domains(const char * s) {
    std::vector<int> res;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const long d = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        res.push_back((int) d);
        p = end;
        if (*p == ',') {
            ++p;
        }
    }
    return res;
}

static bool read_samples(const char * path, std::vector<sample> & out) {
    FILE * f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "error: cannot open sample file %s\n", path);
        return false;
    }
    char line[256];
    int  line_no = 0;
    while (std::fgets(line, sizeof(line), f)) {
        ++line_no;
        char * p = line;
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') {
            continue;
        }
        char proj_s[32] = {0};
        int  layer = -1;
        int  expert = -1;
        if (std::sscanf(p, "%d %31s %d", &layer, proj_s, &expert) != 3) {
            std::fprintf(stderr, "error: %s:%d: expected 'layer proj expert'\n", path, line_no);
            return false;
        }
        const int proj = parse_proj(proj_s);
        if (proj < 0) {
            std::fprintf(stderr, "error: %s:%d: unknown projection '%s'\n", path, line_no, proj_s);
            return false;
        }
        out.push_back({ layer, proj, expert });
    }
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------------
// expert matmul microbenchmark: mul_mat_id on the real expert shapes at decode shape.
// the patched experts are plain Q4_0 rows, so their speed is compared with the types a
// native Q4_K_M file uses (Q4_K for most experts, Q6_K for some, Q2_K for the 2bit variant).
// ---------------------------------------------------------------------------------

struct bench_shape {
    const char * name;
    int64_t      k; // input width of one expert slice
    int64_t      m; // output width of one expert slice
};

// E21: one fixed codebook for the tcmoe_q2nl bench rows (the dot kernel reads it from extra)
static const float bench_cb_q2nl[4] = { -0.78f, -0.215f, 0.225f, 0.78f };

static void bench_fill(ggml_tensor * t, const std::vector<float> & src, ggml_type type,
                       int64_t k, int64_t m, int n_expert) {
    const size_t row_size    = ggml_row_size(type, k);
    const size_t expert_size = row_size * (size_t) m;
    for (int e = 0; e < n_expert; ++e) {
        ggml_quantize_chunk(type, src.data() + (size_t) e * (size_t) k * (size_t) m,
                (char *) t->data + (size_t) e * expert_size, 0, m, k, nullptr);
    }
}

// one mul_mat_id (type_patch == GGML_TYPE_COUNT), the two-pass base+patch+add form, or the
// fused mul_mat_id_mixed node when fused is set
static double bench_moe(ggml_backend_t backend, const bench_shape & sh, int n_expert,
                        ggml_type type_base, ggml_type type_patch,
                        int n_used, int n_patch, int n_tok, int iters, int repeats, bool fused = false) {
    const int64_t k = sh.k;
    const int64_t m = sh.m;

    std::vector<float> src((size_t) k * (size_t) m * (size_t) n_expert);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float) (std::sin((double) i * 0.001) * 0.4 + ((int) (i % 11) - 5) * 0.02);
    }

    ggml_context * ctx = ggml_init({ (size_t) (ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 4), nullptr, true });
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_base, k, m, n_expert);
    ggml_tensor * ap  = type_patch == GGML_TYPE_COUNT ? nullptr : ggml_new_tensor_3d(ctx, type_patch, k, m, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * ids_p = ap ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok) : nullptr;

    ggml_tensor * out = nullptr;
    if (ap && fused) {
        out = ggml_mul_mat_id_mixed(ctx, as, ap, b, ids, ids_p);
    } else {
        out = ggml_mul_mat_id(ctx, as, b, ids);
        if (ap) {
            out = ggml_add(ctx, out, ggml_mul_mat_id(ctx, ap, b, ids_p));
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // the graph must exist before the allocation, otherwise the results have no buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "error: cannot allocate the bench tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    bench_fill(as, src, type_base, k, m, n_expert);
    if (type_base == GGML_TYPE_TCMOE_Q2NL) {
        as->extra = (void *) bench_cb_q2nl;
    }
    if (ap) {
        bench_fill(ap, src, type_patch, k, m, n_expert);
    }

    for (int i = 0; i < (int) (k * n_used * n_tok); ++i) {
        ((float *) b->data)[i] = (float) (std::sin((double) i * 0.01) * 0.5);
    }
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const bool patched = (s >= n_used - n_patch);
            ((int32_t *) ids->data)[t * n_used + s]   = patched ? -1 : s;
            if (ap) {
                ((int32_t *) ids_p->data)[t * n_used + s] = patched ? s - (n_used - n_patch) : -1;
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }

    // the timings on this machine swing a lot between runs, so report the best of several
    double ms = 1e9;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(backend, gf);
        }
        ms = std::min(ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (double) iters);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms;
}

// the fused node must reproduce mul_mat_id(base) + mul_mat_id(patch) + add exactly
static double check_mixed(ggml_backend_t backend, const bench_shape & sh, int n_expert,
                          ggml_type type_base, ggml_type type_patch, int n_used, int n_patch, int n_tok) {
    const int64_t k = sh.k;
    const int64_t m = sh.m;

    std::vector<float> src((size_t) k * (size_t) m * (size_t) n_expert);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float) (std::sin((double) i * 0.001) * 0.4 + ((int) (i % 11) - 5) * 0.02);
    }

    ggml_context * ctx = ggml_init({ (size_t) (ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 8), nullptr, true });
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_base,  k, m, n_expert);
    ggml_tensor * ap  = ggml_new_tensor_3d(ctx, type_patch, k, m, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * idp = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);

    ggml_tensor * ref = ggml_add(ctx, ggml_mul_mat_id(ctx, as, b, ids), ggml_mul_mat_id(ctx, ap, b, idp));
    ggml_tensor * mix = ggml_mul_mat_id_mixed(ctx, as, ap, b, ids, idp);

    ggml_cgraph * gf_ref = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_ref, ref);
    ggml_cgraph * gf_mix = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_mix, mix);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "error: cannot allocate the check tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    bench_fill(as, src, type_base,  k, m, n_expert);
    if (type_base == GGML_TYPE_TCMOE_Q2NL) {
        as->extra = (void *) bench_cb_q2nl;
    }
    bench_fill(ap, src, type_patch, k, m, n_expert);

    for (int i = 0; i < (int) (k * n_used * n_tok); ++i) {
        ((float *) b->data)[i] = (float) (std::sin((double) i * 0.01) * 0.5);
    }
    // one slot of every token is unused (both ids -1), the rest is split between the tables
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const bool patched = (s >= n_used - n_patch);
            const bool unused  = (s == n_used - 1 && !patched);
            ((int32_t *) ids->data)[t * n_used + s] = (patched || unused) ? -1 : s;
            ((int32_t *) idp->data)[t * n_used + s] = patched ? s - (n_used - n_patch) : -1;
        }
    }

    ggml_backend_graph_compute(backend, gf_ref);
    ggml_backend_graph_compute(backend, gf_mix);

    double max_diff = 0.0;
    for (int64_t i = 0; i < ggml_nelements(ref); ++i) {
        max_diff = std::max(max_diff, std::fabs((double) ((float *) ref->data)[i] - (double) ((float *) mix->data)[i]));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return max_diff;
}

static int bench_run(int iters, int n_tok, int n_used, int n_threads, int repeats) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    static const bench_shape shapes[] = {
        { "gate", 2048, 512 },
        { "up",   2048, 512 },
        { "down",  512, 2048 },
    };
    static const ggml_type ref_types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q2_K, GGML_TYPE_TCMOE_Q2, GGML_TYPE_TCMOE_Q2NL };

    const int n_patch = n_used / 4; // the deployed patch fraction is 25%
    const int n_expert = 8;         // eight experts per pass, the patch rows are re-indexed

    std::printf("bench: %d threads, %d experts per pass, %d tokens, %d patched, %d iterations x %d repeats\n",
            n_threads, n_used, n_tok, n_patch, iters, repeats);

    double ratio_two_q4_0_min = 1e9; // two-pass with Q4_0 patch rows against the native Q4_K expert
    double ratio_two_q4_k_min = 1e9; // two-pass with Q4_K patch rows
    double ratio_mix_q4_0_min = 1e9; // fused mul_mat_id_mixed, Q4_0 patch rows
    double ratio_mix_q4_k_min = 1e9; // fused mul_mat_id_mixed, Q4_K patch rows
    double ratio_ideal_k_min  = 1e9; // fused design with zero dispatch/barrier cost
    double ratio_same_min     = 1e9; // same-type single pass (patched rows kept in the base type)
    double max_diff           = 0.0; // fused node against the two-pass reference

    for (const auto & sh : shapes) {
        std::printf("\nbench %-4s k=%lld m=%lld (per pass of %d experts)\n", sh.name,
                (long long) sh.k, (long long) sh.m, n_used);
        std::printf("  %-22s %10s %12s %10s\n", "case", "ms/pass", "us/expert", "rel Q4_K");

        double ms_q4k   = 0.0;
        double ms_tcmoe = 0.0;
        for (const auto type : ref_types) {
            const double ms = bench_moe(backend, sh, n_expert, type, GGML_TYPE_COUNT, n_used, n_patch, n_tok, iters, repeats);
            if (type == GGML_TYPE_Q4_K) {
                ms_q4k = ms;
            }
            if (type == GGML_TYPE_TCMOE_Q2) {
                ms_tcmoe = ms;
            }
            std::printf("  %-22s %10.3f %12.2f %10s\n", ggml_type_name(type), ms,
                    ms * 1000.0 / (double) n_used, "-");
        }

        // the deployed forms: base pass carries the unpatched slots, patch pass the rest
        const double ms_two_q4_0 = bench_moe(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_0, n_used, n_patch, n_tok, iters, repeats);
        const double ms_two_q4_k = bench_moe(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_K, n_used, n_patch, n_tok, iters, repeats);
        std::printf("  %-22s %10.3f %12.2f %10.3f  (two-pass, Q4_0 patch)\n", "two-pass q4_0",
                ms_two_q4_0, ms_two_q4_0 * 1000.0 / (double) n_used, ms_q4k / ms_two_q4_0);
        std::printf("  %-22s %10.3f %12.2f %10.3f  (two-pass, Q4_K patch)\n", "two-pass q4_k",
                ms_two_q4_k, ms_two_q4_k * 1000.0 / (double) n_used, ms_q4k / ms_two_q4_k);

        // the fused node: one dispatch and one barrier for both expert tables
        const double ms_mix_q4_0 = bench_moe(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_0, n_used, n_patch, n_tok, iters, repeats, true);
        const double ms_mix_q4_k = bench_moe(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_K, n_used, n_patch, n_tok, iters, repeats, true);
        std::printf("  %-22s %10.3f %12.2f %10.3f  (mixed node, Q4_0 patch)\n", "mixed q4_0",
                ms_mix_q4_0, ms_mix_q4_0 * 1000.0 / (double) n_used, ms_q4k / ms_mix_q4_0);
        std::printf("  %-22s %10.3f %12.2f %10.3f  (mixed node, Q4_K patch)\n", "mixed q4_k",
                ms_mix_q4_k, ms_mix_q4_k * 1000.0 / (double) n_used, ms_q4k / ms_mix_q4_k);

        // the best the fused design can do: the base slots at the tcmoe_q2 rate and the
        // patched slots at the patch type rate, with no dispatch or barrier cost at all
        const double frac_patch = (double) n_patch / (double) n_used;
        const double ms_ideal_k = (1.0 - frac_patch)*ms_tcmoe + frac_patch*ms_q4k;
        std::printf("  %-22s %10.3f %12.2f %10.3f  (mixed bound, no overhead)\n", "ideal q4_k patch",
                ms_ideal_k, ms_ideal_k * 1000.0 / (double) n_used, ms_q4k / ms_ideal_k);

        // the diff between the fused node and the two-pass reference must be exactly zero
        const double diff_q4_0 = check_mixed(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_0, n_used, n_patch, n_tok);
        const double diff_q4_k = check_mixed(backend, sh, n_expert, GGML_TYPE_TCMOE_Q2, GGML_TYPE_Q4_K, n_used, n_patch, n_tok);
        std::printf("  %-22s %10.3g %12.3g\n", "check mixed diff", diff_q4_0, diff_q4_k);
        max_diff = std::max(max_diff, std::max(diff_q4_0, diff_q4_k));

        // relative speed: patched experts against the native Q4_K_M expert type
        const double ms_q4_0 = bench_moe(backend, sh, n_expert, GGML_TYPE_Q4_0, GGML_TYPE_COUNT, n_used, n_patch, n_tok, iters, repeats);
        const double rel_patch = ms_q4k / ms_q4_0; // > 1 means the patch kernel is faster
        std::printf("  %-22s %10.3f %12.2f %10.3f  (patch rows only)\n", "Q4_0 (patch rows)", ms_q4_0,
                ms_q4_0 * 1000.0 / (double) n_used, rel_patch);

        // the same-type variant keeps the patched rows in the base type, so one pass covers every slot
        const double rel_same = ms_q4k / ms_tcmoe;
        std::printf("  %-22s %10.3f %12.2f %10.3f  (same-type single pass)\n", "tcmoe_q2 (all slots)",
                ms_tcmoe, ms_tcmoe * 1000.0 / (double) n_used, rel_same);

        ratio_two_q4_0_min = std::min(ratio_two_q4_0_min, ms_q4k / ms_two_q4_0);
        ratio_two_q4_k_min = std::min(ratio_two_q4_k_min, ms_q4k / ms_two_q4_k);
        ratio_mix_q4_0_min = std::min(ratio_mix_q4_0_min, ms_q4k / ms_mix_q4_0);
        ratio_mix_q4_k_min = std::min(ratio_mix_q4_k_min, ms_q4k / ms_mix_q4_k);
        ratio_ideal_k_min  = std::min(ratio_ideal_k_min,  ms_q4k / ms_ideal_k);
        ratio_same_min     = std::min(ratio_same_min,     rel_same);
    }

    const bool summary_ok = ratio_mix_q4_k_min >= 0.95 && max_diff == 0.0;
    std::printf("\nbench summary: two-pass/native Q4_K ratio %.3f (Q4_0 patch) %.3f (Q4_K patch)\n",
            ratio_two_q4_0_min, ratio_two_q4_k_min);
    std::printf("               mixed/native Q4_K ratio    %.3f (Q4_0 patch) %.3f (Q4_K patch) -> %s (target >= 0.95)\n",
            ratio_mix_q4_0_min, ratio_mix_q4_k_min, summary_ok ? "PASS" : "FAIL");
    std::printf("               mixed bound/native Q4_K    %.3f (Q4_K patch, zero overhead)\n", ratio_ideal_k_min);
    std::printf("               same-type/native Q4_K ratio %.3f (upper bound of the all-same-type design)\n",
            ratio_same_min);
    std::printf("               fused node vs two-pass max |diff| %.3g\n", max_diff);

    ggml_backend_free(backend);
    return summary_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * model_path   = nullptr;
    const char * sample_path  = nullptr;
    const char * out_path     = "out.bin";
    const char * manifest_path = nullptr;
    const char * domain_name  = nullptr;
    const char * text_path    = nullptr;
    const char * domains_arg  = nullptr;
    bool         info_only    = false;
    bool         nll_mode     = false;
    bool         nll_prefix   = false;
    bool         bench_mode   = false;
    int          bench_iters  = 50;
    int          bench_repeats = 3;
    int          bench_tokens = 1;
    int          bench_used   = 8;
    int          bench_threads = (int) std::thread::hardware_concurrency();
    int          n_alt        = 0;
    int          n_tokens_max = 64;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            sample_path = argv[++i];
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (std::strcmp(argv[i], "--domain") == 0 && i + 1 < argc) {
            domain_name = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            info_only = true;
        } else if (std::strcmp(argv[i], "--nll") == 0) {
            nll_mode = true;
        } else if (std::strcmp(argv[i], "--nll-prefix") == 0) {
            nll_mode   = true;
            nll_prefix = true;
        } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text_path = argv[++i];
        } else if (std::strcmp(argv[i], "--domains") == 0 && i + 1 < argc) {
            domains_arg = argv[++i];
        } else if (std::strcmp(argv[i], "--alt") == 0 && i + 1 < argc) {
            n_alt = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n-tokens") == 0 && i + 1 < argc) {
            n_tokens_max = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
        } else if (std::strcmp(argv[i], "--bench-iters") == 0 && i + 1 < argc) {
            bench_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench-repeats") == 0 && i + 1 < argc) {
            bench_repeats = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench-tokens") == 0 && i + 1 < argc) {
            bench_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench-used") == 0 && i + 1 < argc) {
            bench_used = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench-threads") == 0 && i + 1 < argc) {
            bench_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (bench_mode) {
        return bench_run(bench_iters, bench_tokens, bench_used, bench_threads, bench_repeats);
    }

    if (!model_path) {
        usage(argv[0]);
        return 1;
    }

    tcmoe_model model;
    std::string err;
    if (!model.load(model_path, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const tcmoe_config & cfg = model.config();
    if (!model.enabled()) {
        std::fprintf(stderr, "error: %s has no enabled tcmoe extension\n", model_path);
        return 1;
    }

    std::printf("tcmoe: version %u, base_mode '%s', patch_type '%s', group %u, n_layer %d\n",
                cfg.version, cfg.base_mode.c_str(), cfg.patch_type.c_str(), cfg.group, model.n_layer());
    std::printf("tcmoe: domains:");
    for (int d = 0; d < model.n_domains(); ++d) {
        std::printf(" %d=%s", d, cfg.domains[d].c_str());
    }
    std::printf("\n");

    if (info_only) {
        for (int l = 0; l < model.n_layer(); ++l) {
            std::printf("layer %2d: n_expert %d", l, model.n_expert(l));
            for (int d = 0; d < model.n_domains(); ++d) {
                const tcmoe_patch_table * t = model.table(l, d);
                std::printf(" | %s:%zu", cfg.domains[d].c_str(), t ? t->idx.size() : (size_t) 0);
            }
            std::printf("\n");
        }
        return 0;
    }

    if (nll_mode) {
        if (!text_path) {
            std::fprintf(stderr, "error: --text is required with --nll\n");
            return 1;
        }

        std::vector<int> domains;
        if (domains_arg) {
            domains = parse_domains(domains_arg);
        } else {
            for (int d = 0; d < model.n_domains(); ++d) {
                domains.push_back(d);
            }
        }
        for (int d : domains) {
            if (d < -1 || d >= model.n_domains()) {
                std::fprintf(stderr, "error: domain %d is not declared (0..%d, -1 for the base experts)\n",
                        d, model.n_domains() - 1);
                return 1;
            }
        }

        std::string text;
        if (!read_text(text_path, text)) {
            return 1;
        }

        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;

        llama_model * lmodel = llama_model_load_from_file(model_path, mp);
        if (lmodel == nullptr) {
            std::fprintf(stderr, "error: llama.cpp cannot load %s\n", model_path);
            return 1;
        }

        const int n_domain_llama = llama_model_tcmoe_n_domain(lmodel);
        if (n_domain_llama != model.n_domains()) {
            std::fprintf(stderr, "error: the model declares %d tcmoe domains, the reader sees %d\n",
                    n_domain_llama, model.n_domains());
            llama_model_free(lmodel);
            return 1;
        }

        std::printf("tcmoe: llama.cpp sees %d domains:", n_domain_llama);
        for (int d = 0; d < n_domain_llama; ++d) {
            std::printf(" %d=%s", d, llama_model_tcmoe_domain_name(lmodel, d));
        }
        std::printf("\n");

        const llama_vocab * vocab = llama_model_get_vocab(lmodel);

        int32_t n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, false);
        if (n_tok < 0) {
            n_tok = -n_tok;
        }
        if (n_tok < 2) {
            std::fprintf(stderr, "error: the text is too short\n");
            llama_model_free(lmodel);
            return 1;
        }

        std::vector<llama_token> tok((size_t) n_tok);
        if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), n_tok, true, false) != n_tok) {
            std::fprintf(stderr, "error: tokenization changed size\n");
            llama_model_free(lmodel);
            return 1;
        }
        if (n_tokens_max > 0 && n_tok > n_tokens_max) {
            n_tok = n_tokens_max;
            tok.resize((size_t) n_tok);
        }
        std::printf("tcmoe: %zu tokens\n", tok.size());
        std::printf("tokens:");
        for (const auto t : tok) {
            std::printf(" %d", t);
        }
        std::printf("\n");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx     = (uint32_t) (tok.size() + 8);
        cp.n_batch   = (uint32_t) tok.size();
        cp.n_ubatch  = (uint32_t) tok.size();
        cp.n_seq_max = 1;

        llama_context * ctx = llama_init_from_model(lmodel, cp);
        if (ctx == nullptr) {
            std::fprintf(stderr, "error: cannot create the context\n");
            llama_model_free(lmodel);
            return 1;
        }

        int rc = 0;

        const auto run_pass = nll_prefix ? nll_prefix_run : nll_pass_run;

        // the first pass of every domain is its reference, the alternating passes below must match it
        std::vector<nll_pass> ref_per_domain(domains.size());
        for (size_t k = 0; k < domains.size(); ++k) {
            const int d = domains[k];
            nll_pass & r = ref_per_domain[k];
            if (!run_pass(ctx, tok, d, r)) {
                llama_free(ctx);
                llama_model_free(lmodel);
                return 1;
            }
            std::printf("nll: domain %2d (%-8s) %f  (%f ms)\n",
                    d, d < 0 ? "base" : llama_model_tcmoe_domain_name(lmodel, d), r.nll, r.ms);

            // per-token NLL, the server checks each position separately
            std::printf("nll_tok:");
            for (const auto v : r.nll_tok) {
                std::printf(" %.6f", v);
            }
            std::printf("\n");
        }

        if (n_alt > 0) {
            double max_diff = 0.0;
            double sum_ms   = 0.0;
            for (int p = 0; p < n_alt; ++p) {
                const size_t k = (size_t) p % domains.size();

                nll_pass r;
                if (!run_pass(ctx, tok, domains[k], r)) {
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    return 1;
                }

                double diff = 0.0;
                for (size_t i = 0; i < r.nll_tok.size(); ++i) {
                    diff = std::max(diff, std::fabs(r.nll_tok[i] - ref_per_domain[k].nll_tok[i]));
                }
                max_diff = std::max(max_diff, diff);
                sum_ms  += r.ms;
            }

            std::printf("alt: %d passes over %zu domains, mean %f ms/pass, max |dNLL| %e %s\n",
                    n_alt, domains.size(), sum_ms / (double) n_alt, max_diff,
                    max_diff < 1e-4 ? "PASS" : "FAIL");

            if (!(max_diff < 1e-4)) {
                rc = 1;
            }
        }

        llama_free(ctx);
        llama_model_free(lmodel);
        return rc;
    }

    if (!sample_path) {
        std::fprintf(stderr, "error: --sample is required\n");
        return 1;
    }

    int domain = domain_name ? model.domain_id(domain_name) : 0;
    if (domain < 0) {
        std::fprintf(stderr, "error: unknown domain '%s'\n", domain_name);
        return 1;
    }

    std::vector<sample> samples;
    if (!read_samples(sample_path, samples)) {
        return 1;
    }

    std::string manifest = manifest_path ? std::string(manifest_path) : std::string(out_path) + ".manifest";
    FILE * f_out = std::fopen(out_path, "wb");
    if (!f_out) {
        std::fprintf(stderr, "error: cannot write %s\n", out_path);
        return 1;
    }
    FILE * f_man = std::fopen(manifest.c_str(), "w");
    if (!f_man) {
        std::fprintf(stderr, "error: cannot write %s\n", manifest.c_str());
        std::fclose(f_out);
        return 1;
    }

    std::fprintf(f_man, "# tcmoe-verify v1\n");
    std::fprintf(f_man, "# model: %s\n", model_path);
    std::fprintf(f_man, "# domain: %d %s\n", domain, cfg.domains[domain].c_str());
    std::fprintf(f_man, "# columns: layer proj expert pos ne0 ne1 offset\n");

    int64_t total = 0;
    int     n_miss = 0;
    int     n_ok   = 0;
    for (const auto & s : samples) {
        if (model.pos_of(s.layer, domain, s.expert) < 0) {
            n_miss++;
            std::fprintf(f_man, "miss %d %d %d - 0 0 -1\n", s.layer, s.proj, s.expert);
            continue;
        }

        int64_t n0 = 0;
        int64_t n1 = 0;
        if (!model.patch_shape(s.layer, (tcmoe_proj) s.proj, domain, n0, n1)) {
            std::fprintf(stderr, "error: layer %d proj %d: no such patch projection\n", s.layer, s.proj);
            std::fclose(f_man);
            std::fclose(f_out);
            return 1;
        }

        std::vector<float> buf((size_t) (n0 * n1));
        if (!model.read_patch_f32(s.layer, (tcmoe_proj) s.proj, domain, s.expert, buf.data(), n0, n1)) {
            std::fprintf(stderr, "error: layer %d proj %d expert %d: read failed\n", s.layer, s.proj, s.expert);
            std::fclose(f_man);
            std::fclose(f_out);
            return 1;
        }

        if (std::fwrite(buf.data(), sizeof(float), buf.size(), f_out) != buf.size()) {
            std::fprintf(stderr, "error: write to %s failed\n", out_path);
            std::fclose(f_man);
            std::fclose(f_out);
            return 1;
        }

        std::fprintf(f_man, "ok %d %d %d %d %lld %lld %lld\n", s.layer, s.proj, s.expert,
                     model.pos_of(s.layer, domain, s.expert), (long long) n0, (long long) n1,
                     (long long) total);
        total += (int64_t) buf.size() * (int64_t) sizeof(float);
        n_ok++;
    }

    std::fclose(f_man);
    std::fclose(f_out);

    std::printf("tcmoe: %d samples, %d dumped, %d not patched, %lld bytes -> %s\n",
                (int) samples.size(), n_ok, n_miss, (long long) total, out_path);
    return 0;
}
