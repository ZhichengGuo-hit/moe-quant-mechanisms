#include "llama-tcmoe.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

static bool tcmoe_seek(FILE * f, int64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, off, SEEK_SET) == 0;
#else
    return fseeko(f, off, SEEK_SET) == 0;
#endif
}

// blk.<layer>.tcmoe.<what>.<domain>
bool tcmoe_parse_name(const char * name, int & layer, std::string & what, int & domain) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return false;
    }

    char * end = nullptr;
    const long l = std::strtol(name + 4, &end, 10);
    if (end == name + 4 || l < 0 || l > 100000) {
        return false;
    }
    if (std::strncmp(end, ".tcmoe.", 7) != 0) {
        return false;
    }

    const char * s   = end + 7;
    const char * dot = std::strchr(s, '.');
    if (dot == nullptr || dot == s || dot[1] == '\0') {
        return false;
    }

    what.assign(s, dot - s);

    const long d = std::strtol(dot + 1, &end, 10);
    if (end == dot + 1 || *end != '\0' || d < 0) {
        return false;
    }

    layer  = (int) l;
    domain = (int) d;
    return true;
}

int tcmoe_proj_from_name(const std::string & what) {
    if (what == "gate")    return TCMOE_PROJ_GATE;
    if (what == "up")      return TCMOE_PROJ_UP;
    if (what == "down")    return TCMOE_PROJ_DOWN;
    if (what == "gate_up") return TCMOE_PROJ_GATE_UP;
    return -1;
}

tcmoe_model::~tcmoe_model() {
    close();
}

void tcmoe_model::close() {
    if (ctx) {
        gguf_free(ctx);
        ctx = nullptr;
    }
    if (file) {
        std::fclose(file);
        file = nullptr;
    }
}

int tcmoe_model::n_expert(int layer) const {
    if (layer < 0 || layer >= (int) n_expert_layer.size()) {
        return 0;
    }
    return n_expert_layer[layer];
}

int tcmoe_model::domain_id(const std::string & name) const {
    for (int i = 0; i < (int) cfg.domains.size(); ++i) {
        if (cfg.domains[i] == name) {
            return i;
        }
    }
    return -1;
}

bool tcmoe_model::is_registered_tensor(const std::string & name) const {
    for (const auto & t : tcmoe_tensors) {
        if (t == name) {
            return true;
        }
    }
    return false;
}

const tcmoe_patch_table * tcmoe_model::table(int layer, int domain) const {
    if (layer < 0 || layer >= (int) tables.size()) {
        return nullptr;
    }
    if (domain < 0 || domain >= (int) tables[layer].size()) {
        return nullptr;
    }
    const auto & t = tables[layer][domain];
    return t.idx.empty() ? nullptr : &t;
}

int tcmoe_model::pos_of(int layer, int domain, int expert) const {
    const tcmoe_patch_table * t = table(layer, domain);
    if (!t || expert < 0 || expert >= (int) t->pos.size()) {
        return -1;
    }
    return t->pos[expert];
}

bool tcmoe_model::parse_kv(std::string & err) {
    const auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
        const int64_t id = gguf_find_key(ctx, key);
        return id < 0 ? def : gguf_get_val_u32(ctx, id);
    };

    const auto get_f32 = [&](const char * key, float def) -> float {
        const int64_t id = gguf_find_key(ctx, key);
        return id < 0 ? def : gguf_get_val_f32(ctx, id);
    };

    const auto get_str = [&](const char * key, const char * def) -> std::string {
        const int64_t id = gguf_find_key(ctx, key);
        return id < 0 ? std::string(def) : std::string(gguf_get_val_str(ctx, id));
    };

    const int64_t id_enabled = gguf_find_key(ctx, "tcmoe.enabled");
    if (id_enabled < 0) {
        // plain GGUF without the extension
        cfg.enabled = false;
        return true;
    }

    cfg.version      = get_u32("tcmoe.version", 0);
    cfg.enabled      = gguf_get_val_bool(ctx, id_enabled);
    cfg.base_mode    = get_str("tcmoe.base_mode", "");
    cfg.patch_type   = get_str("tcmoe.patch_type", "");
    cfg.group        = get_u32("tcmoe.group", 128);
    cfg.n_expert     = get_u32("tcmoe.n_expert", 0);
    cfg.patch_frac   = get_f32("tcmoe.patch_frac", 0.0f);
    cfg.sensor       = get_str("tcmoe.sensor", "");
    cfg.default_domain = get_str("tcmoe.default_domain", "global");

    const int64_t id_domains = gguf_find_key(ctx, "tcmoe.domains");
    if (id_domains < 0) {
        err = "tcmoe.domains missing";
        return false;
    }
    if (gguf_get_kv_type(ctx, id_domains) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, id_domains) != GGUF_TYPE_STRING) {
        err = "tcmoe.domains must be a string array";
        return false;
    }
    const size_t n_domains = gguf_get_arr_n(ctx, id_domains);
    for (size_t i = 0; i < n_domains; ++i) {
        cfg.domains.push_back(gguf_get_arr_str(ctx, id_domains, i));
    }
    if (cfg.domains.empty()) {
        err = "tcmoe.domains is empty";
        return false;
    }

    if (!cfg.enabled) {
        return true;
    }
    if (cfg.version != 1) {
        err = "unsupported tcmoe.version " + std::to_string(cfg.version);
        return false;
    }
    if (cfg.base_mode != "tcmoe_q2" && cfg.base_mode != "tcmoe_q2nl" && cfg.base_mode != "external") {
        err = "unsupported tcmoe.base_mode '" + cfg.base_mode + "'";
        return false;
    }
    if (cfg.patch_type != "q4_0_rows" && cfg.patch_type != "q4_k_rows") {
        err = "unsupported tcmoe.patch_type '" + cfg.patch_type + "'";
        return false;
    }
    if (cfg.patch_type == "q4_0_rows" && cfg.group == 0) {
        err = "tcmoe.group must be non-zero";
        return false;
    }
    return true;
}

bool tcmoe_model::read_tables(std::string & err) {
    const int n_tensors = (int) gguf_get_n_tensors(ctx);

    int n_layer_max = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        std::string  what;
        int          layer = -1;
        int          domain = -1;
        if (!tcmoe_parse_name(name, layer, what, domain)) {
            continue;
        }
        tcmoe_tensors.emplace_back(name);
        n_layer_max = std::max(n_layer_max, layer + 1);
    }

    if (tcmoe_tensors.empty()) {
        if (cfg.enabled) {
            err = "tcmoe.enabled but no tcmoe tensors found";
            return false;
        }
        return true;
    }

    n_layers = n_layer_max;
    tables.assign(n_layers, std::vector<tcmoe_patch_table>(cfg.domains.size()));
    for (int l = 0; l < n_layers; ++l) {
        for (int d = 0; d < (int) cfg.domains.size(); ++d) {
            tables[l][d].layer  = l;
            tables[l][d].domain = d;
        }
    }

    const size_t data_offset = gguf_get_data_offset(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        std::string  what;
        int          layer = -1;
        int          domain = -1;
        if (!tcmoe_parse_name(name, layer, what, domain)) {
            continue;
        }
        if (domain >= (int) cfg.domains.size()) {
            err = "tensor " + std::string(name) + " uses domain id out of range";
            return false;
        }

        const enum ggml_type type = gguf_get_tensor_type(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        auto & t = tables[layer][domain];

        if (what == "idx") {
            if (type != GGML_TYPE_I32) {
                err = "tensor " + std::string(name) + " must be I32";
                return false;
            }
            if (ne[1] != 1 || ne[2] != 1) {
                err = "tensor " + std::string(name) + " must be 1-D";
                return false;
            }
            t.idx.resize(ne[0]);
            const size_t off = data_offset + gguf_get_tensor_offset(ctx, i);
            if (!tcmoe_seek(file, (int64_t) off) ||
                std::fread(t.idx.data(), sizeof(int32_t), t.idx.size(), file) != t.idx.size()) {
                err = "failed to read " + std::string(name);
                return false;
            }
            continue;
        }

        const int proj = tcmoe_proj_from_name(what);
        if (proj < 0) {
            err = "unknown tcmoe tensor " + std::string(name);
            return false;
        }

        tcmoe_proj_info & info = t.proj[proj];
        info.present = true;
        info.type = type;
        info.ne[0] = ne[0];
        info.ne[1] = ne[1];
        info.ne[2] = ne[2];
        info.off   = data_offset + gguf_get_tensor_offset(ctx, i);
        info.row_bytes    = ggml_row_size(type, ne[0]);
        info.expert_bytes = info.row_bytes * (size_t) ne[1];
    }

    // validate idx and build the expert -> position maps
    n_expert_layer.assign(n_layers, 0);

    // take n_expert and standard dims from the base expert tensors when present
    static const char * expert_suffix[3] = {
        ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight",
    };

    std::vector<std::array<int64_t, 3>> std_ne[3];
    for (int p = 0; p < 3; ++p) {
        std_ne[p].assign(n_layers, std::array<int64_t, 3>{0, 0, 0});
    }

    for (int l = 0; l < n_layers; ++l) {
        for (int p = 0; p < 3; ++p) {
            const std::string name = "blk." + std::to_string(l) + expert_suffix[p];
            const int64_t id = gguf_find_tensor(ctx, name.c_str());
            if (id < 0) {
                continue;
            }
            const int64_t * ne = gguf_get_tensor_ne(ctx, id);
            std_ne[p][l] = { ne[0], ne[1], ne[2] };
            if (n_expert_layer[l] == 0) {
                n_expert_layer[l] = (int32_t) ne[2];
            } else if (n_expert_layer[l] != (int32_t) ne[2]) {
                err = "layer " + std::to_string(l) + ": inconsistent n_expert across expert tensors";
                return false;
            }
        }

        for (int d = 0; d < (int) cfg.domains.size(); ++d) {
            auto & t = tables[l][d];

            // range for idx values
            int64_t n_expert = n_expert_layer[l];
            if (n_expert == 0) {
                n_expert = cfg.n_expert;
                for (int32_t v : t.idx) {
                    n_expert = std::max<int64_t>(n_expert, (int64_t) v + 1);
                }
            } else {
                n_expert_layer[l] = (int32_t) n_expert;
            }

            for (size_t k = 0; k < t.idx.size(); ++k) {
                const int32_t v = t.idx[k];
                if (v < 0 || v >= n_expert) {
                    err = "layer " + std::to_string(l) + " domain " + cfg.domains[d] +
                          ": idx out of range";
                    return false;
                }
                if (k > 0 && v <= t.idx[k - 1]) {
                    err = "layer " + std::to_string(l) + " domain " + cfg.domains[d] +
                          ": idx is not strictly increasing";
                    return false;
                }
            }

            t.pos.assign((size_t) n_expert, -1);
            for (size_t k = 0; k < t.idx.size(); ++k) {
                t.pos[t.idx[k]] = (int32_t) k;
            }

            const int64_t n_patch = (int64_t) t.idx.size();
            const bool has_gate_up = t.proj[TCMOE_PROJ_GATE_UP].present;
            const bool has_split   = t.proj[TCMOE_PROJ_GATE].present && t.proj[TCMOE_PROJ_UP].present;
            const bool has_down    = t.proj[TCMOE_PROJ_DOWN].present;

            if (n_patch > 0 && !has_down) {
                err = "layer " + std::to_string(l) + " domain " + cfg.domains[d] + ": down patch missing";
                return false;
            }
            if (n_patch > 0 && !has_gate_up && !has_split) {
                err = "layer " + std::to_string(l) + " domain " + cfg.domains[d] +
                      ": gate/up patch missing";
                return false;
            }

            for (int p = 0; p < TCMOE_PROJ_COUNT; ++p) {
                auto & info = t.proj[p];
                if (!info.present) {
                    continue;
                }
                if (info.type != GGML_TYPE_Q4_0 && info.type != GGML_TYPE_Q4_K) {
                    err = "layer " + std::to_string(l) + ": patch tensor must be Q4_0 or Q4_K";
                    return false;
                }
                if (info.ne[0] % ggml_blck_size(info.type) != 0) {
                    err = "layer " + std::to_string(l) + ": patch ne0 is not a multiple of the block size";
                    return false;
                }
                if (info.ne[2] != n_patch) {
                    err = "layer " + std::to_string(l) + " domain " + cfg.domains[d] +
                          ": patch ne2 != len(idx)";
                    return false;
                }

                // split projections must match the base expert tensor shape
                const int sp = p == TCMOE_PROJ_GATE ? 0 : p == TCMOE_PROJ_UP ? 1 : p == TCMOE_PROJ_DOWN ? 2 : -1;
                if (sp >= 0 && std_ne[sp][l][0] != 0) {
                    if (info.ne[0] != std_ne[sp][l][0] || info.ne[1] != std_ne[sp][l][1]) {
                        err = "layer " + std::to_string(l) + ": patch shape does not match base expert tensor";
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

bool tcmoe_model::load(const char * fname, std::string & err) {
    close();
    tables.clear();
    tcmoe_tensors.clear();
    n_expert_layer.clear();
    cfg = tcmoe_config();

    file = std::fopen(fname, "rb");
    if (!file) {
        err = "failed to open " + std::string(fname);
        return false;
    }

    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    ctx = gguf_init_from_file_ptr(file, params);
    if (!ctx) {
        err = "failed to parse GGUF file " + std::string(fname);
        return false;
    }

    if (gguf_get_version(ctx) != 3) {
        err = "unsupported GGUF version";
        return false;
    }

    if (!parse_kv(err)) {
        return false;
    }
    if (!cfg.enabled) {
        return true;
    }
    return read_tables(err);
}

bool tcmoe_model::patch_shape(int layer, tcmoe_proj proj, int domain, int64_t & n0, int64_t & n1) const {
    const tcmoe_patch_table * t = table(layer, domain);
    if (!t || proj < 0 || proj >= TCMOE_PROJ_COUNT || !t->proj[proj].present) {
        return false;
    }
    n0 = t->proj[proj].ne[0];
    n1 = t->proj[proj].ne[1];
    return true;
}

bool tcmoe_model::read_patch_f32(int layer, tcmoe_proj proj, int domain, int expert, float * out,
                                 int64_t & n0, int64_t & n1) const {
    const tcmoe_patch_table * t = table(layer, domain);
    if (!t) {
        return false;
    }
    if (proj < 0 || proj >= TCMOE_PROJ_COUNT) {
        return false;
    }
    const tcmoe_proj_info & info = t->proj[proj];
    if (!info.present) {
        return false;
    }
    const int pos = pos_of(layer, domain, expert);
    if (pos < 0) {
        return false;
    }

    const ggml_type_traits * tr = ggml_get_type_traits(info.type);
    if (!tr || !tr->to_float) {
        return false;
    }

    std::vector<uint8_t> buf(info.expert_bytes);
    const int64_t off = (int64_t) info.off + (int64_t) pos * (int64_t) info.expert_bytes;
    if (!tcmoe_seek(file, off) || std::fread(buf.data(), 1, buf.size(), file) != buf.size()) {
        return false;
    }

    tr->to_float(buf.data(), out, info.ne[0] * info.ne[1]);
    n0 = info.ne[0];
    n1 = info.ne[1];
    return true;
}
