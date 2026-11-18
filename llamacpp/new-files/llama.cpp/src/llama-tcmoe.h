#pragma once

// Reader for the tcmoe GGUF extension (task-conditioned quantization patch tables).
// Format: tcmoe/spec/TCMOE-GGUF-PATCH-SPEC.md

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

enum tcmoe_proj {
    TCMOE_PROJ_GATE = 0,
    TCMOE_PROJ_UP,
    TCMOE_PROJ_DOWN,
    TCMOE_PROJ_GATE_UP,
    TCMOE_PROJ_COUNT,
};

struct tcmoe_config {
    uint32_t    version   = 0;
    bool        enabled   = false;
    std::string base_mode;    // "tcmoe_q2" | "tcmoe_q2nl" | "external"
    std::string patch_type;   // "q4_0_rows" | "q4_k_rows"
    uint32_t    group     = 0;
    uint32_t    n_expert  = 0; // optional, used when the base expert tensors are absent
    float       patch_frac = 0.0f;
    std::string sensor;
    std::string default_domain;
    std::vector<std::string> domains;
};

struct tcmoe_proj_info {
    bool         present      = false;
    ggml_type    type         = GGML_TYPE_COUNT;
    int64_t      ne[3]        = {0, 0, 0};
    size_t       off          = 0; // absolute byte offset of the tensor data in the file
    size_t       row_bytes    = 0; // bytes of one ne[0] row
    size_t       expert_bytes = 0; // bytes of one expert slice (row_bytes * ne[1])
};

struct tcmoe_patch_table {
    int32_t             layer  = -1;
    int32_t             domain = -1;
    std::vector<int32_t> idx;   // strictly increasing expert ids
    tcmoe_proj_info      proj[TCMOE_PROJ_COUNT];
    std::vector<int32_t> pos;   // expert id -> patch position, -1 when not patched
};

// parses "blk.<layer>.tcmoe.<what>.<domain>", returns false for other names
bool tcmoe_parse_name(const char * name, int & layer, std::string & what, int & domain);

// maps the '<what>' part of a tcmoe tensor name to a projection, -1 when unknown
int tcmoe_proj_from_name(const std::string & what);

struct tcmoe_model {
    tcmoe_model() = default;
    ~tcmoe_model();

    tcmoe_model(const tcmoe_model &) = delete;
    tcmoe_model & operator=(const tcmoe_model &) = delete;

    // Loads metadata and patch table locations. Returns false and fills err on validation failure.
    bool load(const char * fname, std::string & err);
    void close();

    bool                 enabled() const { return cfg.enabled; }
    const tcmoe_config & config()  const { return cfg; }
    int                  n_layer() const { return n_layers; }
    int                  n_expert(int layer) const;
    int                  n_domains() const { return (int) cfg.domains.size(); }
    int                  domain_id(const std::string & name) const; // -1 if unknown

    bool is_registered_tensor(const std::string & name) const;

    const tcmoe_patch_table * table(int layer, int domain) const;
    int                      pos_of(int layer, int domain, int expert) const; // -1: no patch

    // Shape of one patched expert slice (ne0 * ne1 values, row-major).
    bool patch_shape(int layer, tcmoe_proj proj, int domain, int64_t & n0, int64_t & n1) const;

    // Dequantizes one patched expert to f32 (ne0 * ne1 values, row-major).
    bool read_patch_f32(int layer, tcmoe_proj proj, int domain, int expert, float * out,
                        int64_t & n0, int64_t & n1) const;

private:
    FILE *                  file = nullptr;
    struct gguf_context *  ctx  = nullptr;

    tcmoe_config cfg;
    int          n_layers = 0;

    std::vector<std::vector<tcmoe_patch_table>> tables; // [layer][domain]
    std::vector<int32_t>                        n_expert_layer;
    std::vector<std::string>                    tcmoe_tensors;

    bool parse_kv(std::string & err);
    bool read_tables(std::string & err);
};
