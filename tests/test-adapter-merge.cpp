// test-adapter-merge.cpp: runtime LoRA merge parity harness
//
// Loads the AR and NAR halves of a backbone GGUF with an adapter through the
// real load path, then checks a sample of patched tensors (standalone
// projections, fused qkv/gate_up slices, io projections) against an
// independent F32 reference computed straight from the GGUF mmap and the
// adapter factors. Exits non-zero on any mismatch beyond the target dtype's
// rounding.

#include "adapter-merge.h"
#include "nar.h"
#include "qwen3-lm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Reference delta: d[o][i] = sum_k up[out_off+o][k] * down[k][i]
static void ref_delta(const float * up, int64_t out_off, int64_t out, int64_t r, const float * down, int64_t in, float * d) {
    for (int64_t o = 0; o < out; o++) {
        float *         dst = d + o * in;
        const float * u = up + (out_off + o) * r;
        for (int64_t i = 0; i < in; i++) {
            float s = 0.0f;
            for (int64_t k = 0; k < r; k++) {
                s += u[k] * down[k * in + i];
            }
            dst[i] = s;
        }
    }
}

// Dequantize a GGUF tensor [in, out] row by row into F32 [out*in]
static bool read_tensor(const GGUFModel & gf, const char * name, float * dst, int64_t * in, int64_t * out) {
    int64_t idx = gguf_find_tensor(gf.gguf, name);
    if (idx < 0) {
        return false;
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name);
    const int64_t        i0  = src->ne[0], o0 = src->ne[1];
    const uint8_t *      raw = (const uint8_t *) (gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, idx));
    const struct ggml_type_traits * tr = ggml_get_type_traits(src->type);
    const size_t               row_size = ggml_row_size(src->type, i0);
    for (int64_t o = 0; o < o0; o++) {
        tr->to_float(raw + o * row_size, dst + o * i0, i0);
    }
    if (in) {
        *in = i0;
    }
    if (out) {
        *out = o0;
    }
    return true;
}

// Dequantize rows [row_off, row_off+n) of a merged backend tensor into F32
static bool get_rows(struct ggml_tensor * t, int64_t row_off, int64_t n, float * dst) {
    const int64_t in = t->ne[0];
    if (t->ne[1] < row_off + n) {
        return false;
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(t->type);
    const size_t               row_size = ggml_row_size(t->type, in);
    for (int64_t o = 0; o < n; o++) {
        tr->to_float((const uint8_t *) ggml_get_data(t) + (row_off + o) * row_size, dst + o * in, in);
    }
    return true;
}

static bool get_factors(const Yue2Adapter & ad, int branch, int layer, int proj,
                        std::vector<float> & down_f, std::vector<float> & up_f,
                        int64_t * r, int64_t * in, int64_t * out) {
    auto it = ad.pairs.find({ branch, layer, proj });
    if (it == ad.pairs.end()) {
        return false;
    }
    const auto & pr = it->second;
    *r   = pr.down->shape[0];
    *in  = pr.down->shape[1];
    *out = pr.up->shape[0];
    down_f.resize((size_t) *r * *in);
    up_f.resize((size_t) *out * *r);
    return adapter_f32(ad.st, *pr.down, down_f.data()) && adapter_f32(ad.st, *pr.up, up_f.data());
}

// One check: base + reference delta must match the merged tensor within tol.
struct Checker {
    GGUFModel &     gf;
    const char *    name;
    int64_t         in, out;
    float           tol;
    int             fails = 0;
    std::vector<float> base;
    std::vector<float> got;

    Checker(GGUFModel & g, const char * n) : gf(g), name(n) {}

    bool expect(int64_t exp_in, int64_t exp_out, float exp_tol) {
        struct ggml_tensor * t = ggml_get_tensor(gf.meta, name);
        if (!t) {
            fprintf(stderr, "[Test-Adapter] FAIL: %s not in the GGUF\n", name);
            fails++;
            return false;
        }
        if (t->ne[0] != exp_in || t->ne[1] != exp_out) {
            fprintf(stderr, "[Test-Adapter] FAIL: %s shape [%lld,%lld] != expected [%lld,%lld]\n", name,
                    (long long) t->ne[1], (long long) t->ne[0], (long long) exp_out, (long long) exp_in);
            fails++;
            return false;
        }
        this->in  = exp_in;
        this->out = exp_out;
        this->tol = exp_tol;
        base.resize((size_t) exp_in * exp_out);
        got.resize((size_t) exp_in * exp_out);
        read_tensor(gf, name, base.data(), nullptr, nullptr);
        return true;
    }

    void check(const float * delta, int64_t out_off, int64_t n_rows, struct ggml_tensor * merged,
               int64_t merged_row_off, const char * label) {
        for (int64_t o = 0; o < n_rows; o++) {
            for (int64_t i = 0; i < in; i++) {
                base[(size_t) o * in + i] += delta[(size_t) (out_off + o) * in + i];
            }
        }
        if (!get_rows(merged, merged_row_off, n_rows, got.data())) {
            fprintf(stderr, "[Test-Adapter] FAIL: cannot read merged %s (%s)\n", name, label);
            fails++;
            return;
        }
        float md = 0.0f;
        for (size_t i = 0; i < (size_t) in * n_rows; i++) {
            md = std::max(md, std::fabs(base[i] - got[i]));
        }
        const bool ok = md < tol;
        printf("[Test-Adapter] %-48s max|diff| = %.5g %s\n", label, md, ok ? "OK" : "MISMATCH");
        if (!ok) {
            fails++;
        }
    }
};

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s backbone.gguf adapter.safetensors\n", argv[0]);
        return 1;
    }
    auto * ad = new Yue2Adapter();
    if (!yue2_adapter_load(ad, argv[2])) {
        fprintf(stderr, "[Test-Adapter] FAIL: adapter load\n");
        return 1;
    }

    int fails = 0;

    // ---------- AR half ----------
    if (ad->has_ar) {
        Qwen3LM     lm;
        GGUFModel   gf;
        if (!qw3lm_load(&lm, argv[1], ad)) {
            fprintf(stderr, "[Test-Adapter] FAIL: AR load+merge\n");
            return 1;
        }
        if (!gf_load(&gf, argv[1])) {
            fprintf(stderr, "[Test-Adapter] FAIL: GGUF read\n");
            return 1;
        }

        // standalone projections across the layer range
        for (int layer : { 0, lm.cfg.n_layers / 2, lm.cfg.n_layers - 1 }) {
            for (int proj : { 1, 3 }) {
                char name[96];
                snprintf(name, sizeof(name), proj == 1 ? "model.layers.%d.self_attn.o_proj.weight"
                                                       : "model.layers.%d.mlp.down_proj.weight", layer);
                Checker c(gf, name);
                std::vector<float> down_f, up_f, delta;
                int64_t r, in, out;
                if (!get_factors(*ad, 0, layer, proj, down_f, up_f, &r, &in, &out)) {
                    fprintf(stderr, "[Test-Adapter] FAIL: factors for %s\n", name);
                    fails++;
                    continue;
                }
                delta.resize((size_t) in * out);
                ref_delta(up_f.data(), 0, out, r, down_f.data(), in, delta.data());
                char label[128];
                snprintf(label, sizeof(label), "AR layer %d %s", layer, proj == 1 ? "o_proj" : "down_proj");
                if (c.expect(in, out, 0.02f)) {
                    c.check(delta.data(), 0, out, proj == 1 ? lm.layers[layer].o_proj : lm.layers[layer].down_proj, 0, label);
                }
                fails += c.fails;
            }
        }

        // fused qkv: verify the q slice of one layer
        {
            const int   layer = lm.cfg.n_layers / 4;
            std::string name = "model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight";
            Checker     c(gf, name.c_str());
            std::vector<float> down_f, up_f, delta;
            int64_t r, in, out;
            if (!get_factors(*ad, 0, layer, 0, down_f, up_f, &r, &in, &out)) {
                fprintf(stderr, "[Test-Adapter] FAIL: qkv factors\n");
                fails++;
            } else {
                const int64_t q_out = ggml_get_tensor(gf.meta, "model.layers.0.self_attn.q_proj.weight")->ne[1];
                delta.resize((size_t) q_out * in);
                ref_delta(up_f.data(), 0, q_out, r, down_f.data(), in, delta.data());
                char label[128];
                snprintf(label, sizeof(label), "AR layer %d fused q slice", layer);
                if (c.expect(in, q_out, 0.02f)) {
                    c.check(delta.data(), 0, q_out, lm.layers[layer].qkv, 0, label);
                }
                fails += c.fails;
            }
        }

        qw3lm_free(&lm);
        gf_close(&gf);
    }

    // ---------- NAR half ----------
    if (ad->has_nar) {
        Yue2NAR     nar;
        GGUFModel   gf;
        if (!nar_load(&nar, argv[1], ad)) {
            fprintf(stderr, "[Test-Adapter] FAIL: NAR load+merge\n");
            return 1;
        }
        if (!gf_load(&gf, argv[1])) {
            fprintf(stderr, "[Test-Adapter] FAIL: GGUF read\n");
            return 1;
        }

        // io projections: base + diff * b
        struct io_target {
            const STEntry * diff;
            const STEntry * b;
            const char *    name;
            struct ggml_tensor * merged;
        };
        for (auto t : { io_target{ ad->vae2llm_diff, ad->vae2llm_b, "vae2llm.weight", nar.vae2llm_w },
                        io_target{ ad->llm2vae_diff, ad->llm2vae_b, "llm2vae.weight", nar.llm2vae_w } }) {
            if (!t.diff) {
                continue;
            }
            Checker   c(gf, t.name);
            int64_t   in = t.diff->shape[1], out = t.diff->shape[0];
            std::vector<float> diff_f((size_t) out * in), b_f(out);
            if (!adapter_f32(ad->st, *t.diff, diff_f.data()) || !adapter_f32(ad->st, *t.b, b_f.data())) {
                fprintf(stderr, "[Test-Adapter] FAIL: io factors for %s\n", t.name);
                fails++;
                continue;
            }
            for (int64_t o = 0; o < out; o++) {
                for (int64_t i = 0; i < in; i++) {
                    diff_f[(size_t) o * in + i] *= b_f[o];
                }
            }
            if (c.expect(in, out, 0.02f)) {
                c.check(diff_f.data(), 0, out, t.merged, 0, t.name);
            }
            fails += c.fails;
        }

        // fused gate_up: verify the up slice of one layer
        {
            const int   layer = 3;
            std::string name = "model.layers." + std::to_string(layer) + ".nar_mlp.up_proj.weight";
            Checker     c(gf, name.c_str());
            std::vector<float> down_f, up_f, delta;
            int64_t r, in, out;
            if (!get_factors(*ad, 1, layer, 2, down_f, up_f, &r, &in, &out)) {
                fprintf(stderr, "[Test-Adapter] FAIL: gate_up factors\n");
                fails++;
            } else {
                const int64_t g_out = ggml_get_tensor(gf.meta, "model.layers.0.nar_mlp.gate_proj.weight")->ne[1];
                delta.resize((size_t) in * out);
                ref_delta(up_f.data(), 0, out, r, down_f.data(), in, delta.data());
                char label[128];
                snprintf(label, sizeof(label), "NAR layer %d fused up slice", layer);
                if (c.expect(in, out - g_out, 0.02f)) {
                    c.check(delta.data(), g_out, out - g_out, nar.layers[layer].gate_up, g_out, label);
                }
                fails += c.fails;
            }
        }

        nar_free(&nar);
        gf_close(&gf);
    }

    yue2_adapter_free(ad);
    if (fails) {
        fprintf(stderr, "[Test-Adapter] FAIL: %d check(s) mismatched\n", fails);
        return 1;
    }
    printf("[Test-Adapter] ALL MERGE CHECKS PASSED\n");
    return 0;
}
