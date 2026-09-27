#pragma once
// adapter-merge.h: runtime LoRA merge into the backbone GGUF weights
//
// Supports the ComfyUI-style single-file "artist LoRA" family for YuE2-3B
// (base Comfy-Org/YuE2), one file covering both halves of the backbone:
//
//   text_encoders.model.layers.N.self_attn.qkv_proj.lora_{down,up}.weight
//   text_encoders.model.layers.N.self_attn.o_proj.lora_{down,up}.weight
//   text_encoders.model.layers.N.mlp.gate_up_proj.lora_{down,up}.weight
//   text_encoders.model.layers.N.mlp.down_proj.lora_{down,up}.weight
//
// The factor suffixes also accept the PEFT / ai-toolkit aliases
// lora_A = lora_down ([r, in]) and lora_B = lora_up ([out, r]).
//       -> AR half: model.layers.N.self_attn.{q,k,v,o}_proj.weight,
//                   model.layers.N.mlp.{gate,up,down}_proj.weight
//   diffusion_model.model.layers.N.*  (the same eight keys)
//       -> NAR half: model.layers.N.nar_self_attn.*.weight,
//                    model.layers.N.nar_mlp.*.weight
//   diffusion_model.vae2llm.diff(.diff_b), diffusion_model.llm2vae.diff(.diff_b)
//       -> vae2llm.weight / llm2vae.weight, delta = diff * diff_b per row
//
// The layer deltas are delta = lora_up @ lora_down (this family carries no
// alpha), multiplied by the per-branch strength passed to
// yue2_adapter_merge: clip scales the AR half, model scales the NAR half.
// The merge runs on CPU after every tensor of a half is
// registered in the WeightCtx and before wctx_alloc uploads: each patched
// PendingCopy is swapped for a staging buffer holding the dequantized base
// plus the delta, re-encoded to the native GGUF type, so the upload path is
// untouched and the result lands in every quantization the GGUF ships.
// Fused QKV / gate_up targets are patched through their raw projections,
// split on the GGUF shapes, before the load-time fusion reads them back.

#include "gguf-weights.h"
#include "safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct Yue2Adapter {
    std::string name;  // file base without the .safetensors extension

    STFile st;

    struct Pair {
        const STEntry * down = nullptr;  // [r, in]
        const STEntry * up   = nullptr;  // [out, r]
    };
    // (branch, layer, proj): branch 0 = AR (text_encoders), 1 = NAR (diffusion_model);
    // proj 0 = qkv_proj, 1 = o_proj, 2 = gate_up_proj, 3 = down_proj
    std::map<std::tuple<int, int, int>, Pair> pairs;

    const STEntry * vae2llm_diff = nullptr;  // [out, in]
    const STEntry * vae2llm_b    = nullptr;  // [out]
    const STEntry * llm2vae_diff = nullptr;  // [out, in]
    const STEntry * llm2vae_b    = nullptr;  // [out]

    bool has_ar  = false;
    bool has_nar = false;
    bool has_io  = false;
};

// Per-branch strength bounds. Above 1.0 the clip side collapses the vocal
// (the planner writes rests instead of sung bars); below 1.0 the model side
// is untested by the adapter recipes.
#define YUE2_CLIP_STRENGTH_MIN   0.5f
#define YUE2_CLIP_STRENGTH_MAX   1.0f
#define YUE2_MODEL_STRENGTH_MIN  1.0f
#define YUE2_MODEL_STRENGTH_MAX  1.5f

static inline bool yue2_adapter_strengths_valid(float clip, float model) {
    return clip >= YUE2_CLIP_STRENGTH_MIN && clip <= YUE2_CLIP_STRENGTH_MAX &&
           model >= YUE2_MODEL_STRENGTH_MIN && model <= YUE2_MODEL_STRENGTH_MAX;
}

static void yue2_adapter_free(Yue2Adapter * ad) {
    if (!ad) {
        return;
    }
    st_close(&ad->st);
    delete ad;
}

namespace {

int adapter_proj_index(const char * proj) {
    if (!strcmp(proj, "qkv_proj")) {
        return 0;
    }
    if (!strcmp(proj, "o_proj")) {
        return 1;
    }
    if (!strcmp(proj, "gate_up_proj")) {
        return 2;
    }
    if (!strcmp(proj, "down_proj")) {
        return 3;
    }
    return -1;
}

bool adapter_entry_ok(const STEntry & e) {
    return e.dtype == "F32" || e.dtype == "BF16" || e.dtype == "F16";
}

void adapter_fail(const char * name, const std::string & key, const char * why) {
    fprintf(stderr, "[Adapter] FATAL: %s: %s: %s\n", name ? name : "?", key.c_str(), why);
}

}  // namespace

// Parse and validate one adapter file. Every key must fit the family layout
// above; each (branch, layer, proj) needs both factors with a consistent
// rank, and the io diffs their per-row scales.
static bool yue2_adapter_load(Yue2Adapter * ad, const char * path) {
    *ad = {};
    if (!st_open(&ad->st, path)) {
        return false;
    }

    std::string p    = path;
    size_t      dot  = p.rfind('.');
    std::string base = dot == std::string::npos ? p : p.substr(0, dot);
    size_t      slash = base.find_last_of("/\\");
    ad->name          = slash == std::string::npos ? base : base.substr(slash + 1);

    for (auto & e : ad->st.entries) {
        if (!adapter_entry_ok(e)) {
            adapter_fail(ad->name.c_str(), e.name, ("unsupported dtype " + e.dtype).c_str());
            return false;
        }

        std::vector<std::string> parts;
        size_t                   start = 0;
        for (size_t i = 0; i <= e.name.size(); i++) {
            if (i == e.name.size() || e.name[i] == '.') {
                parts.push_back(e.name.substr(start, i - start));
                start = i + 1;
            }
        }

        if (parts.size() == 8) {
            // <branch>.model.layers.<N>.<module>.<proj>.lora_<which>.weight
            int branch = parts[0] == "text_encoders" ? 0 : parts[0] == "diffusion_model" ? 1 : -1;
            if (branch < 0 || parts[1] != "model" || parts[2] != "layers" || parts[7] != "weight") {
                adapter_fail(ad->name.c_str(), e.name, "unrecognized key layout");
                return false;
            }
            char * end   = nullptr;
            long   layer = strtol(parts[3].c_str(), &end, 10);
            if (end == parts[3].c_str() || *end != 0 || layer < 0) {
                adapter_fail(ad->name.c_str(), e.name, "bad layer index");
                return false;
            }
            int proj = adapter_proj_index(parts[5].c_str());
            bool ok_module = (proj == 0 || proj == 1) ? parts[4] == "self_attn" : parts[4] == "mlp";
            if (proj < 0 || !ok_module) {
                adapter_fail(ad->name.c_str(), e.name, "unrecognized projection");
                return false;
            }
            Yue2Adapter::Pair & pr = ad->pairs[{ branch, (int) layer, proj }];
            // PEFT / ai-toolkit exports use lora_A / lora_B for the same
            // factors: A is the down projection [r, in], B the up [out, r].
            if (parts[6] == "lora_down" || parts[6] == "lora_A") {
                if (pr.down || e.n_dims != 2) {
                    adapter_fail(ad->name.c_str(), e.name, "duplicate or malformed lora_down");
                    return false;
                }
                pr.down = &e;
            } else if (parts[6] == "lora_up" || parts[6] == "lora_B") {
                if (pr.up || e.n_dims != 2) {
                    adapter_fail(ad->name.c_str(), e.name, "duplicate or malformed lora_up");
                    return false;
                }
                pr.up = &e;
            } else {
                adapter_fail(ad->name.c_str(), e.name, "expected lora_down (lora_A) or lora_up (lora_B)");
                return false;
            }
            if (branch == 0) {
                ad->has_ar = true;
            } else {
                ad->has_nar = true;
            }
        } else if (parts.size() == 3 && parts[0] == "diffusion_model") {
            const STEntry ** diff = nullptr;
            const STEntry ** b    = nullptr;
            if (parts[1] == "vae2llm") {
                diff = &ad->vae2llm_diff;
                b    = &ad->vae2llm_b;
            } else if (parts[1] == "llm2vae") {
                diff = &ad->llm2vae_diff;
                b    = &ad->llm2vae_b;
            } else {
                adapter_fail(ad->name.c_str(), e.name, "unrecognized key");
                return false;
            }
            if (parts[2] == "diff") {
                if (*diff || e.n_dims != 2) {
                    adapter_fail(ad->name.c_str(), e.name, "duplicate or malformed diff");
                    return false;
                }
                *diff = &e;
            } else if (parts[2] == "diff_b") {
                if (*b || e.n_dims != 1) {
                    adapter_fail(ad->name.c_str(), e.name, "duplicate or malformed diff_b");
                    return false;
                }
                *b = &e;
            } else {
                adapter_fail(ad->name.c_str(), e.name, "expected diff or diff_b");
                return false;
            }
            ad->has_io = true;
        } else {
            adapter_fail(ad->name.c_str(), e.name, "unrecognized key");
            return false;
        }
    }

    for (auto & kv : ad->pairs) {
        Yue2Adapter::Pair & pr = kv.second;
        if (!pr.down || !pr.up) {
            adapter_fail(ad->name.c_str(), "", "layer projection has only one factor");
            return false;
        }
        // down [r, in], up [out, r]: the rank is shared
        if (pr.down->shape[0] != pr.up->shape[1]) {
            adapter_fail(ad->name.c_str(), "", "lora_down / lora_up rank mismatch");
            return false;
        }
    }
    for (auto d : { &ad->vae2llm_diff, &ad->llm2vae_diff }) {
        if (*d) {
            const STEntry * b = d == &ad->vae2llm_diff ? ad->vae2llm_b : ad->llm2vae_b;
            if (!b || b->shape[0] != (*d)->shape[0]) {
                adapter_fail(ad->name.c_str(), "", "io diff missing its per-row scale");
                return false;
            }
        }
    }

    fprintf(stderr, "[Adapter] %s: %zu layer pairs (AR %d, NAR %d), io %s\n", ad->name.c_str(),
            ad->pairs.size(), ad->has_ar, ad->has_nar, ad->has_io ? "yes" : "no");
    return true;
}

// Entry data to F32 row-major, [shape[0], shape[1]] for 2D, [shape[0]] for 1D
static bool adapter_f32(const STFile & st, const STEntry & e, float * dst) {
    const void * raw = st_data(st, e);
    int64_t      n   = 1;
    for (int i = 0; i < e.n_dims; i++) {
        n *= e.shape[i];
    }
    if (e.dtype == "F32") {
        memcpy(dst, raw, (size_t) n * sizeof(float));
    } else if (e.dtype == "BF16") {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw, dst, n);
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, dst, n);
    }
    return true;
}

// delta[out][in] = up[out][r] @ down[r][in], computed on the CPU backend.
// delta[o][i] = sum_k up[o][k] * down[k][i], up [out, r], down [r, in].
// A plain exact F32 matmul: the factors are small rank and this runs once at
// load, so no backend graph is worth the layout games. The result comes out
// [out rows, in cols] with the inner dim contiguous, the GGUF weight layout.
static bool adapter_delta(const float * up, int64_t out, int64_t r, const float * down, int64_t in, float * delta) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t o = 0; o < out; o++) {
        float * dst = delta + o * in;
        const float * u = up + o * r;
        for (int64_t i = 0; i < in; i++) {
            float s = 0.0f;
            for (int64_t k = 0; k < r; k++) {
                s += u[k] * down[k * in + i];
            }
            dst[i] = s;
        }
    }
    return true;
}

// Add one delta matrix, [out rows, in cols] F32 inner contiguous, to the
// GGUF tensor `name` of the freshly built WeightCtx. The PendingCopy whose
// src points at the tensor's mmap region is swapped for a staging buffer:
// dequantized base row + delta row, re-encoded to the native GGUF type.
static bool adapter_patch(WeightCtx * wctx, const GGUFModel & gf, const char * name,
                          const float * delta, int64_t in, int64_t out) {
    int64_t idx = gguf_find_tensor(gf.gguf, name);
    if (idx < 0) {
        fprintf(stderr, "[Adapter] FATAL: %s not in the GGUF\n", name);
        return false;
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name);
    if (!src || src->ne[0] != in || src->ne[1] != out) {
        fprintf(stderr, "[Adapter] FATAL: %s shape mismatch (gguf %lldx%lld, delta %lldx%lld)\n", name,
                src ? (long long) src->ne[1] : -1, src ? (long long) src->ne[0] : -1, (long long) out,
                (long long) in);
        return false;
    }

    const void * expected = gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, idx);
    int          pi       = -1;
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        if (wctx->pending[i].src == expected) {
            pi = (int) i;
            break;
        }
    }
    if (pi < 0) {
        fprintf(stderr, "[Adapter] FATAL: %s has no pending copy at its mmap offset\n", name);
        return false;
    }

    WeightCtx::PendingCopy & pc = wctx->pending[pi];
    auto          merged = std::make_unique<uint8_t[]>(pc.nbytes);
    const struct ggml_type_traits * tr = ggml_get_type_traits(src->type);
    std::vector<float>              row((size_t) in);
    size_t                          row_size = ggml_row_size(src->type, in);

    for (int64_t o = 0; o < out; o++) {
        const uint8_t * base_row = (const uint8_t *) pc.src + (size_t) o * row_size;
        tr->to_float(base_row, row.data(), in);
        for (int64_t i = 0; i < in; i++) {
            row[i] += delta[(size_t) o * in + i];
        }
        uint8_t * dst_row = merged.get() + (size_t) o * row_size;
        switch (src->type) {
            case GGML_TYPE_F32:
                memcpy(dst_row, row.data(), (size_t) in * sizeof(float));
                break;
            case GGML_TYPE_F16:
                ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) dst_row, in);
                break;
            case GGML_TYPE_BF16:
                ggml_fp32_to_bf16_row(row.data(), (ggml_bf16_t *) dst_row, in);
                break;
            default:
                ggml_quantize_chunk(src->type, row.data(), dst_row, 0, 1, in, nullptr);
                break;
        }
    }

    wctx->staging_raw.push_back(std::move(merged));
    pc.src = wctx->staging_raw.back().get();
    return true;
}

// Materialize one (branch, layer, proj) delta and patch its target tensors.
// branch 0 patches self_attn/mlp names, branch 1 the nar_ set. qkv and
// gate_up are split across their raw projections on the GGUF shapes.
static bool adapter_patch_proj(WeightCtx * wctx, const GGUFModel & gf,
                               const Yue2Adapter & ad, int branch, int layer, int proj, float scale) {
    // Resolve the delta first: up [out, r] @ down [r, in]
    auto it = ad.pairs.find({ branch, layer, proj });
    if (it == ad.pairs.end()) {
        fprintf(stderr, "[Adapter] FATAL: %s has no %s layer %d proj %d\n", ad.name.c_str(),
                branch == 0 ? "AR" : "NAR", layer, proj);
        return false;
    }
    const auto & pr = it->second;
    const STEntry & down = *pr.down;
    const STEntry & up   = *pr.up;
    const int64_t r  = down.shape[0];
    const int64_t in = down.shape[1];
    const int64_t out = up.shape[0];

    std::vector<float> down_f((size_t) r * in), up_f((size_t) out * r), delta((size_t) out * in);
    if (!adapter_f32(ad.st, down, down_f.data()) || !adapter_f32(ad.st, up, up_f.data())) {
        return false;
    }
    if (!adapter_delta(up_f.data(), out, r, down_f.data(), in, delta.data())) {
        return false;
    }
    if (scale != 1.0f) {
        for (size_t i = 0; i < delta.size(); i++) {
            delta[i] *= scale;
        }
    }

    // Target GGUF names per projection
    char qn[96], kn[96], vn[96], on[96], gn[96], un[96], dn[96];
    if (branch == 0) {
        snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.q_proj.weight", layer);
        snprintf(kn, sizeof(kn), "model.layers.%d.self_attn.k_proj.weight", layer);
        snprintf(vn, sizeof(vn), "model.layers.%d.self_attn.v_proj.weight", layer);
        snprintf(on, sizeof(on), "model.layers.%d.self_attn.o_proj.weight", layer);
        snprintf(gn, sizeof(gn), "model.layers.%d.mlp.gate_proj.weight", layer);
        snprintf(un, sizeof(un), "model.layers.%d.mlp.up_proj.weight", layer);
        snprintf(dn, sizeof(dn), "model.layers.%d.mlp.down_proj.weight", layer);
    } else {
        snprintf(qn, sizeof(qn), "model.layers.%d.nar_self_attn.q_proj.weight", layer);
        snprintf(kn, sizeof(kn), "model.layers.%d.nar_self_attn.k_proj.weight", layer);
        snprintf(vn, sizeof(vn), "model.layers.%d.nar_self_attn.v_proj.weight", layer);
        snprintf(on, sizeof(on), "model.layers.%d.nar_self_attn.o_proj.weight", layer);
        snprintf(gn, sizeof(gn), "model.layers.%d.nar_mlp.gate_proj.weight", layer);
        snprintf(un, sizeof(un), "model.layers.%d.nar_mlp.up_proj.weight", layer);
        snprintf(dn, sizeof(dn), "model.layers.%d.nar_mlp.down_proj.weight", layer);
    }

    if (proj == 0) {
        // qkv: split the [out, in] delta into q / k / v row blocks
        const int64_t q_out = ggml_get_tensor(gf.meta, qn)->ne[1];
        const int64_t k_out = ggml_get_tensor(gf.meta, kn)->ne[1];
        const int64_t v_out = ggml_get_tensor(gf.meta, vn)->ne[1];
        if (q_out + k_out + v_out != out) {
            fprintf(stderr, "[Adapter] FATAL: layer %d qkv delta width %lld != %lld+%lld+%lld\n", layer,
                    (long long) out, (long long) q_out, (long long) k_out, (long long) v_out);
            return false;
        }
        if (!adapter_patch(wctx, gf, qn, delta.data(), in, q_out)) {
            return false;
        }
        if (!adapter_patch(wctx, gf, kn, delta.data() + (size_t) q_out * in, in, k_out)) {
            return false;
        }
        if (!adapter_patch(wctx, gf, vn, delta.data() + (size_t) (q_out + k_out) * in, in, v_out)) {
            return false;
        }
    } else if (proj == 1) {
        if (!adapter_patch(wctx, gf, on, delta.data(), in, out)) {
            return false;
        }
    } else if (proj == 2) {
        // gate_up: split into gate / up row blocks
        const int64_t g_out = ggml_get_tensor(gf.meta, gn)->ne[1];
        const int64_t u_out = ggml_get_tensor(gf.meta, un)->ne[1];
        if (g_out + u_out != out) {
            fprintf(stderr, "[Adapter] FATAL: layer %d gate_up delta width %lld != %lld+%lld\n", layer,
                    (long long) out, (long long) g_out, (long long) u_out);
            return false;
        }
        if (!adapter_patch(wctx, gf, gn, delta.data(), in, g_out)) {
            return false;
        }
        if (!adapter_patch(wctx, gf, un, delta.data() + (size_t) g_out * in, in, u_out)) {
            return false;
        }
    } else {
        if (!adapter_patch(wctx, gf, dn, delta.data(), in, out)) {
            return false;
        }
    }
    return true;
}

// Patch the io projections: delta = diff * diff_b per row, scale 1.0
static bool adapter_patch_io(WeightCtx * wctx, const GGUFModel & gf, const Yue2Adapter & ad, float scale) {
    struct io_target {
        const STEntry * diff;
        const STEntry * b;
        const char *    name;
    };
    for (auto t : { io_target{ ad.vae2llm_diff, ad.vae2llm_b, "vae2llm.weight" },
                    io_target{ ad.llm2vae_diff, ad.llm2vae_b, "llm2vae.weight" } }) {
        if (!t.diff) {
            continue;
        }
        const int64_t out = t.diff->shape[0];
        const int64_t in  = t.diff->shape[1];
        std::vector<float> diff_f((size_t) out * in), b_f((size_t) out);
        if (!adapter_f32(ad.st, *t.diff, diff_f.data()) || !adapter_f32(ad.st, *t.b, b_f.data())) {
            return false;
        }
        for (int64_t o = 0; o < out; o++) {
            const float s = scale * b_f[o];
            for (int64_t i = 0; i < in; i++) {
                diff_f[(size_t) o * in + i] *= s;
            }
        }
        if (!adapter_patch(wctx, gf, t.name, diff_f.data(), in, out)) {
            return false;
        }
    }
    return true;
}

// Apply the adapter to one freshly built half: every (layer, proj) pair of
// its branch must be present, plus the io projections for the NAR. scale is
// the strength of this branch, 1.0 merges the trained delta as is. Call
// after all loads, before wctx_alloc.
static bool yue2_adapter_merge(WeightCtx * wctx, const GGUFModel & gf,
                               const Yue2Adapter * ad, int n_layers, bool ar, float scale = 1.0f) {
    if (!ad) {
        return true;
    }
    const bool has = ar ? ad->has_ar : ad->has_nar;
    if (!has) {
        fprintf(stderr, "[Adapter] %s carries no %s half, skipping\n", ad->name.c_str(), ar ? "AR" : "NAR");
        return true;
    }

    int branch = ar ? 0 : 1;
    for (int layer = 0; layer < n_layers; layer++) {
        for (int proj = 0; proj < 4; proj++) {
            if (!adapter_patch_proj(wctx, gf, *ad, branch, layer, proj, scale)) {
                return false;
            }
        }
    }
    if (!ar && ad->has_io) {
        if (!adapter_patch_io(wctx, gf, *ad, scale)) {
            return false;
        }
    }

    fprintf(stderr, "[Adapter] %s merged into the %s half (%d layers, strength %.2f)\n", ad->name.c_str(),
            ar ? "AR" : "NAR", n_layers, (double) scale);
    return true;
}
