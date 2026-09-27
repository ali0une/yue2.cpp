// test-adapter.cpp: runtime LoRA parse + validate harness
//
// Loads each given .safetensors through the real adapter code path and prints
// its structure (layer pairs per half, ranks, io shapes). Exits non-zero if
// any file fails to parse or validate. No backbone GGUF is needed, so this is
// a fast smoke test of the key layout before touching a model.
//
// `--selftest` writes synthetic adapters in both factor naming conventions
// (GGUF-style lora_down/lora_up and PEFT/ai-toolkit lora_A/lora_B) plus a
// bogus-name file, and checks that both conventions parse to identical
// structures while the bogus one is rejected.

#include "adapter-merge.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool summarize(const Yue2Adapter & ad) {
    std::vector<int> ar_l, nar_l;
    int64_t rank_ar = -1, rank_nar = -1;
    for (const auto & kv : ad.pairs) {
        const int      branch = std::get<0>(kv.first);
        const int      layer  = std::get<1>(kv.first);
        const int64_t  r      = kv.second.down->shape[0];
        if (branch == 0) {
            ar_l.push_back(layer);
            rank_ar = r;
        } else {
            nar_l.push_back(layer);
            rank_nar = r;
        }
    }
    auto uniq = [](std::vector<int> & v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    uniq(ar_l);
    uniq(nar_l);
    printf("[Test-Adapter] %s: pairs=%zu ar_layers=%d (rank %lld) nar_layers=%d (rank %lld) io=%s\n",
           ad.name.c_str(), ad.pairs.size(), (int) ar_l.size(), (long long) rank_ar,
           (int) nar_l.size(), (long long) rank_nar, ad.has_io ? "yes" : "no");
    if (ad.vae2llm_diff) {
        printf("[Test-Adapter]   vae2llm.diff [%lld, %lld] b[%lld]\n", (long long) ad.vae2llm_diff->shape[0],
               (long long) ad.vae2llm_diff->shape[1], (long long) ad.vae2llm_b->shape[0]);
    }
    if (ad.llm2vae_diff) {
        printf("[Test-Adapter]   llm2vae.diff [%lld, %lld] b[%lld]\n", (long long) ad.llm2vae_diff->shape[0],
               (long long) ad.llm2vae_diff->shape[1], (long long) ad.llm2vae_b->shape[0]);
    }
    return true;
}

namespace {

// First fill values of the synthetic tensors: down AR starts at v_fill[0],
// up AR at v_fill[6] (fill order: down AR, up AR, down NAR, up NAR, diff, b).
const float v_fill[14] = { 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f,
                           7.0f, 7.5f };

// One F32 tensor of the synthetic file. nd 2 -> shape [s[0], s[1]], nd 1 -> [s[0]].
struct SynthTensor {
    std::string name;
    int         nd;
    int64_t     s[2];
};

// Write a minimal valid safetensors: one AR qkv pair, one NAR down_proj pair,
// and the vae2llm io diff. dsfx / usfx select the factor naming convention.
// Data is filled with 1.0f, 1.5f, 2.0f, ... in entry order.
bool write_synth(const char * path, const std::string & dsfx, const std::string & usfx) {
    const int64_t r = 2, in = 3, out = 4;
    std::vector<SynthTensor> ts = {
        { "text_encoders.model.layers.0.self_attn.qkv_proj.lora_" + dsfx + ".weight", 2, { r, in } },
        { "text_encoders.model.layers.0.self_attn.qkv_proj.lora_" + usfx + ".weight", 2, { out, r } },
        { "diffusion_model.model.layers.0.mlp.down_proj.lora_" + dsfx + ".weight", 2, { r, in } },
        { "diffusion_model.model.layers.0.mlp.down_proj.lora_" + usfx + ".weight", 2, { out, r } },
        { "diffusion_model.vae2llm.diff", 2, { 5, in } },
        { "diffusion_model.vae2llm.diff_b", 1, { 5 } },
    };

    std::string data;
    std::vector<size_t> starts(ts.size()), ends(ts.size());
    size_t off = 0;
    float  v   = 1.0f;
    for (size_t i = 0; i < ts.size(); i++) {
        const int64_t n = ts[i].s[0] * (ts[i].nd == 2 ? ts[i].s[1] : 1);
        starts[i]       = off;
        for (int64_t k = 0; k < n; k++) {
            data.append(reinterpret_cast<const char *>(&v), sizeof(float));
            v += 0.5f;
            off += sizeof(float);
        }
        ends[i] = off;
    }

    std::string hdr = "{";
    for (size_t i = 0; i < ts.size(); i++) {
        hdr += "\"" + ts[i].name + "\": {\"dtype\": \"F32\", \"shape\": [";
        hdr += std::to_string(ts[i].s[0]);
        if (ts[i].nd == 2) {
            hdr += ", " + std::to_string(ts[i].s[1]);
        }
        hdr += "], \"data_offsets\": [" + std::to_string(starts[i]) + ", " + std::to_string(ends[i]) + "]}";
        if (i + 1 < ts.size()) {
            hdr += ", ";
        }
    }
    hdr += "}";

    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    const uint64_t len = (uint64_t) hdr.size();
    bool ok = fwrite(&len, 8, 1, f) == 1 && fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size() &&
              fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

// Verify one parsed synthetic adapter: 2 pairs (AR qkv + NAR down_proj), io,
// rank 2, and the deterministic fill values intact.
bool check_synth(const Yue2Adapter & ad) {
    if (ad.pairs.size() != 2 || !ad.has_ar || !ad.has_nar || !ad.has_io) {
        fprintf(stderr, "[Test-Adapter] synth: wrong structure (pairs=%zu ar=%d nar=%d io=%d)\n", ad.pairs.size(),
                ad.has_ar, ad.has_nar, ad.has_io);
        return false;
    }
    auto it = ad.pairs.find({ 0, 0, 0 });  // AR layer 0 qkv
    if (it == ad.pairs.end() || !it->second.down || !it->second.up) {
        fprintf(stderr, "[Test-Adapter] synth: missing AR qkv pair\n");
        return false;
    }
    const STEntry & down = *it->second.down;
    const STEntry & up   = *it->second.up;
    if (down.shape[0] != 2 || down.shape[1] != 3 || up.shape[0] != 4 || up.shape[1] != 2) {
        fprintf(stderr, "[Test-Adapter] synth: factor shapes wrong\n");
        return false;
    }
    if (std::memcmp(st_data(ad.st, down), &v_fill[0], sizeof(float)) != 0 ||
        std::memcmp(st_data(ad.st, up), &v_fill[6], sizeof(float)) != 0) {
        fprintf(stderr, "[Test-Adapter] synth: factor data mismatch\n");
        return false;
    }
    if (!ad.vae2llm_diff || !ad.vae2llm_b || ad.vae2llm_diff->shape[0] != 5 || ad.vae2llm_diff->shape[1] != 3) {
        fprintf(stderr, "[Test-Adapter] synth: io diff wrong\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) {
        int fails = 0;
        const std::string convs[2][2] = { { "down", "up" }, { "A", "B" } };
        for (auto & c : convs) {
            char path[128];
            snprintf(path, sizeof(path), "test-adapter-synth-%s.safetensors", c[0].c_str());
            if (!write_synth(path, c[0], c[1])) {
                fprintf(stderr, "[Test-Adapter] selftest: cannot write %s\n", path);
                fails++;
                continue;
            }
            auto * ad = new Yue2Adapter();
            if (!yue2_adapter_load(ad, path) || !check_synth(*ad)) {
                fprintf(stderr, "[Test-Adapter] selftest FAIL: %s\n", path);
                fails++;
            } else {
                printf("[Test-Adapter] selftest OK: %s (lora_%s / lora_%s)\n", path, c[0].c_str(), c[1].c_str());
            }
            yue2_adapter_free(ad);
            remove(path);
        }
        // a bogus factor name must be rejected
        {
            const char * path = "test-adapter-synth-bogus.safetensors";
            if (write_synth(path, "C", "D")) {
                auto * ad = new Yue2Adapter();
                if (yue2_adapter_load(ad, path)) {
                    fprintf(stderr, "[Test-Adapter] selftest FAIL: bogus lora_C accepted\n");
                    fails++;
                } else {
                    printf("[Test-Adapter] selftest OK: bogus factor name rejected\n");
                }
                yue2_adapter_free(ad);
            }
            remove(path);
        }
        if (fails) {
            fprintf(stderr, "[Test-Adapter] selftest FAIL: %d check(s)\n", fails);
            return 1;
        }
        printf("[Test-Adapter] SELFTEST PASSED\n");
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s adapter.safetensors [adapter.safetensors ...] | --selftest\n", argv[0]);
        return 1;
    }
    int fails = 0;
    for (int i = 1; i < argc; i++) {
        auto * ad = new Yue2Adapter();
        if (!yue2_adapter_load(ad, argv[i])) {
            fprintf(stderr, "[Test-Adapter] FAIL: %s\n", argv[i]);
            yue2_adapter_free(ad);
            fails++;
            continue;
        }
        summarize(*ad);
        yue2_adapter_free(ad);
    }
    return fails ? 1 : 0;
}
