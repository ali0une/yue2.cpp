# Adapters

Drop runtime LoRA `.safetensors` files in this folder and point the server or
the CLI at it:

```bash
./build/yue-server --model models/YuE2-3B-BF16.gguf --vae models/YuE2-Vae-F32.gguf --adapters ./adapters
```

Each file is parsed and validated at startup; its name (without extension)
becomes an adapter selectable per request via the `adapter` field. `GET
/props` lists the loaded names under `"adapters"`. Unknown names are rejected
(400 on the server, a fatal on the CLI).

## Server

Submit a job, then poll for the result:

```bash
# 1. queue a generation with an adapter
curl -s localhost:8087/synth -d '{
    "style":  "militant reggae, roots, dub bassline",
    "lyrics": "[Chorus]\nrise up and stand\nstand your ground",
    "duration": 30,
    "adapter": "mltnt_roots",
    "adapter_clip_strength": 1.0,
    "adapter_model_strength": 1.0
}'
# -> {"id":"7533dd64cae69f20"}

# 2. poll status, then fetch the audio (multipart: replay request + track)
curl -s "localhost:8087/job?id=7533dd64cae69f20"
curl -s "localhost:8087/job?id=7533dd64cae69f20&result=1" -o track
```

## CLI

`yue-synth` runs the same pipeline from a request file; the `adapter` field
picks the LoRA:

```bash
cat > /tmp/req.json <<'EOF'
{
    "style":   "militant reggae, roots, dub bassline",
    "lyrics":  "[Chorus]\nrise up and stand\nstand your ground",
    "duration": 30,
    "adapter":  "mltnt_frontline",
    "adapter_clip_strength":  1.0,
    "adapter_model_strength": 1.0
}
EOF

./build/yue-synth \
    --model models/YuE2-3B-BF16.gguf \
    --vae models/YuE2-Vae-F32.gguf \
    --adapters ./adapters \
    --request /tmp/req.json \
    --out song.mp3
```

An empty or absent `adapter` runs the base weights. The same backbone under a
different adapter is a different store entry, so switching adapters reloads
the half (and re-merges) under the default STRICT policy; with `--keep-loaded`
each variant accumulates in VRAM.

## Strengths

The LoRA is merged at a strength per half of the backbone, both defaulting to
1.0 (the trained delta as is):

| Field | Half | Range |
|---|---|---|
| `adapter_clip_strength` | planner (AR), `text_encoders` | [0.5, 1.0] |
| `adapter_model_strength` | decoder (NAR), `diffusion_model` | [1.0, 1.5] |

Below 1.0 the clip strength hands song structure and tempo back to the base
model (the fast/hard-style recipe):

```json
{ "adapter": "mltnt_soundclash", "adapter_clip_strength": 0.5, "adapter_model_strength": 1.0 }
```

above 1.0 the model strength pushes the voice timbre further (the Fusion
recipe):

```json
{ "adapter": "mltnt_fusion", "adapter_clip_strength": 1.0, "adapter_model_strength": 1.5, "cfg_scale": 1.4 }
```

Out of range is rejected (400 on
the server, a fatal on the CLI). The CLI also takes `--clip-strength` and
`--model-strength` as overrides of the request fields. The strength is part
of the store key: the same adapter under two strengths is two entries, so a
strength switch reloads like an adapter switch.

## Supported layout: the ComfyUI artist LoRA family

Single-file ComfyUI-style adapters for the YuE2-3B backbone (base
Comfy-Org/YuE2), covering one or both halves (a half is simply skipped if
the file has no keys for it), with optional io projections:

| Key | Target |
|-----|--------|
| `text_encoders.model.layers.N.self_attn.qkv_proj.lora_{down,up}.weight` | AR `self_attn.{q,k,v}_proj` |
| `text_encoders.model.layers.N.self_attn.o_proj.lora_{down,up}.weight` | AR `self_attn.o_proj` |
| `text_encoders.model.layers.N.mlp.gate_up_proj.lora_{down,up}.weight` | AR `mlp.{gate,up}_proj` |
| `text_encoders.model.layers.N.mlp.down_proj.lora_{down,up}.weight` | AR `mlp.down_proj` |
| `diffusion_model.model.layers.N.*` (the same eight keys) | NAR `nar_self_attn.*`, `nar_mlp.*` |
| `diffusion_model.vae2llm.diff(.diff_b)` | `vae2llm.weight` |
| `diffusion_model.llm2vae.diff(.diff_b)` | `llm2vae.weight` |

The factor suffix also accepts the PEFT / ai-toolkit aliases `lora_A` (=
`lora_down`) and `lora_B` (= `lora_up`); some files in the family use that
naming. The io projection keys are optional: files without them load fine,
the merge just skips that step.

The layer delta is `lora_up @ lora_down` (this family carries no alpha),
multiplied by the strength of its half; when present, the io delta is
`diff * diff_b` per row, same scale. The merge happens on CPU at
backbone load time, before the weights upload: the dequantized base plus the
delta is re-encoded to the tensor's native GGUF type, so it works for every
quantization the backbone GGUF ships in.

Example: `becausereasons/yue2-mltnt-militant-reggae` (CC-BY-NC-4.0), six
variants, one file each:

| Adapter name | File |
|---|---|
| `mltnt_roots` | [mltnt_roots.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_roots.safetensors) |
| `mltnt_frontline` | [mltnt_frontline.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_frontline.safetensors) |
| `mltnt_fusion` | [mltnt_fusion.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_fusion.safetensors) |
| `mltnt_steppers` | [mltnt_steppers.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_steppers.safetensors) |
| `mltnt_soundclash` | [mltnt_soundclash.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_soundclash.safetensors) |
| `mltnt_chanter` | [mltnt_chanter.safetensors](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_chanter.safetensors) |

The last two are a "new-trainer" generation
(Ostris AI Toolkit, `lora_A`/`lora_B` naming, no io projections). Their
recommended recipe uses a planner strength of 0.5 for fast or hard styles:
set `"adapter_clip_strength": 0.5` in the request (or `--clip-strength 0.5`
on the CLI) and the same prompt keeps its structure instead of running away.
The four earlier variants were tuned at full strength and are unaffected.

Also in the family: `Mothersuperior/YuE2-instrumental-cot-full-loras`
(CC-BY-NC-4.0), [ar_lora_inst_v3abc_comfyui.safetensors](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc_comfyui.safetensors)
(rename it when you drop it in this folder if you want a shorter adapter
name — the name is just the file name without extension). AR half only —
it steers the planner toward instrumental music with a section plan, so run
it with `"cot": "full"` (the default) and put the section tags in `lyrics`
(`[intro]`, `[verse 0:15-0:45]`, ...). The NAR half stays base; for
production sound the author pairs it with their v4 NAR LoRA, which has to
be loaded as its own adapter file since one request takes one `adapter`.
