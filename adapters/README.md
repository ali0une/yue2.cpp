# Adapters

Place your LoRA adapters here: a `.safetensors` file, or a folder holding
one next to its `adapter_config.json`. The server lists them at startup with
the halves of the backbone each one changes, AR (the score and the codes),
NAR (the acoustics) or both, and the WebUI stacks any number of them, each
with its own strength.

| Trainer              | Layout                                 | Alpha source                        |
|----------------------|----------------------------------------|-------------------------------------|
| AI Toolkit, ComfyUI  | single `.safetensors`                  | per module `.alpha`, else the rank  |
| Native YuE2 trainers | single `.safetensors`                  | `__metadata__` alpha, else the rank |
| Folder trainers      | `.safetensors` + `adapter_config.json` | `lora_alpha` or `alpha`             |
| Sliders              | single `.safetensors`                  | per module `.alpha`                 |

ComfyUI and AI Toolkit files carry `text_encoders.*` for the AR half and
`diffusion_model.*` for the NAR half, with fused `qkv_proj` and
`gate_up_proj`. Native files name the projections `layers.N.self_attn.*`,
`layers.N.nar_self_attn.*` and their MLPs, and may replace or adapt
`vae2llm` and `llm2vae`. A file holding anything else (LoKr, DoRA, an extra
conditioning branch) is skipped with the name of the tensor at fault.

Point the server at this folder:

```bash
./build/yue-server --model models/YuE2-3B-Q8_0.gguf --vae models/YuE2-Vae-F32.gguf --adapters ./adapters
```

## In a request

A request stacks any number of the loaded adapters, each at its own
strength:

```bash
curl -s localhost:8087/synth -d '{
    "style":  "militant reggae, roots, dub bassline",
    "lyrics": "[Chorus]\nrise up and stand\nstand your ground",
    "duration": 30,
    "adapters": [
        { "name": "mltnt_roots" },
        { "name": "ar_instrumental_cot_full", "scale": 0.8 }
    ]
}'
```

`scale` (default 1.0) applies to every half the adapter changes. `ar_scale`
and `nar_scale` override it on one half: an adapter trained on both halves
can be pushed harder on the decoder than the planner, or damped on the
planner to hand song structure back to the base model. Empty (or 0) keeps
`scale`.

```json
{ "name": "mltnt_soundclash", "ar_scale": 0.5 }
{ "name": "mltnt_fusion", "nar_scale": 1.5 }
```

The same backbone under a different adapter list or strength is a different
store entry, so switching reloads the half (and re-merges) under the default
STRICT policy; with `--keep-loaded` each variant accumulates in VRAM.

The CLI runs the same request file through `yue-synth`:

```bash
cat > /tmp/req.json <<'EOF'
{
    "style":   "militant reggae, roots, dub bassline",
    "lyrics":  "[Chorus]\nrise up and stand\nstand your ground",
    "duration": 30,
    "adapters": [ { "name": "mltnt_frontline" } ]
}
EOF

./build/yue-synth \
    --model models/YuE2-3B-BF16.gguf \
    --vae models/YuE2-Vae-F32.gguf \
    --adapters ./adapters \
    --request /tmp/req.json \
    --out song.mp3
```

Unknown names are rejected (400 on the server, a fatal on the CLI).

## LoRA families in this repo

`dl.sh` downloads them into this directory.

### MLTNT — militant roots reggae (both halves)

`becausereasons/yue2-mltnt-militant-reggae` (CC-BY-NC-4.0), six
artist-style LoRAs, trigger `mltnt`: `mltnt_roots`, `mltnt_frontline`,
`mltnt_fusion`, `mltnt_steppers`, `mltnt_soundclash`, `mltnt_chanter`.
The last two are a new-trainer generation (Ostris AI Toolkit,
`lora_A`/`lora_B` naming, no io projections) that over-commits at full
planner strength on fast or hard styles — run them with
`{ "name": "mltnt_soundclash", "ar_scale": 0.5 }` and the same prompt keeps
its structure instead of running away; slow songs take full strength. The
Fusion recipe pushes the decoder:
`{ "name": "mltnt_fusion", "nar_scale": 1.5 }` with `"cfg_scale": 1.4`.
See [README-mltnt-militant-reggae.md](README-mltnt-militant-reggae.md) for
per-LoRA character, demos and prompting.

### Instrumental CoT (AR half only)

`Mothersuperior/YuE2-instrumental-cot-full-loras` (CC-BY-NC-4.0),
downloaded as `ar_instrumental_cot_full.safetensors`. Steers the planner
toward instrumental music with a section plan: run it with
`"cot": "full"` (the default) and put the section tags in `lyrics`
(`[intro]`, `[verse 0:15-0:45]`, ...). The NAR half stays base; for
production sound the author pairs it with their v4 NAR LoRA, which stacks
as its own entry in the same `adapters` list. See
[README-instrumental-cot-full.md](README-instrumental-cot-full.md) for
caption styles, recipes and training notes.

### QWWL / DRKSF — qawwali and dark Sufi (both halves)

`becausereasons/yue2-qwwl-qawwali-sufi-tabla` (CC-BY-NC-4.0), two LoRAs,
one sound each: `sufi_qwwl_mehfil.safetensors` (trigger `qwwl`, live
traditional qawwali) and `sufi_drksf_midnight.safetensors` (trigger
`drksf`, dark Sufi fusion). Run them with `"cot": "off"`: they were
trained without scores, and with the score on the voice follows a pop tune
shape instead of the qawwali form. Lyrics are syllables (alaap, sargam,
tarana) or romanized couplets in qawwali form, not verse-chorus. See
[README-qwwl-qawwali-sufi-tabla.md](README-qwwl-qawwali-sufi-tabla.md) for
character, prompting and lyrics recipes.
