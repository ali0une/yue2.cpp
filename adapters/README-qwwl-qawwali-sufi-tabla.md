---
license: cc-by-nc-4.0
base_model:
  - m-a-p/YuE2-3B
  - Comfy-Org/YuE2
pipeline_tag: text-to-audio
language:
  - ur
  - pa
tags:
  - lora
  - yue2
  - music-generation
  - text-to-music
  - song-generation
  - qawwali
  - sufi
  - south-asian
  - world-fusion
  - dark-ambient
  - comfyui
  - fs_audio
---

# QWWL / DRKSF — Qawwali & Dark Sufi Fusion LoRAs for YuE2

> Trained on my own GPU and free to use. If it's useful, you can chip in for compute on [Ko-fi](https://ko-fi.com/becausereasons) <3

Two artist-style LoRAs for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), one sound each:

| | LoRA | File | Trigger | Character | Start here |
|---|---|---|---|---|---|
| 🕌 | **QWWL Mehfil** | `qwwl_mehfil.safetensors` | `qwwl` | Traditional Sufi qawwali as a live party performs it: a lead voice declaiming couplets, a male chorus throwing the refrain back, insistent repetition, staccato sargam and tarana bursts, harmonium, tabla, dholak, handclaps. | clip 1.0 / model 1.0, **score off** |
| 🌑 | **DRKSF Midnight** | `drksf_midnight.safetensors` | `drksf` | The same kind of voice set in dark studio production: long aching held notes and sargam riffs over ambient guitar and synth textures, deep bass, slow heavy beats. Haunting, nocturnal, anthemic. | clip 1.0 / model 1.0, **score off** |

Each file patches **both halves** of YuE2: the autoregressive planner (arrangement and vocal lines) and the flow-matching decoder (the sound).

The first attempt at this release was a single LoRA trained on everything at once, and it came back as glossy film-pop. Splitting it into focused sets, one sound per LoRA, and rendering with the score switched off, is what fixed it. The notes on that are under [Training](#training).

## Listen

All demos: 32 steps `dpm_2` / `sgm_uniform`, **`score_mode off`**, cfg 1.0, music temperature 1.0, no post-processing, 170 s cap, seed 7 unless marked.

### Syllable singing

The lyric for these is nothing but syllables: long `Aaa` alaap lines, sargam runs (`Sa re ga ma pa dha ni`) and tarana (`Tana dere na`, `Ta re da ni`). This is the improvising voice of the tradition, and it is where these LoRAs sound most like themselves. See [Lyrics](#lyrics).

**DRKSF Midnight — sargam over dark beats**, prompt `prompts/sargam_dark_beats.txt`:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/drksf_midnight__sargam_dark_beats.mp3"></audio>

**DRKSF Midnight — ethno downtempo**, prompt `prompts/ethno_downtempo.txt`:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/drksf_midnight__ethno_downtempo.mp3"></audio>

Same prompt, seed 11:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/drksf_midnight__ethno_downtempo_seed11.mp3"></audio>

**QWWL Mehfil — live syllable singing**, prompt `prompts/live_syllables_qawwali.txt`: the acoustic party, lead and chorus trading syllable runs:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/qwwl_mehfil__live_syllables.mp3"></audio>

### With words

These four share one original Urdu lyric written in qawwali form (couplets, a repeated refrain, a sargam break).

**QWWL Mehfil — dark traditional qawwali**, prompt `prompts/dark_traditional_qawwali.txt`:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/qwwl_mehfil__dark_traditional.mp3"></audio>

Same prompt, seed 11:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/qwwl_mehfil__dark_traditional_seed11.mp3"></audio>

**DRKSF Midnight — dark fusion**, prompt `prompts/dark_fusion.txt`:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/drksf_midnight__dark_fusion.mp3"></audio>

Same prompt, seed 11:

<audio controls src="https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/demos/drksf_midnight__dark_fusion_seed11.mp3"></audio>

## Quick start (ComfyUI)

The LoRAs are in the fused-key layout that Comfy's YuE2 implementation uses (`text_encoders.*` for the planner, `diffusion_model.*` for the decoder). They were trained with, and load through, the **[FS_Audio Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite)** node pack.

1. ComfyUI ≥ v0.36.0 (native YuE2 support) and the FS_Audio Suite custom node pack.
2. Base model: `yue2_3b_bf16.safetensors` from [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) in `models/checkpoints/`.
3. Drop the `.safetensors` files into `models/loras/`.
4. Chain the nodes:

```
🧩 FS_Audio Lora Loader  ──loras──▶  🎤 FS_Audio Model Loader  ──pipe──▶  🎵 FS_Audio Sampler  ──▶  💿 FS_Audio Output
   lora_name      = qwwl_mehfil.safetensors          yue2_checkpoint = yue2_3b_bf16       style      = <prompt, starts with "qwwl,">      
   strength_clip  = 1.0   (planner)                  melody_transcriber = none            lyrics     = <tagged blocks, romanized>
   strength_model = 1.0   (decoder)                                                       score_mode = off
```

Sampler settings used for every demo:

| widget | value |
|---|---|
| steps / sampler / scheduler | 32 / `dpm_2` / `sgm_uniform` |
| **score_mode** | **`off`** |
| song_length_cap | 170 |
| Weirdness (cfg) / score_temperature / music_temperature | 1.0 / 0.7 / 1.0 |
| repetition_penalty | 1.2 |

### Why score off

With `score_mode full` YuE2 first writes itself a melody score in ABC notation and then sings it. The base model has never seen qawwali, so the score it writes is a tidy four-bar pop tune, and the voice follows it no matter what the LoRA learned. These LoRAs were trained without scores, so `off` is also simply the mode they were trained in. If a render sounds like a film song, check this setting first.

Headless, as an API prompt:

```python
prompt = {
  "1": {"class_type": "FSAudioLoraLoader", "inputs": {"lora_name": "qwwl_mehfil.safetensors", "strength_model": 1.0, "strength_clip": 1.0}},
  "2": {"class_type": "FSAudioModelLoader", "inputs": {"yue2_checkpoint": "yue2_3b_bf16.safetensors", "melody_transcriber": "none", "loras": ["1", 0]}},
  "3": {"class_type": "FSAudioSampler", "inputs": {"pipe": ["2", 0], "style": STYLE, "lyrics": LYRICS, "seed": 7,
        "song_length_cap": 170, "score_mode": "off", "steps": 32, "sampler": "dpm_2", "scheduler": "sgm_uniform",
        "Weirdness (cfg)": 1.0, "score_temperature": 0.7, "music_temperature": 1.0, "repetition_penalty": 1.2}},
  "4": {"class_type": "FSAudioOutput", "inputs": {"song": ["3", 0], "score": ["3", 1], "info": ["3", 2], "filename_prefix": "qwwl"}},
}
```

## Prompting

### Style prompt

Start with the trigger, then **one descriptive sentence**: language → genre → vocal delivery → instruments → mood → production. Describe the *delivery*, not just the genre; the training captions did.

```
qwwl, Urdu, dark traditional Sufi qawwali, guttural powerful male qawwal declaiming poetic couplets in free rhythm, male chorus answering the refrain in call and response, relentless repetition of the refrain building in intensity, rapid staccato sargam and tarana bursts, harmonium, full resonant tabla with booming bass strokes, dholak, handclaps, haunting longing mood building to ecstatic trance, live room recording
```

Words that the captions used and the LoRAs respond to: *free-rhythm alaap, declaimed poetic couplets, call and response between lead and chorus, insistent repetition of a refrain, rapid staccato sargam (or tarana) runs, strained high held notes, guttural*; moods *dark, haunting, longing, ecstatic, devotional, ominous, anthemic*.

| file | what it gets you |
|---|---|
| `sargam_dark_beats.txt` | syllable runs over programmed beats, frame drums, sub bass (demo, **Midnight**) |
| `ethno_downtempo.txt` | slow broken beat, dub bass, tabla, drones, long held notes into runs (demo, **Midnight**) |
| `live_syllables_qawwali.txt` | lead and chorus trading sargam and tarana, harmonium, tabla, handclaps (demo, **Mehfil**) |
| `dark_traditional_qawwali.txt` | the live qawwali party (demo, **Mehfil**) |
| `dark_fusion.txt` | the voice over dark bass, slow beat, drones, sarangi (demo, **Midnight**) |

### Lyrics

Two ways to write for these LoRAs.

**Syllables only** - the closest to the source, and the easiest. No words at all, just what the voice improvises with, in tagged blocks:

```
[Verse]
Aaa, aaa
Sa re ga ma pa dha ni sa
Sa ni dha pa ma ga re sa
Ga re ga ma pa ma ga re ga ma dha ni dha ni sa

[Chorus]
Ta re da ni ta da ni
Ta re da ni ta da ni
Dim ta na na na dere na

[Bridge]
Aaa, aaa
Dha ni sa, dha ni sa, dha ni sa re ga
```

Long lines of sargam make fast staccato runs; short `Aaa` lines make slow held notes; repeat a tarana line to get the chorus answering. Start the style prompt with `wordless sargam and tarana syllables` (or `wordless vocalise and sargam`) in the language slot.

**With words** - **romanized** Urdu or Punjabi (Latin script, the way South Asian lyric sites spell it), in tagged blocks. Do not write a verse-chorus pop lyric; write the form:

```
[Verse]
one or two wordless "Aaa" lines, then a couplet or two, declaimed slowly (8-10 syllables a line)

[Chorus]
the refrain, written out again and again the way a chorus party repeats it,
with a longer variation line every third line

[Verse]
a second couplet (the inserted verse)

[Chorus]
the refrain broken down to its shortest phrase, repeated six to eight times

[Bridge]
sargam and tarana syllables as sung:  Sa re ga, re ga ma ...  Tana dere na, dim tana dere na ...

[Chorus]
the refrain, ending on its last two words repeated
```

Repetition is the point: write every repeat out.

### Which knob does what

- **File** first: Mehfil for the acoustic party, Midnight for the produced dark sound.
- **strength_model (decoder)** is the lever to push ; **strength_clip above 1.0 collapses the vocal** on YuE2 artist LoRAs in general.
- **score_mode off.** See above.

## Training

Trained on a single RTX 5090 with the FS_Audio Suite **Artist Trainer** on Comfy's own YuE2 weights. One run trains the planner LoRA and the decoder LoRA together; each file also carries full `vae2llm` / `llm2vae` projection diffs.

| | Mehfil | Midnight |
|---|---|---|
| dataset | 50 pieces, 153 min: long-form live traditional qawwali party performances, cut into distinct pieces of about three minutes | 37 pieces, 109 min: studio recordings of a qawwali voice with Western ambient, dub and trip-hop production |
| checkpoint | **step 300** of 500 | **step 300** of 400 |
| planner / decoder steps | 500 / 1000 | 400 / 800 |
| artist (planner) loss | 5.851 → 5.140 | 5.241 → 4.818 |
| decoder loss at checkpoint | 1.152 (from 1.192; jumps to 1.201 at step 500) | 1.151 (from 1.349) |

Hyper-parameters: rank 64 planner / rank 32 decoder, LR 3e-5 planner / 4e-5 decoder / 2e-5 I/O projections, artist fraction 0.6 vs regularizer pack, KL 0.1, batch 2 songs, 8192 max tokens, 750-frame windows, EMA 0.99, score-first 0, no transcription.

### What went wrong first, and what fixed it

The first run was one LoRA on a 71-item mix: crossover albums, club remixes, devotional lounge tracks, tabla electronica. The renders were rejected outright as film-pop with a thin tabla. What was actually wrong, in order of how much it mattered:

- **One sound per LoRA.** On the mixed set the planner loss fell 0.33 over the run; on every earlier LoRA in this series that worked it fell 0.64 to 1.03. The model was averaging three musics. Split into focused sets the drops were 0.71 and 0.42 at the published checkpoints.
- **Build from the purest recordings.** The long live performances, where the form actually lives (couplet, refrain, repetition, improvised syllable runs), had been left out in favour of studio crossover material. They became the whole of the qawwali set.
- **The score.** See [Why score off](#why-score-off).
- **The test lyric.** A rhymed verse-chorus lyric gets a verse-chorus song. The lyric shape above is the one the demos use.
- **Piece length.** A training piece has to fit the planner's 8192-token window (25 audio tokens per second plus the text). Pieces near 320 s were being dropped from planner training silently; everything here is ≤ 270 s.
- **The codec was not the problem.** Real tabla and real qawwali passed through YuE2's VAE alone (encode, decode, no generation) came back with every frequency band within 1 dB and onset envelopes correlated at 0.98 to 0.99. A weak tabla is a generation problem, which training can move.

Other notes:

- **Captions need a careful model and a human pass.** A fast audio LLM described instrumental tabla pieces as reggae (echoing example vocabulary from its own instructions) and called a high male voice female in a third of the items. A stronger model, an instruction prompt with no genre examples in it, and one line of factual context per set fixed it. Cross-checking the style caption against the lyric transcription (a described singer on a track transcribed as instrumental) catches fabrications cheaply.
- **Lyrics** were transcribed in Latin transliteration with sargam, tarana and tabla bols written as sung; wordless pieces get `[Instrumental]`.
- **Audio prep**: FLAC 44.1 kHz / 16-bit, ≤ 270 s, long performances cut into distinct pieces with short fades, nothing repeated.

## Known limitations

- Judged by ear only. Speech recognition is no use on this material, so lyric adherence is not measured.
- The voice is one tradition's male lead and chorus; there is no female lead in the sets.
- A third LoRA for solo tabla was part of this release and has been withdrawn: it did not produce a convincing instrument. Both LoRAs here carry tabla from their own recordings.
- Midnight's planner moved less than Mehfil's; much of its character comes from the decoder. It leans on the prompt more than Mehfil does.
- Renders here are capped at under three minutes; longer caps were not tested.
- With `score_mode full` both drift back toward pop song shapes.

## Files

```
qwwl_mehfil.safetensors       177 MB   traditional qawwali run, step 300 — planner + decoder LoRA (bf16 weights, fp32 projection diffs)
drksf_midnight.safetensors    177 MB   dark Sufi fusion run, step 300
demos/                        mp3 renders (192 kbps from the FLAC masters)
prompts/                      style prompts
```

## Support

These LoRAs are trained on my own GPU and released free. If they're useful to you and you'd like to chip in for compute, there's a Ko-fi: **[ko-fi.com/becausereasons](https://ko-fi.com/becausereasons)** <3

## License and credits

Weights are released under **CC BY-NC 4.0**, inherited from the YuE2-3B base model. Non-commercial use only; attribute "QWWL / DRKSF LoRAs by becausereasons".

- [YuE2](https://huggingface.co/m-a-p/YuE2-3B) by the Multimodal Art Projection (m-a-p) team; ComfyUI repack by [Comfy-Org](https://huggingface.co/Comfy-Org/YuE2).
- [ComfyUI-FS_Audio_Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite) by KytraScript / The Fixed Seed Company: inference nodes and the artist trainer.
- Sister releases: [BLGR Bulgarian voices](https://huggingface.co/becausereasons/yue2-blgr-bulgarian-voices), [CHNSN chanson française](https://huggingface.co/becausereasons/yue2-chnsn-chanson-francaise), [CNZN canzone italiana](https://huggingface.co/becausereasons/yue2-cnzn-canzone-italiana), [MLTNT militant roots reggae](https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae).
- Trained and documented by becausereasons, September 2026.
