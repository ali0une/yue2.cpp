---
license: cc-by-nc-4.0
base_model:
  - m-a-p/YuE2-3B
  - Comfy-Org/YuE2
pipeline_tag: text-to-audio
language:
  - en
  - jam
tags:
  - lora
  - yue2
  - music-generation
  - text-to-music
  - song-generation
  - reggae
  - roots-reggae
  - dancehall
  - steppers
  - comfyui
  - fs_audio
  - ai-toolkit
---

# MLTNT — Militant Roots Reggae LoRAs for YuE2

Six artist-style LoRAs that push [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) into **modern militant roots reggae**: dark raspy male patois vocals, steppers grooves, deep sub bass, bubbling Hammond, nyabinghi drums, horn stabs, dub sirens and spring reverb. Conscious, apocalyptic, anthemic.

Each file patches **both halves** of YuE2 in one go: the autoregressive planner (writes the score, decides the arrangement and the vocal lines) and the flow-matching decoder (the sound). Trigger word for all of them: **`mltnt`**.

**September 2026 update: a new-trainer generation.** `mltnt_soundclash` and `mltnt_chanter` were trained on the Frontline data with a different trainer (the experimental YuE2 support in [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit)). The voice is markedly more expressive and more idiomatic than on the four earlier files, and the range is wider: hip-hop, dancehall, lovers rock, fast steppers and slow roots from the same weights. They need one extra habit, planner strength 0.5 for anything fast or unusual, explained below. The four earlier files stay available and unchanged.

*Last updated: 20 September 2026.*

**Try it in the browser:** [MLTNT demo Space](https://huggingface.co/spaces/becausereasons/mltnt-reggae-yue2-demo), all six LoRAs on free ZeroGPU, with one-click recipes for the demos below. Built by the Hugging Face open-source team and kindly transferred to this account.

| | LoRA | File | Character | Start here |
|---|---|---|---|---|
| 💥 | **MLTNT Soundclash** | `mltnt_soundclash.safetensors` | New-trainer generation, step 500. The versatile one: hip-hop with sub drops, hard dancehall, tender lovers rock. The most expressive voice of the set. | **clip 0.5** / model 1.0 for fast and hard styles, clip 1.0 for slow songs |
| 🦁 | **MLTNT Chanter** | `mltnt_chanter.safetensors` | New-trainer generation, step 600. A more settled signature voice. At clip 1.0 it writes like Frontline (fast 160 BPM grid) with far more vocal musicality; at clip 0.5 it turns the same prompt into a slow heavy roots tune. | clip 1.0 or **clip 0.5**, model 1.0 |
| 🔥 | **MLTNT Frontline** | `mltnt_frontline.safetensors` | The newest. Most distinctive voice tonality of the four, and the one trained with extra short excerpts of fast-delivery verses, so it is the pick for rapid-fire lyrics (see *Rapid-fire verses* below). Works on both recipes. | clip 1.0 / model 1.0, cfg 1.0 |
| 🎤 | **MLTNT Fusion** | `mltnt_fusion.safetensors` | Widest palette. Reggae hip-hop fusion: boom-bap over one-drop, deejay toasting, sampled roots hooks. Happily moves off the 76 BPM roots grid. | clip 1.0 / model 1.5, cfg 1.4 |
| 🥁 | **MLTNT Steppers** | `mltnt_steppers.safetensors` | The flagship. Tight, hard-hitting steppers with a big anthemic chorus. Most reliable vocal density. | clip 1.0 / model 1.0, cfg 1.0 |
| 🌿 | **MLTNT Roots** | `mltnt_roots.safetensors` | The purist. Straightest roots timbre, least steered toward any one arrangement. | clip 1.0 / model 1.0, cfg 1.0 |

## Listen: new-trainer generation

Seed 7, 32 steps `dpm_2` / `sgm_uniform`, 360 s cap, no post-processing. Every song below ended on its own. `clip` is `strength_clip` (the planner), `model` is `strength_model` (the decoder).

**Soundclash, hip-hop bass drops** — `prompts/hiphop_bass_drops.txt`, dense lyric, clip 0.5 / model 1.0 (96 BPM, D♯ minor, 3:02):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_soundclash__hiphop_bass_drops.mp3"></audio>

**Soundclash, lovers rock** — `prompts/lovers_rock.txt`, standard lyric, full strength (73 BPM, D major, 4:26):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_soundclash__lovers_rock.mp3"></audio>

**Soundclash, dancehall rapid-fire** — `prompts/dancehall_rapidfire.txt`, standard lyric, clip 0.5 / model 1.0 (3:14):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_soundclash__dancehall_rapidfire.mp3"></audio>

**Chanter, the Frontline recipe at full strength** — `prompts/steppers_baseline.txt`, dense lyric, baseline recipe: the same prompt, lyric and seed as the Frontline demo below, and the planner picks the same 160 BPM F minor grid, with a much more musical voice (3:12):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_chanter__frontline_recipe.mp3"></audio>

**Chanter, the same recipe at clip 0.5** — nothing else changed. With the planner at half strength the base model has more say, and the same words become a 76 BPM roots groove in A minor (3:10). This pair is the clearest picture of what the planner-strength knob does:

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_chanter__frontline_recipe_clip05.mp3"></audio>

### Using the new-trainer files

They load exactly like the others (same fused-key layout, `text_encoders.*` planner and `diffusion_model.*` decoder; the LoRA matrices are stored as `lora_A` / `lora_B`), through the same FS_Audio graph shown in *Quick start*.

- **Fast, hard or unusual styles (hip-hop, dancehall, fusion, fire chant): clip 0.5 / model 1.0.** At full planner strength these files over-commit: everything gets faster and more frantic, and occasionally the planner writes a runaway score stuck in one section. Half strength keeps the voice and hands song structure back to the base model.
- **Slow songs (lovers rock, roots): clip 1.0 / model 1.0** works and gives the most character.
- **Use a 360 s cap.** These files plan longer songs than the earlier ones (typically 250 to 320 s for the full lyric); a 240 s cap cuts most of them before the last chorus. In score mode the song length is decided up front by the score, so the cap only truncates.
- **Fast delivery is a lever, not a default.** The phrase `rapid-fire double-time deejay flow` led the captions of the fast training excerpts and steers these files strongly: put it first in the style and pair it with a dense lyric and the *whole song* goes double time. For a fast chant in one section only, describe it in the style as a bridge feature and write dense lines in that section alone.
- **Checkpoints differ in variety.** Soundclash (step 500) varies more from seed to seed and prompt to prompt; Chanter (step 600) converges on one voice and style. Pick by whether you want range or a signature.
- The Fusion recipe's `strength_model` 1.5 was tuned for the earlier files; start the new ones at model 1.0.

What the new trainer does differently, what it costs, how to spot a runaway score from the `.abc` sidecar, and the Windows training notes are written up on the sister page: [CNZN, *Two trainers: what we learned*](https://huggingface.co/becausereasons/yue2-cnzn-canzone-italiana#two-trainers-what-we-learned). The reggae run confirmed all of it: same recipe (650 steps, rank 32, planner KL anchor, short whole excerpts in the set), usable window steps 400 to 600, step 300 too early to trust.

## Listen: the four earlier files

All demos use the same original lyric, seed 7, 32 steps `dpm_2` / `sgm_uniform`, no post-processing.

**MLTNT Frontline** — baseline recipe, prompt `prompts/steppers_baseline.txt`, **dense lyric** (verses written at ~17 words per line; planner chose a 160 BPM grid in F minor, 3:57, roughly double the words per minute of the standard lyric)

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__demo.mp3"></audio>

The same model, prompt, recipe and seed with the **standard 8-word-line lyric** — only the lyric differs, so this pair is the rapid-fire comparison (135 BPM F minor, 3:02):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__standard_lyric.mp3"></audio>

Frontline on the Fusion recipe with the dense lyric, prompt `prompts/fusion_hiphop.txt` (160 BPM, 3:21):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__fusion_dense.mp3"></audio>

Frontline on the Fusion recipe with the trap-dub prompt, `prompts/trap_dub.txt` with the phrase `rapid-fire double-time deejay flow` added to the vocal description, standard lyric (150 BPM F minor, 3:00). The cue phrase does not speed the delivery up (see *Rapid-fire verses*), but this is the prompt family that gives Frontline its wildest, most trap-leaning plans:

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__trap_dub.mp3"></audio>

Three renders kept for reference from **earlier Frontline builds** (not published as weights). First, the trap-dub prompt with the **dense lyric** on the build that preceded the current one, i.e. the same data without the fast-verse excerpts (160 BPM F minor, 3:13, about 110 words per minute):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__trap_dub_dense_previous_build.mp3"></audio>

Second, the same earlier build on the Fusion recipe with a style prompt written in the strict native caption order (language → genre → vocal → instruments → mood → BPM → production, the shape of `prompts/steppers_native_format.txt`) and a different, longer original lyric sheet. The planner wrote a full 4:38 song at 145 BPM in E minor and sang it through:

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__native_caption_previous_build.mp3"></audio>

Third, the original Frontline demo: the plain trap-dub prompt on the first Frontline build (160 BPM F minor, 88 % of vocal bars sung):

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_frontline__trap_dub_previous_build.mp3"></audio>

**MLTNT Fusion** — Fusion recipe, prompt `prompts/fusion_hiphop.txt` (planner chose a 150 BPM grid in G♯ minor, three verse-chorus rounds and an outro)

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_fusion__demo.mp3"></audio>

**MLTNT Steppers** — baseline recipe, prompt `prompts/steppers_baseline.txt`

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_steppers__demo.mp3"></audio>

**MLTNT Roots** — baseline recipe, prompt `prompts/steppers_baseline.txt` (planner stayed on the 76 BPM roots grid in A minor, 4:59 long, 149 of 190 vocal bars sung)

<audio controls src="https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/demos/mltnt_roots__demo.mp3"></audio>

## Quick start (ComfyUI)

These LoRAs are in the fused-key layout that Comfy's YuE2 implementation uses (`text_encoders.*` for the planner, `diffusion_model.*` for the decoder). They were trained with, and load through, the **[FS_Audio Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite)** node pack.

1. ComfyUI ≥ v0.36.0 (native YuE2 support) and the FS_Audio Suite custom node pack.
2. Base model: `yue2_3b_bf16.safetensors` from [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) in `models/checkpoints/`.
3. Drop one `mltnt_*.safetensors` into `models/loras/`.
4. Chain the nodes:

```
🧩 FS_Audio Lora Loader  ──loras──▶  🎤 FS_Audio Model Loader  ──pipe──▶  🎵 FS_Audio Sampler  ──▶  💿 FS_Audio Output
   lora_name      = mltnt_steppers.safetensors      yue2_checkpoint = yue2_3b_bf16       style   = <prompt, starts with "mltnt,">
   strength_clip  = 1.0   (planner)                 melody_transcriber = none            lyrics  = <tagged lyric blocks>
   strength_model = 1.0   (decoder)                                                      score_mode = full
```

Sampler settings used for every demo:

| widget | value |
|---|---|
| steps / sampler / scheduler | 32 / `dpm_2` / `sgm_uniform` |
| score_mode | `full` |
| song_length_cap | 360 (a full song; 150 truncates mid-way) |
| repetition_penalty | 1.2 |

Two proven recipes:

| | Baseline (Steppers / Roots / Frontline) | Fusion |
|---|---|---|
| strength_clip (planner) | 1.0 | 1.0 |
| strength_model (decoder) | 1.0 | **1.5** |
| Weirdness (cfg) | 1.0 | **1.4** |
| score_temperature | 0.7 | **0.9** |
| music_temperature | 1.0 | **1.2** |

Headless, the same graph as an API prompt:

```python
prompt = {
  "1": {"class_type": "FSAudioLoraLoader", "inputs": {"lora_name": "mltnt_steppers.safetensors", "strength_model": 1.0, "strength_clip": 1.0}},
  "2": {"class_type": "FSAudioModelLoader", "inputs": {"yue2_checkpoint": "yue2_3b_bf16.safetensors", "melody_transcriber": "none", "loras": ["1", 0]}},
  "3": {"class_type": "FSAudioSampler", "inputs": {"pipe": ["2", 0], "style": STYLE, "lyrics": LYRICS, "seed": 7,
        "song_length_cap": 360, "score_mode": "full", "steps": 32, "sampler": "dpm_2", "scheduler": "sgm_uniform",
        "Weirdness (cfg)": 1.0, "score_temperature": 0.7, "music_temperature": 1.0, "repetition_penalty": 1.2}},
  "4": {"class_type": "FSAudioOutput", "inputs": {"song": ["3", 0], "score": ["3", 1], "info": ["3", 2], "filename_prefix": "mltnt"}},
}
```

## Prompting

### Style prompt

Start with the trigger, then write **one descriptive sentence** in this order: language → genre → vocal → instruments → mood → BPM → production. This is the format the planner responds to. A bare tag list or `[Intro]…[Outro]` scaffolding in the style field produces "odd, not reggae" output.

```
mltnt, Jamaican Patois English, modern militant roots reggae, dark raspy male vocal with heavy patois delivery, deep bassline with massive sub bass drops, skanking guitar, hammond organ bubble, nyabinghi percussion, horn stabs and dub sirens, apocalyptic conscious mood, 76 BPM, sparse hard-hitting arrangement, dub impact hits, harder drum impact, heavier low end, spring reverb and tape delay, space before an explosive anthemic chorus
```

Ready-made prompts in `prompts/`:

| file | what it gets you |
|---|---|
| `steppers_baseline.txt` | the approved baseline: sub bass drops, dub impact hits, steppers groove, explosive chorus |
| `steppers_native_format.txt` | the same brief rewritten in the strict caption order |
| `fusion_hiphop.txt` | reggae hip-hop fusion, deejay toasting, boom-bap over one-drop (best with **MLTNT Fusion**) |
| `trap_dub.txt` | trap and dubstep low end, gospel-stack chorus, tape-stop drops, 80 BPM written but the planner lands at 150–160. Behind three of the Frontline clips; results swing from great to chaotic between seeds, so treat it as the wild card |
| `trap_dubstep_fusion.txt` | militant roots fused with trap and dubstep low end, halftime breakdowns, gang-vocal chorus |

Production descriptors ("boom-bap", "deejay toasting", "steppers") move the planner's tempo grid and key far more than the numeric BPM in the same prompt.

### Lyrics

Tagged blocks, 4 lines per verse works best. Use `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`. **Do not add an empty `[Intro]`** — the planner will happily write an 18-bar instrumental intro on its own. Shape:

```
[Verse]
four lines, roughly eight words each

[Verse]
four more lines

[Pre-Chorus]
two short lines that build tension

[Chorus]
four lines, the hook repeated at least twice across the song

[Bridge]
a chant or a two-word call, then four lines

[Outro]
two to four closing lines
```

**Rapid-fire verses.** The deejay's delivery speed comes from the *lyric*, not from the prompt. Words like "rapid-fire", "double-time" or "fast flow" in the style prompt do not speed the vocal up (tested: they mostly push the planner toward a shorter, trap-flavoured plan). What works is line density: write the verse at **15–17 words per line**, eight lines to the verse, and keep the chorus at the normal 7–8 words so the contrast lands. On the same prompt and seed that roughly doubles the words per minute and the LoRAs pack the lines intelligibly instead of overrunning them. Shape of a fast verse line:

```
Dem seh the future bright but mi seh where the people stand, where the promise and the plan
```

Keep the syllables simple (one- and two-syllable words); dense lines full of long words still garble. The two Frontline clips above are this exact comparison: same model, prompt, recipe and seed, standard lyric vs dense lyric.

A 2-word pre-chorus and a chant bridge give the planner clear pacing cues. Repeat the `[Chorus]` block verbatim wherever you want it sung again.

### Which knob does what

- **strength_model (decoder)** is the lever to push. 1.0 → 1.5 makes the sound darker, rougher and more present without touching the writing.
- **strength_clip (planner) above 1.0 collapses the vocal.** At 1.25 the planner wrote 1 sung bar against 424 rests. Keep it at 1.0.
- **Weirdness (cfg)** changes only the sound, not the arrangement. 1.0 is clean, 1.4 is the Fusion sweet spot, 1.7 gets noticeably wilder.
- **music_temperature / score_temperature** are what actually change the writing (tempo grid, key, section lengths). 1.2 / 0.9 gave the most adventurous still-coherent plans.
- **Seeds** can end a song early (seed 44 stopped at 4:30 with a 360 s cap). That is the seed, not the cap.

## Training

Trained on a single RTX 5090 with the FS_Audio Suite **Artist Trainer** on Comfy's own YuE2 weights. One run trains the planner LoRA and the decoder LoRA together; the file also carries full `vae2llm` / `llm2vae` projection diffs.

| | Steppers | Fusion | Roots | Frontline |
|---|---|---|---|---|
| checkpoint | step 300 (final) | **step 200** of 350 | step 300 (final) | step 350 (final) |
| planner / decoder steps | 300 / 800 | 350 / 1000 | 300 / 800 | 350 / 600 |
| final artist (planner) loss | 4.650 | 5.280 | 4.471 | 4.853 |
| decoder loss at checkpoint | 1.191 | 1.219 | 1.151 | 1.280 |

Shared hyper-parameters: rank 64 planner / rank 32 decoder, LR 3e-5 planner / 4e-5 decoder / 2e-5 I/O projections, artist fraction 0.5 vs regularizer pack, KL 0.1, batch 2 songs, 8192 max tokens, 750-frame windows, EMA 0.99, score-first 0, no transcription.

**Why step 200 for Fusion.** The trainer names its `_best` checkpoint by *planner* loss, which keeps falling. The decoder bottomed at steps 150–200 and then overfit hard (1.22 → 1.58 by step 350); the `_best` file sounds burnt. Pick checkpoints from the decoder curve. Frontline was trained with the decoder capped at 600 steps; its decoder loss bottomed at step 200 and drifted up by only 0.002 by step 350, so the final checkpoint is used. Frontline's set also adds eight short excerpts (20–45 s) of the fastest-delivery verses, each with only its own lyric lines — that did not make speed promptable from the caption, but it is where its voice comes from.

### Captioning your own dataset

The planner learns from the captions as much as from the audio, so the caption format decides whether your prompts work later. What worked here:

- **One descriptive sentence per song, trigger first**, in the same order you will prompt in: language → genre → vocal → instruments → mood → BPM → production. Example shape: `mltnt, Jamaican Patois English, modern militant roots reggae, dark raspy male vocal with heavy patois delivery, deep bassline, skanking guitar, bubbling hammond organ, nyabinghi percussion, horn stabs and dub sirens, urgent conscious mood, 76 BPM, sparse hard-hitting arrangement, spring reverb and dub delays`.
- **No tag lists, no section scaffolding** (`[Intro]…[Outro]`) in the caption. Tag-list captions produced a model that responds to tag-list prompts and writes odd, un-reggae plans.
- **Measure BPM, don't guess.** `librosa.beat.beat_track` on each file, rounded, written as `NN BPM` near the end of the sentence.
- A vision-language model can draft the captions from the audio, but **check the vocal gender by hand**: high male registers get labelled "female" often enough that every batch needs a grep-and-fix pass. A wrong gender in the caption shows up as a wrong voice at inference.
- **Lyrics as tagged blocks** in the same format you will prompt with: `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, patois spelling kept as sung, no empty intro tag. Transcribe at high confidence or leave the line out; a mis-heard lyric teaches the planner the wrong syllable count for the bar.
- **Audio prep**: FLAC, trimmed to ≤ 320 s with a fade. MP3 rips with a damaged leading frame crash the dataset builder ("Header missing"), so transcode with `ffmpeg -err_detect ignore_err` first.
- **Trainer flags**: `score-first 0` and no automatic transcription. Letting the trainer transcribe melodies itself put the vocal melody in the instrument voice and rests in the vocal voice, and the decoder loss started climbing after step 200.

## Known limitations

- Vocals are patois-flavoured English only.
- Dense multi-syllable lines overrun their bars and come out garbled. Budget roughly words ÷ 2 seconds per line at ~120 wpm and shorten the line rather than fighting pronunciation.
- With the 150 s cap the planner still writes full-length intros and interludes, so the song truncates before the second chorus. Use 360.
- The planner occasionally writes a plan with almost no sung bars. Change the seed; do not raise planner strength.
- **The one drop rhythm is not properly learned.** Asking for it, even spelled out (kick and rim together on the third beat, empty first beat, 70 to 80 BPM), still gives a steppers or rockers kick pattern, on every file here including the new-trainer generation. The training captions mention one drop only as a phrase in the middle of the instrument list, and drum patterns are not part of the score the planner writes (melody and chords only), so the rhythm has to be learned from caption to sound and that handle was too weak. It likely needs more specific data: songs labelled with a leading rhythm tag right after the trigger, checked by ear, plus riddim or dub versions and short drum-and-bass excerpts where the pattern is exposed. By contrast, a tag that did lead the caption in training (`rapid-fire double-time deejay flow`) steers the new-trainer files strongly.
- The new-trainer files are less forgiving than the earlier four: step through clip 1.0 and 0.5 before judging a prompt, and expect the occasional runaway score at full strength on fast styles.
- No instrumental-only mode is baked in; the LoRAs assume a lyric.

## Files

```
mltnt_soundclash.safetensors    118 MB   new-trainer generation, step 500 (planner + decoder LoRA, bf16)
mltnt_chanter.safetensors       118 MB   new-trainer generation, step 600
mltnt_steppers.safetensors      177 MB   planner + decoder LoRA (bf16 weights, fp32 projection diffs)
mltnt_fusion.safetensors       177 MB
mltnt_roots.safetensors         177 MB
mltnt_frontline.safetensors     177 MB
demos/                          mp3 renders (192 kbps from the FLAC masters), seed 7 throughout
prompts/                        style prompts
```

## Support

These LoRAs are trained on my own GPU and released free. If they're useful to you and you'd like to chip in for compute, there's a Ko-fi: **[ko-fi.com/becausereasons](https://ko-fi.com/becausereasons)** <3

## License and credits

Weights are released under **CC BY-NC 4.0**, inherited from the YuE2-3B base model. Non-commercial use only; attribute "MLTNT LoRAs by becausereasons".

- [YuE2](https://huggingface.co/m-a-p/YuE2-3B) by the Multimodal Art Projection (m-a-p) team; ComfyUI repack by [Comfy-Org](https://huggingface.co/Comfy-Org/YuE2).
- [AI Toolkit](https://github.com/ostris/ai-toolkit) by Ostris: the trainer behind Soundclash and Chanter; real-audio tokenizer by Kytra.
- [ComfyUI-FS_Audio_Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite) by KytraScript / The Fixed Seed Company: inference nodes and the artist trainer.
- Trained and documented by becausereasons, September 2026.
