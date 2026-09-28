#!/bin/bash

wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_frontline.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_fusion.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_roots.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_steppers.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_chanter.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-mltnt-militant-reggae/resolve/main/mltnt_soundclash.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc_comfyui.safetensors -O ar_instrumental_cot_full.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/drksf_midnight.safetensors -O sufi_drksf_midnight.safetensors
wget --continue --limit-rate=10000k https://huggingface.co/becausereasons/yue2-qwwl-qawwali-sufi-tabla/resolve/main/qwwl_mehfil.safetensors -O sufi_qwwl_mehfil.safetensors

beep

