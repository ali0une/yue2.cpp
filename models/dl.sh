#!/bin/bash

wget --continue --limit-rate=10000k https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/main/YuE2-3B-BF16.gguf
wget --continue --limit-rate=10000k https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/main/YuE2-Vae-F32.gguf
wget --continue --limit-rate=10000k https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/main/SheetSage2-F32.gguf

beep

