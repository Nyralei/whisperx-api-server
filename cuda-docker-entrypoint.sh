#!/bin/bash
readarray -t gpu_free_mem < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

if [ "${#gpu_free_mem[@]}" -eq 0 ]; then
    echo "Error: Could not retrieve GPU memory information from nvidia-smi."
    exit 1
fi

max_free=-1
max_idx=-1

for i in "${!gpu_free_mem[@]}"; do
    mem="${gpu_free_mem[$i]}"
    if [ "$mem" -gt "$max_free" ]; then
        max_free="$mem"
        max_idx="$i"
    fi
done

export CUDA_VISIBLE_DEVICES="$max_idx"
echo "GPU with index $max_idx has the most available memory (${max_free} MiB)."
echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec "$@"