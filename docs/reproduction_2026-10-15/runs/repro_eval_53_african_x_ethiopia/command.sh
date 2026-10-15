CUDA_VISIBLE_DEVICES=7 /home/yding/workbar/morph-runtime/bin/micromamba run -p /home/yding/workbar/morph-runtime/micromamba/envs/morph-anole python -u /home/yding/workbar/guide/scripts/28_eval_post_boi_protocol.py \
 --model-path /home/yding/workbar/morph-runtime/models/anole-7b-v0.1 --code-root /home/yding/workbar/morph-runtime/external/anole \
 --output-dir /home/yding/workbar/morph-runtime/runs/repro_eval_53_african_x_ethiopia/results --suffix-paths /home/yding/workbar/morph-runtime/runs/repro_person_53_african_x_ethiopia/results/soft_suffix.pt \
 --prompt-id ethiopia_coffee --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
 --clip-dir /home/yding/workbar/morph-runtime/models/clip-vit-base-patch32 --extra-captions "a photo of an African man" "a photo of coffee beans"
