# !/bin/bash

#scratch_vae_anomal_data_with_pe
#scratch_vae_anomal_data_without_pe
#scratch_vae_anomal_nomal_data_with_pe
#scratch_vae_anomal_nomal_data_without_pe


port_number=50006
bench_mark="Tuft"
obj_name='teeth_crop_onlyanormal'
trigger_word='teeth'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="scratch_vae_anomal_data_without_pe"
#  --use_position_embedder
accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder \
 --start_epoch 0 --max_train_epochs 60 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --vae_model_dir "/home/dreamyou070/SupervisedMED/result/Tuft/vae_train/train_vae_reconstruction_nomal_data/vae_models/vae_104.safetensors" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --attn_loss_weight 1.0 \
 --do_cls_train \
 --normal_weight 1