export MODEL_NAME="pretrain/stable-diffusion-v1-5"
export DATASET_NAME="data/coco/train2017"
accelerate launch script/finetune_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=50 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="work_dirs/sd_coco_lora" \
  --validation_prompt="cute dragon creature"