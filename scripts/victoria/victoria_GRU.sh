export CUDA_VISIBLE_DEVICES=0 

hidden_dim=52
MODEL=GRU
python -u run.py \
  --model $MODEL \
  --data victoria \
  --root_path ./data/victoria \
  --mse_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over1.2_under1.0/checkpoint_mse.pth \
  --custom_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over1.2_under1.0/checkpoint_custom.pth \
  --hidden_dim $hidden_dim \
  --seq_len 4 \
  --label_len 2 \
  --pred_len 1 \
  --over_penalty 1.2 \
  --under_penalty 1.0 \


python -u run.py \
  --model $MODEL \
  --data victoria \
  --root_path ./data/victoria \
  --mse_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over1.5_under1.0/checkpoint_mse.pth \
  --custom_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over1.5_under1.0/checkpoint_custom.pth \
  --hidden_dim $hidden_dim \
  --seq_len 4 \
  --label_len 2 \
  --pred_len 1 \
  --over_penalty 1.5 \
  --under_penalty 1.0 \

python -u run.py \
  --model $MODEL \
  --data victoria \
  --root_path ./data/victoria \
  --mse_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over2.0_under1.0/checkpoint_mse.pth \
  --custom_model_path checkpoints/victoria/modelGRU_seqlen4_labellen2_predlen1_over2.0_under1.0/checkpoint_custom.pth \
  --hidden_dim $hidden_dim \
  --seq_len 4 \
  --label_len 2 \
  --pred_len 1 \
  --over_penalty 2.0 \
  --under_penalty 1.0 \