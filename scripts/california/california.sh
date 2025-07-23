export CUDA_VISIBLE_DEVICES=0 

python -u run.py \
  --model MLP \
  --data california \
  --root_path ./data/california \
  --mse_model_path checkpoints/california/modelMLP_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_mse.pth \
  --custom_model_path checkpoints/california/modelMLP_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_custom.pth \
  --hidden_dim 120 \
  --seq_len 8 \
  --label_len 2 \
  --pred_len 4 \
  --over_penalty 1.1 \
  --under_penalty 0.9 \


python -u run.py \
  --model GRU \
  --data california \
  --root_path ./data/california \
  --mse_model_path checkpoints/california/modelGRU_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_mse.pth \
  --custom_model_path checkpoints/california/modelGRU_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_custom.pth \
  --hidden_dim 118 \
  --seq_len 8 \
  --label_len 2 \
  --pred_len 4 \
  --over_penalty 1.1 \
  --under_penalty 0.9 \

python -u run.py \
  --model LSTM \
  --data california \
  --root_path ./data/california \
  --mse_model_path checkpoints/california/modelLSTM_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_mse.pth \
  --custom_model_path checkpoints/california/modelLSTM_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_custom.pth \
  --hidden_dim 127 \
  --seq_len 8 \
  --label_len 2 \
  --pred_len 4 \
  --over_penalty 1.1 \
  --under_penalty 0.9 \

python -u run.py \
  --model Crossformer \
  --data california \
  --root_path ./data/california \
  --mse_model_path checkpoints/california/modelCrossformer_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_mse.pth \
  --custom_model_path checkpoints/california/modelCrossformer_seqlen8_labellen2_predlen4_over1.1_under0.9/checkpoint_custom.pth \
  --hidden_dim 104 \
  --seq_len 8 \
  --label_len 2 \
  --pred_len 4 \
  --over_penalty 1.1 \
  --under_penalty 0.9 \