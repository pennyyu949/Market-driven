export CUDA_VISIBLE_DEVICES=0 

python -u run.py \
  --model MLP \
  --data catalonia \
  --root_path ./data/catalonia \
  --mse_model_path checkpoints/catalonia/modelMLP_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_mse.pth \
  --custom_model_path checkpoints/catalonia/modelMLP_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_custom.pth \
  --hidden_dim 179 \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --over_penalty 1.0 \
  --under_penalty 1.75 \


python -u run.py \
  --model GRU \
  --data catalonia \
  --root_path ./data/catalonia \
  --mse_model_path checkpoints/catalonia/modelGRU_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_mse.pth \
  --custom_model_path checkpoints/catalonia/modelGRU_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_custom.pth \
  --hidden_dim 131 \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --over_penalty 1.0 \
  --under_penalty 1.75 \

python -u run.py \
  --model LSTM \
  --data catalonia \
  --root_path ./data/catalonia \
  --mse_model_path checkpoints/catalonia/modelLSTM_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_mse.pth \
  --custom_model_path checkpoints/catalonia/modelLSTM_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_custom.pth \
  --hidden_dim 168 \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --over_penalty 1.0 \
  --under_penalty 1.75 \

python -u run.py \
  --model Crossformer \
  --data catalonia \
  --root_path ./data/catalonia \
  --mse_model_path checkpoints/catalonia/modelCrossformer_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_mse.pth \
  --custom_model_path checkpoints/catalonia/modelCrossformer_seqlen24_labellen12_predlen24_over1.0_under1.75/checkpoint_custom.pth \
  --hidden_dim 264 \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --over_penalty 1.0 \
  --under_penalty 1.75 \