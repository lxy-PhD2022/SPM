export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=SPM_TST

root_path_name=./dataset/
data_path_name=PEMS03.npz
model_id_name=traffic
data_name=PEMS04

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path PEMS04.npz \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 307 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 512 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 300\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.2\
      --dishts 0\
      --num_layers 3\
      --hidden_size 32 8 \
      --affine 1 \
      --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'PEMS04'_'$seq_len'_'$pred_len.log
done