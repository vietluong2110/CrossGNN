
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=CrossGNN
gpu=3
e_layers=1
for pred_len in 96 192 336 720; do
python -u run_longExp.py \
  --train_epochs 10\
  --is_training 1\
  --e_layers $e_layers\
  --gpu $gpu\
  --root_path ./dataset/solar \
  --data_path solar_AL.txt \
  --model_id Solar_$seq_len'_'$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len\
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/'0'_$model_name'_'$seq_len'_'$pred_len'_'0.01.log
done