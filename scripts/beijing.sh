# python -u train.py Chinatown UCR --loader UCR --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval --train \
# # # --method ts2vec
# python -u train.py Changping_PM2.5 Changping_2.5_weather_20k_0.9_eval --loader Monash \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --eval --method ts2vec --tags 'training' 'beijing' --target_col_indices -1 --iters 20000 \
# --load_ckpt "trained_models/Changping_2.5_weather_20k_0.9_train/model_test.pkl"

python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_eval --loader Monash \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
--eval --method ts2vec --tags 'training' 'beijing' --target_col_indices -1 --iters 20000 \
--load_ckpt "trained_models/Changping_2.5_weather_30k_0.9_train/model_test.pkl"