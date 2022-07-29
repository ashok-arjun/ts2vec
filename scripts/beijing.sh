# # PM2.5

# python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_train_test --loader 'PM2.5' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' --target_col_indices -1 \
# --ckpt_location "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --train \
# --train_slice_end 0.9 --valid_slice_end 0.95 --max-train-length 12 --iters 2

# python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_train_test --loader 'PM2.5' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' --target_col_indices -1 \
# --load_ckpt "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --eval --max-train-length 12

# PM2.5_forecasting


# python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_train_test --loader 'PM2.5_forecasting' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' --target_col_indices -1 \
# --ckpt_location "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --train \
# --train_slice_end 0.9 --valid_slice_end 0.95 --max-train-length 12 --iters 2

# python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_train_test --loader 'PM2.5_forecasting' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' --target_col_indices -1 \
# --load_ckpt "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --eval --max-train-length 12

# Beijing WD classification

# python -u train.py Changping_WD Changping_2.5_weather_30k_0.9_train_test --loader 'BeijingWD' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' --target_col_indices -1 \
# --ckpt_location "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --train \
# --train_slice_end 0.9 --valid_slice_end 0.95 --max-train-length 12 --iters 2

python -u train.py Changping_WD Changping_2.5_weather_30k_0.9_train_test --loader 'BeijingWD' \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
--method cost --tags 'training' 'beijing' --target_col_indices -1 \
--load_ckpt "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --eval --max-train-length 12