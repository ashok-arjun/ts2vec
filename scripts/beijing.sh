# # PM2.5

# python -u train.py Changping_PM2.5 Changping_2.5_weather_30k_0.9_train --loader 'PM2.5_forecasting' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' 'tests' --target_col_indices -1 \
# --ckpt_location "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" --train \
# --train_slice_end 0.9 --valid_slice_end 0.95 --iters 2 --max-train-length 24 # use 201 for forecasting

python -u train.py Changping_WD Changping_2.5_weather_30k_0.9_train_eval_classif --loader 'BeijingWD' \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
--method cost --tags 'evaluation' 'beijing' 'tests' --target_col_indices -1 \
--load_ckpt "trained_models/Changping_2.5_weather_2_0.9_train_test/model_test.pkl" \
--max-train-length 24 --eval 

# python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
# --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
# --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
# --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
# --max-train-length 12 --eval \
# --train_slice_start $downstart

# python -u train.py Changping_WD "${EVAL_RUN_NAME}" --loader 'BeijingWD' \
# --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
# --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
# --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
# --max-train-length 12 --eval \
# --train_slice_start $downstart