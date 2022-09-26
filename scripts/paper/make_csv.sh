# # # PM2.5 Regression

# LOADER="PM2.5"
# DATASET="PM2.5"
# TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)

# for i in ${!TRAIN_SLICE_START[@]}; do
#     RUN_NAME="naive_baseline_${DATASET}_regression_train_start_${TRAIN_SLICE_START[$i]}"

#     python -u train.py $DATASET $RUN_NAME --loader $LOADER \
#     --max-threads 8 --seed 42 --batch-size 32  \
#     --tags 'evaluation' 'beijing' 'paper' 'tests' --target_col_indices -1 \
#     --max-train-length 24 --eval --save_csv_dir "datasets/ForTSAI/${DATASET}/${LOADER}_train_start_${TRAIN_SLICE_START[$i]}" \
#     --train_slice_start ${TRAIN_SLICE_START[$i]} --train_slice_end 0.6 --valid_slice_end 0.8
# done

LOADER="PM10"
DATASET="PM10"
TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)

for i in ${!TRAIN_SLICE_START[@]}; do
    RUN_NAME="naive_baseline_${DATASET}_regression_train_start_${TRAIN_SLICE_START[$i]}"

    python -u train.py $DATASET $RUN_NAME --loader $LOADER \
    --max-threads 8 --seed 42 --batch-size 32  \
    --tags 'evaluation' 'beijing' 'paper' 'tests' --target_col_indices -1 \
    --max-train-length 24 --eval --save_csv_dir "datasets/ForTSAI/${DATASET}/${LOADER}_train_start_${TRAIN_SLICE_START[$i]}" \
    --train_slice_start ${TRAIN_SLICE_START[$i]} --train_slice_end 0.6 --valid_slice_end 0.8
done

# LOADER="BeijingWD"
# DATASET="Changping_WD"
# TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)

# for i in ${!TRAIN_SLICE_START[@]}; do
#     RUN_NAME="naive_baseline_${DATASET}_classif_train_start_${TRAIN_SLICE_START[$i]}"

#     python -u train.py $DATASET $RUN_NAME --loader $LOADER \
#     --max-threads 8 --seed 42 --batch-size 32  \
#     --tags 'evaluation' 'beijing' 'paper' 'tests' --target_col_indices -1 \
#     --max-train-length 24 --eval --save_csv_dir "datasets/ForTSAI/${DATASET}/${LOADER}_train_start_${TRAIN_SLICE_START[$i]}" \
#     --train_slice_start ${TRAIN_SLICE_START[$i]} --train_slice_end 0.6 --valid_slice_end 0.8
# done

