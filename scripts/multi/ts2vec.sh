# Train

# TRAIN_SLICE_END=(0.6 0.4 0.2)
# SUFFIXES=("2018" "2017" "2016")
# VALID_SLICE_END=(0.8 0.6 0.4)

TRAIN_SLICE_END=(0.6)
SUFFIXES=("2018")
VALID_SLICE_END=(0.8)

PRETRAIN_START_DATES=("2004-12-31 19:00:00")
PRETRAIN_START_DATE_SUFFIXES=("2004")
# PRETRAIN_END_DATES=("2015-08-15 15:00:00")
PRETRAIN_END_DATES=("2020-01-14 23:00:00")
PRETRAIN_END_DATE_SUFFIXES=("2021")

for i in ${!TRAIN_SLICE_END[@]}; do
    RUN_NAME_3="RF_Only_Features_Training_With_CoST_until_${SUFFIXES[$i]}"
    
    # python -u train.py bahia_windspeed $RUN_NAME_3 --loader forecast_csv \
    # --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201 --batch-size 32  \
    # --train --load_feats --method cost --tags 'training' 'new-2' \
    # --ckpt_location "trained_models/${RUN_NAME_3}/model.pkl" \
    # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

    RUN_NAME_EVAL="${RUN_NAME_3}_then_eval_with_regression_until_${SUFFIXES[$i]}"

    python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
    --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
    --eval --method cost \
    --batch-size 32 \
    --load_ckpt "trained_models/${RUN_NAME_3}/model.pkl" \
    --tags 'evaluation-U' 'new-2' --target_col_indices -1 \
    --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]} --plot_preds

    # python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
    # --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
    # --eval --method cost \
    # --batch-size 32 \
    # --load_ckpt "trained_models/${RUN_NAME_3}/model.pkl" \
    # --load_feats --tags 'evaluation-MV' 'new-2' \
    # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
done

# for i in ${!PRETRAIN_START_DATES[@]}
# do
#     for j in ${!PRETRAIN_END_DATES[@]}
#     do
#         RUN_NAME_1="RF_Pretraining_from_${PRETRAIN_START_DATE_SUFFIXES[$i]}_to_${PRETRAIN_END_DATE_SUFFIXES[$j]}"
#         echo $RUN_NAME_1

#         python -u train.py neighbour "${RUN_NAME_1}" --loader forecast_csv \
#         --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
#         --train --load_feats --method cost --tags 'pretraining' 'new-2' \
#         --start_date "${PRETRAIN_START_DATES[$i]}" --end_date "${PRETRAIN_END_DATE_SUFFIXES[$j]}" \
#         --train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
#         --ckpt_location "trained_models/${RUN_NAME_1}/model.pkl"

#         for i in ${!TRAIN_SLICE_END[@]}; do
#             RUN_NAME_EVAL="${RUN_NAME_1}_then_eval_with_regression_until_${SUFFIXES[$i]}"

#             python -u train.py bahia_windspeed "${RUN_NAME_EVAL}" --loader forecast_csv \
#             --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
#             --eval --method cost \
#             --batch-size 32 \
#             --load_ckpt "trained_models/${RUN_NAME_1}/model.pkl" \
#             --tags 'evaluation-U' 'new-2' --target_col_indices -1 \
#             --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

#             # python -u train.py bahia_windspeed "${RUN_NAME_EVAL}" --loader forecast_csv \
#             # --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
#             # --eval --method cost \
#             # --batch-size 32 \
#             # --load_ckpt "trained_models/${RUN_NAME_1}/model.pkl" \
#             # --load_feats --tags 'evaluation-MV' 'new-2' \
#             # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
#         done

#         # for i in ${!TRAIN_SLICE_END[@]}; do

#         #     RUN_NAME_2="${RUN_NAME_1}_then_train_again_with_CoST_until_${SUFFIXES[$i]}"
#         #     echo $RUN_NAME_2

#         #     python -u train.py bahia_windspeed "${RUN_NAME_2}" --loader forecast_csv \
#         #     --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201 --batch-size 32  \
#         #     --train --load_feats --method cost --tags 'training' 'new-2' \
#         #     --load_ckpt "trained_models/${RUN_NAME_1}/model.pkl" \
#         #     --ckpt_location "trained_models/${RUN_NAME_2}/model.pkl" \
#         #     --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

#         #     RUN_NAME_EVAL="${RUN_NAME_2}_then_eval_with_regression_until_${SUFFIXES[$i]}"

#         #     python -u train.py bahia_windspeed "${RUN_NAME_EVAL}" --loader forecast_csv \
#         #     --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
#         #     --eval --method cost \
#         #     --batch-size 32 \
#         #     --load_ckpt "trained_models/${RUN_NAME_2}/model.pkl" \
#         #     --tags 'evaluation-U' 'new-2' --target_col_indices -1 \
#         #     --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

#         #     # python -u train.py bahia_windspeed "${RUN_NAME_EVAL}" --loader forecast_csv \
#         #     # --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
#         #     # --eval --method cost \
#         #     # --batch-size 32 \
#         #     # --load_ckpt "trained_models/${RUN_NAME_2}/model.pkl" \
#         #     # --load_feats --tags 'evaluation-MV' 'new-2' \
#         #     # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
#         # done
#     done
# done