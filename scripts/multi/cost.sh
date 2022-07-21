# Train

TRAIN_SLICE_END=(0.6 0.4 0.2)
SUFFIXES=("_train_until_2018" "_train_until_2017" "_train_until_2016")
VALID_SLICE_END=(0.8 0.6 0.4)

for START_DATE in "2004-12-31 19:00:00"
do
    RUN_NAME_1="RF_Only_Pretraining"
    echo $RUN_NAME_1

    python -u train.py neighbour $RUN_NAME_1 --loader forecast_csv \
    --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
    --train --load_feats --method cost --tags 'pretraining' 'new' \
    --start_date "${START_DATE}" --end_date '2015-08-15 15:00:00' \
    --train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
    --ckpt_location "trained_models/$RUN_NAME_1/model.pkl"

    for i in ${!TRAIN_SLICE_END[@]}; do
        RUN_NAME_EVAL="${RUN_NAME_1}_${SUFFIXES[$i]}"

        python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
        --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
        --eval --method cost \
        --batch-size 32 \
        --load_ckpt "trained_models/$RUN_NAME_1/model.pkl" \
        --tags 'evaluation-U' 'new' --target_col_indices -1 \
        --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

        python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
        --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
        --eval --method cost \
        --batch-size 32 \
        --load_ckpt "trained_models/$RUN_NAME_1/model.pkl" \
        --load_feats --tags 'evaluation-MV' 'new' \
        --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
    done

    RUN_NAME_2="RF_Only_Pretraining+Train_No_Target_$START_DATE"
    echo $RUN_NAME_2

    python -u train.py bahia_windspeed $RUN_NAME_2 --loader forecast_csv \
    --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201 --batch-size 32  \
    --train --load_feats --method cost --tags 'training' 'new' \
    --load_ckpt "trained_models/$RUN_NAME_1/model.pkl" \
    --ckpt_location "trained_models/$RUN_NAME_2/model.pkl"
    
    for i in ${!TRAIN_SLICE_END[@]}; do
        RUN_NAME_EVAL="${RUN_NAME_2}_${SUFFIXES[$i]}"

        python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
        --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
        --eval --method cost \
        --batch-size 32 \
        --load_ckpt "trained_models/$RUN_NAME_2/model.pkl" \
        --tags 'evaluation-U' 'new' --target_col_indices -1 \
        --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

        python -u train.py bahia_windspeed $RUN_NAME_EVAL --loader forecast_csv \
        --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
        --eval --method cost \
        --batch-size 32 \
        --load_ckpt "trained_models/$RUN_NAME_2/model.pkl" \
        --load_feats --tags 'evaluation-MV' 'new' \
        --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
    done

done