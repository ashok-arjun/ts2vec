METHOD='ts2vec'
TRAIN_SLICE_END=(0.95 0.6)
VALID_SLICE_END=(0.98 0.8)

for i in ${!TRAIN_SLICE_END[@]}; do
    for iters in 10000 20000 30000
    do
        for dims in 128 256 320
        do
            RUN_NAME="${METHOD}_unsupervised_trainend_${TRAIN_SLICE_END[$i]}_dims_${dims}_iters_${iters}"
        
            python -u train.py Changping_PM2.5 $RUN_NAME --loader 'PM2.5' \
            --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
            --method $METHOD --tags 'pretraining' 'beijing' 'paper' --target_col_indices -1 \
            --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
            --max-train-length 12 --iters $iters --train \
            --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}
        done
    done
done
