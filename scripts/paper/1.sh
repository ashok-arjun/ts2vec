METHOD='ts2vec'
TRAIN_SLICE_END=(0.95 0.6)
VALID_SLICE_END=(0.98 0.8)

DOWNSTREAM_TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)

for i in ${!TRAIN_SLICE_END[@]}; do
    for iters in 10000 20000 30000
    do
        for dims in 128 256 320
        do
            for downstart in 0 0.1 0.2 0.3 0.4 0.5
            do
                RUN_NAME="${METHOD}_unsupervised_trainend_${TRAIN_SLICE_END[$i]}_dims_${dims}_iters_${iters}"
                EVAL_RUN_NAME="${METHOD}_unsupervised_eval_trainEnd_${TRAIN_SLICE_END[$i]}_dims_${dims}_iters_${iters}_downStream_train_start_${downstart}"

                # python -u train.py Changping_PM2.5 $RUN_NAME --loader 'PM2.5' \
                # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                # --method $METHOD --tags 'pretraining' 'beijing' 'paper' --target_col_indices -1 \
                # --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
                # --max-train-length 12 --iters $iters --train \
                # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

                python -u train.py Changping_PM2.5 "${EVAL_RUN_NAME}" --loader 'PM2.5' \
                --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
                --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                --max-train-length 12 --eval \
                --train_slice_start $downstart

                python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM10' \
                --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
                --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                --max-train-length 12 --eval \
                --train_slice_start $downstart

                # python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
                # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                # --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
                # --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                # --max-train-length 12 --eval \
                # --train_slice_start $downstart

                # python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM10_forecasting' \
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
            done
        done
    done
done
