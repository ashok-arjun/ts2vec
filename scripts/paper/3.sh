TRAIN_SLICE_END=(0.95 0.6)
VALID_SLICE_END=(0.98 0.8)

DOWNSTREAM_TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)

for METHOD in 'cost'
do
    for i in ${!TRAIN_SLICE_END[@]}; do
        for iters in 10000 20000 30000
        do
            for dims in 256
            do
                for downstart in 0 0.1 0.2 0.3 0.4 0.5
                do
                    RUN_NAME="${METHOD}_unsupervised_trainend_${TRAIN_SLICE_END[$i]}_dims_${dims}_iters_${iters}"
                    EVAL_RUN_NAME="${METHOD}_unsupervised_eval_trainEnd_${TRAIN_SLICE_END[$i]}_dims_${dims}_iters_${iters}_downStream_train_start_${downstart}"

                    # python -u train.py Changping_PM2.5 $RUN_NAME --loader 'PM2.5_forecasting' \
                    # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    # --method $METHOD --tags 'pretraining' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
                    # --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    # --max-train-length 24 --iters $iters --train \
                    # --train_slice_end ${TRAIN_SLICE_END[$i]} --valid_slice_end ${VALID_SLICE_END[$i]}

                    python -u train.py Changping_PM2.5 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'finetuning' 'beijing' 'paper' 'modified-ts'  --target_col_indices -1 \
                    --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    --max-train-length 24 --eval \
                    --train_slice_start $downstart

                    python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM10' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'finetuning' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
                    --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    --max-train-length 24 --eval \
                    --train_slice_start $downstart

                    python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'finetuning' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
                    --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    --max-train-length 24 --eval \
                    --train_slice_start $downstart

                    python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM10_forecasting' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'finetuning' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
                    --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    --max-train-length 24 --eval \
                    --train_slice_start $downstart

                    python -u train.py Changping_WD "${EVAL_RUN_NAME}" --loader 'BeijingWD' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'finetuning' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
                    --load_ckpt "trained_models_Changping/${RUN_NAME}/model.pkl" \
                    --max-train-length 24 --eval \
                    --train_slice_start $downstart
                done
            done
        done
    done
done

# METHOD='ts2vec'
# dims=512
# for iters in 10000 20000 30000
# do
#     RUN_NAME="${METHOD}_unsupervised_trainend_0.6_dims_${dims}_iters_${iters}"
#     python -u train.py Changping_PM2.5 $RUN_NAME --loader 'PM2.5_forecasting' \
#     --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
#     --method $METHOD --tags 'pretraining' 'beijing' 'paper' 'modified-ts' --target_col_indices -1 \
#     --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
#     --max-train-length 24 --iters $iters --train \
#     --train_slice_end 0.6 --valid_slice_end 0.8
# done

# RUN_NAME="Changping_PM2.5_cost_pretraining_debug_forecasting"
# python -u train.py Changping_PM2.5 $RUN_NAME --loader 'PM2.5_forecasting' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'tests' --target_col_indices -1 \
# --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
# --max-train-length 24 --iters 2 --train --eval

# RUN_NAME="Changping_WD_cost_pretraining_debug_classification_no_fullseries"
# python -u train.py Changping_WD $RUN_NAME --loader 'BeijingWD' \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'tests' --target_col_indices -1 \
# --ckpt_location "trained_models_Changping/${RUN_NAME}/model.pkl" \
# --max-train-length 24 --iters 2 --train --eval