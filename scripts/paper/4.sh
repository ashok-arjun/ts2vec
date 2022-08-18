METHOD='cost'

DOWNSTREAM_TRAIN_SLICE_START=(0 0.1 0.2 0.3 0.4 0.5)
CKPT_FOLDER_NAME="trained_models_Changping_ERA5"

for METHOD in 'cost'
do
    for iters in 100
    do
        for dims in 512
        do
            RUN_NAME="${METHOD}_unsupervised_training_on_ERA5_dims_${dims}_iters_${iters}_testing"

            # python -u train.py Changping_ERA5 $RUN_NAME --loader 'PM2.5_forecasting' \
            # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
            # --method $METHOD --tags 'Beijing-ERA5-Pretraining' \
            # --ckpt_location "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
            # --max-train-length 201 --iters $iters --train

            for protocol in "ridge" "neural_network"
            do
                for downstart in $DOWNSTREAM_TRAIN_SLICE_START
                do
                    EVAL_RUN_NAME="${METHOD}_unsupervised_model_eval_dims_${dims}_iters_${iters}_downStream_train_start_${downstart}_protocol_${protocol}"
                    python -u train.py Changping_PM2.5_With_ERA5 "${EVAL_RUN_NAME}" --loader 'PM2.5' \
                    --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    --method $METHOD --tags 'Beijing-ERA5-Evaluation' 'tests' --target_col_indices -1 \
                    --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                    --max-train-length 201 --eval \
                    --train_slice_start $downstart --regression_protocol $protocol

                    # python -u train.py Changping_PM10 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
                    # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    # --method $METHOD --tags 'evaluation' 'beijing' 'paper' 'finetuning' 'new-evaluation' --target_col_indices -1 \
                    # --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                    # --max-train-length 201 --eval \
                    # --train_slice_start $downstart
                done
            done
        done
    done
done