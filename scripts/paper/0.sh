export CUDA_VISIBLE_DEVICES=0

CKPT_FOLDER_NAME="trained_models_Changping_ERA5_cost"

for METHOD in 'cost'
do
    for train_slice_start in 0.2
    do
        for iters in 10000 # 20000 30000
        do
            for dims in 512
            do
                RUN_NAME="${METHOD}_unsupervised_training_on_ERA5_dims_${dims}_iters_${iters}_trainstart_${train_slice_start}"

                python -u train.py 'PM2.5' $RUN_NAME --loader 'PM2.5' \
                --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                --method $METHOD --tags 'Beijing-ERA5-Pretraining' 'Varying-Pretraining-History' \
                --ckpt_location "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" --target_col_indices -1 \
                --max-train-length 201 --iters $iters --train --train_slice_start $train_slice_start

                for downstart in 0 0.1 0.2 0.3 0.4 0.5
                do
                    for protocol in "ridge" # "neural_network"
                    do
                        EVAL_RUN_NAME="${METHOD}_unsupervised_model_eval_dims_${dims}_iters_${iters}_pretrain_trainstart_${train_slice_start}_downStream_train_start_${downstart}_protocol_${protocol}"

                        python -u train.py PM2.5 "${EVAL_RUN_NAME}" --loader 'PM2.5' \
                        --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                        --method $METHOD --tags 'Beijing-ERA5-Evaluation' --target_col_indices -1 \
                        --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                        --max-train-length 201 --eval \
                        --train_slice_start $downstart --regression_protocol $protocol

                        python -u train.py PM2.5 "${EVAL_RUN_NAME}" --loader 'PM2.5_forecasting' \
                        --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                        --method $METHOD --tags 'Beijing-ERA5-Evaluation' --target_col_indices -1 \
                        --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                        --max-train-length 201 --eval \
                        --train_slice_start $downstart --regression_protocol $protocol

                        python -u train.py PM10 "${EVAL_RUN_NAME}" --loader 'PM10' \
                        --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                        --method $METHOD --tags 'Beijing-ERA5-Evaluation' --target_col_indices -1 \
                        --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                        --max-train-length 201 --eval \
                        --train_slice_start $downstart --regression_protocol $protocol

                        python -u train.py PM10 "${EVAL_RUN_NAME}" --loader 'PM10_forecasting' \
                        --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                        --method $METHOD --tags 'Beijing-ERA5-Evaluation' --target_col_indices -1 \
                        --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                        --max-train-length 201 --eval \
                        --train_slice_start $downstart --regression_protocol $protocol
                    done

                    # EVAL_RUN_NAME="${METHOD}_unsupervised_model_eval_dims_${dims}_iters_${iters}_pretrain_trainstart_${train_slice_start}_downStream_train_start_${downstart}"

                    # python -u train.py Changping_WD_With_ERA5 "${EVAL_RUN_NAME}" --loader 'BeijingWD' \
                    # --repr-dims $dims --max-threads 8 --seed 42 --batch-size 32  \
                    # --method $METHOD --tags 'Beijing-ERA5-Evaluation' --target_col_indices -1 \
                    # --load_ckpt "${CKPT_FOLDER_NAME}/${RUN_NAME}/model.pkl" \
                    # --max-train-length 201 --eval \
                    # --train_slice_start $downstart
                done

            done
        done
    done
done