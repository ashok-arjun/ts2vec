python -u train.py neighbour 'neighbour_feats_ts2vec' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--train --eval --load_feats --method ts2vec --tags 'test' \
--start_date '2010-08-15 15:00:00' --end_date '2015-08-15 15:00:00' \
--train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
--ckpt_location "trained_models/neighbour_feats_ts2vec/model.pkl"

python -u train.py neighbour 'neighbour_feats_ts2vec_bsz8' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--train --eval --load_feats --method ts2vec --tags 'pretraining' \
--start_date '2010-08-15 15:00:00' --end_date '2015-08-15 15:00:00' \
--train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 8 \
--ckpt_location "trained_models/neighbour_feats_ts2vec_bsz8/model.pkl"

python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 \
--train --eval --load_feats --method ts2vec --tags 'training' \
--ckpt_location "trained_models/bahia_windspeed_feats_ts2vec/model.pkl"

python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 \
--train --eval --load_feats --method ts2vec --tags 'training' \
--load_ckpt "trained_models/neighbour_feats_ts2vec/model.pkl" \
--ckpt_location "trained_models/bahia_windspeed_feats_ts2vec_continuing_neighbour/model.pkl"