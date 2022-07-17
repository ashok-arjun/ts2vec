# multivar
# python -u train.py ETTh1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval
# python -u train.py ETTh2 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ETTm1 ETTm1_feats --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval --load_feats

# univar
# python -u train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
# python -u train.py ETTh2 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
# python -u train.py ETTm1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
