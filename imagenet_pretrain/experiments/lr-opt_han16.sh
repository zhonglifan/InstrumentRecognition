
seed=$1
model="han_model"
EXP_DIR="han16+lr-opt"
RUN_DIR="seed-$seed"

python main.py \
--model $model \
--train-meta "metadata/irmas_train_full_shuffle_slice.json" \
--valid-meta "metadata/irmas_valid_full_shuffle_slice.json" \
--wav-dir "./irmas_data/IRMAS-TrainingData" \
--seed $seed \
--epoch 50 \
--bs 128 \
--lr 0.001 \
--lr-warmup-epochs 5 \
--wd 1e-4 \
--optim "cos" \
--save-metric "loss" \
--output_dir "results/$EXP_DIR/$RUN_DIR"

python evaluator.py --model $model --ckpt_dir "results/$EXP_DIR/$RUN_DIR" --mode test --seg-dur 1
