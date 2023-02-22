for seed in 2233 3073
do
  seed=$seed
  bs=128
  epoch=35
  model="resnet50"
  EXP_DIR="resnet50-lr-4"
  RUN_DIR="seed-$seed-bs-$bs-epoch-$epoch"

  python main.py \
  --model $model \
  --train-meta "metadata/irmas_train_slice.json" \
  --valid-meta "metadata/irmas_valid_slice.json" \
  --wav-dir "./irmas_data/IRMAS-TrainingData" \
  --seed $seed \
  --epoch $epoch \
  --bs $bs \
  --lr 0.0004 \
  --lr-warmup-epochs 5 \
  --wd 5e-4 \
  --optim "cos" \
  --save-metric "loss" \
  --els 100 \
  --normalize_amp \
  --imagenet_pretrained \
  --ch_expand "copy" \
  --output_dir "results/$EXP_DIR/$RUN_DIR"

  python evaluator.py --model $model --ckpt_dir "results/$EXP_DIR/$RUN_DIR" --mode test --seg-dur 1

done


