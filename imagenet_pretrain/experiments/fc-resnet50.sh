for lr in 0.0005 0.001
do
  wd=1.8
  epoch=350
  seed=2233
  bs=128
  model="resnet50_revised"
  EXP_DIR="resnet50-fc-adamw-exp-lr-$lr"
  RUN_DIR="seed-$seed-bs-$bs-epoch-$epoch-wd-$wd"

  python main.py \
  --model $model \
  --train-meta "metadata/irmas_train_slice.json" \
  --valid-meta "metadata/irmas_valid_slice.json" \
  --wav-dir "./irmas_data/IRMAS-TrainingData" \
  --seed $seed \
  --epoch $epoch \
  --bs $bs \
  --lr $lr \
  --lr-warmup-epochs 5 \
  --wd $wd \
  --optim "exp" \
  --save-metric "loss" \
  --els $epoch \
  --normalize_amp \
  --imagenet_pretrained \
  --ch_expand "copy" \
  --output_dir "results/$EXP_DIR/$RUN_DIR"

  python evaluator.py --model $model --ckpt_dir "results/$EXP_DIR/$RUN_DIR" --mode test --seg-dur 1

done


