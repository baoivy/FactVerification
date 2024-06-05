python NCKH/rationale/train.py \
    --input_file NCKH/data/ise-dsc01-train_1.json \
    --encoder_name vinai/phobert-large \
    --lr 5e-6 \
    --epoch 2\
    --frac_warmup 0.1 \
    --gradient_accumulations 8 \
    --batch_size 16 \
    --monitor valid_accuracy \
    --num_workers 16