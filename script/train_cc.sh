python NCKH/cc/train.py \
    --input_file NCKH/data/ise-dsc01-train_1.json \
    --encoder_name joeddav/xlm-roberta-large-xnli \
    --lr 5e-6 \
    --epoch 3\
    --frac_warmup 0.1 \
    --gradient_accumulations 8 \
    --batch_size 8 \
    --num_workers 16