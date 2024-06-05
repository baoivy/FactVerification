python NCKH/relevant/train.py \
    --input_file NCKH/data/sentence_extract.json \
    --encoder_name vinai/phobert-large \
    --lr 5e-6 \
    --epoch 3\
    --frac_warmup 0.1 \
    --gradient_accumulations 8 \
    --batch_size 16 \
    --num_workers 16