python src/model.py \
    --input_file data/sample.json \
    --encoder_name xlm-roberta-large \
    --lr 5e-6 \
    --epoch 3\
    --frac_warmup 0.1 \
    --gradient_accumulations 8 \
    --batch_size 2 