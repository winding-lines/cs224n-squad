python bert/train.py  --bert_model bert-base-uncased --output_dir bert_output \
    --do_predict --predict_file data/dev-v2.0.json \
    --do_train --train_file data/train-v2.0.json --num_train_epochs 1.0 \
    --max_seq_length 64 --train_batch_size 64 \
    --version_2_with_negative
