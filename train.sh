# Anneal
python trainer.py -t all-gt --anneal=gt \
    --train_pickle_path=data/dialogues_train_keywords_2500.pickle \
    --valid_pickle_path=data/dialogues_test_keywords_2500.pickle \
    --n_accum_batches=2 -c
python trainer.py -t all-pred --anneal=pred \
    --train_pickle_path=data/dialogues_train_keywords_2500.pickle \
    --valid_pickle_path=data/dialogues_test_keywords_2500.pickle \
    --n_accum_batches=2 -c

# Seq2Seq-12
python trainer.py -t seq2seq-12 --no_keywords -c \
    --train_pickle_path=data/dialogues_train_keywords_2000.pickle
python trainer.py -t seq2seq-12-nobert --no_keywords --no_bert -c \
    --train_pickle_path=data/dialogues_train_keywords_2000.pickle


python trainer.py -t kw-01 \
    --train_pickle_path=data/kw-ratio/dialogues_train_kw_01.pickle \
    --valid_pickle_path=data/kw-ratio/dialogues_test_kw_01.pickle \
    --n_accum_batches=2

python trainer.py -t kw-02 \
    --train_pickle_path=data/kw-ratio/dialogues_train_kw_02.pickle \
    --valid_pickle_path=data/kw-ratio/dialogues_test_kw_02.pickle \
    --n_accum_batches=2

python trainer.py -t kw-04 \
    --train_pickle_path=data/kw-ratio/dialogues_train_kw_04.pickle \
    --valid_pickle_path=data/kw-ratio/dialogues_test_kw_04.pickle \
    --n_accum_batches=2

python trainer.py -t kw-05 \
    --train_pickle_path=data/kw-ratio/dialogues_train_kw_05.pickle \
    --valid_pickle_path=data/kw-ratio/dialogues_test_kw_05.pickle \
    --n_accum_batches=2

# Case
python trainer.py -t kw-sfs \
    --valid_pickle_path=data/dialogues_test_keywords_2500.pickle \
    --metric_from=293
python trainer.py -t seq2seq-tie --no_keywords \
    --valid_pickle_path=data/dialogues_test_keywords_2500.pickle \
    --metric_from=176