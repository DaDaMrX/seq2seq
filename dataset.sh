python dataset.py -d daily -s test \
    -p data/dialogues_test_keywords_1500.pickle \
    --max_batch_tokens 1500

python dataset.py -d daily -s train \
    -p data/dialogues_train_keywords_1200.pickle \
    --max_batch_tokens=1200


python dataset.py -d daily -s train \
    -p data/dialogues_train_keywords_2000.pickle \
    --max_batch_tokens=2000



echo test 0.1
python dataset.py -d daily -s test \
    -k data/kw-ratio/keywords_test_01.txt \
    -p data/kw-ratio/dialogues_test_kw_01.pickle
echo test 0.2
python dataset.py -d daily -s test \
    -k data/kw-ratio/keywords_test_02.txt \
    -p data/kw-ratio/dialogues_test_kw_02.pickle
echo test 0.4
python dataset.py -d daily -s test \
    -k data/kw-ratio/keywords_test_04.txt \
    -p data/kw-ratio/dialogues_test_kw_04.pickle
echo test 0.5
python dataset.py -d daily -s test \
    -k data/kw-ratio/keywords_test_05.txt \
    -p data/kw-ratio/dialogues_test_kw_05.pickle


echo train 0.1
python dataset.py -d daily -s train \
    -k data/kw-ratio/keywords_train_01.txt \
    -p data/kw-ratio/dialogues_train_kw_01.pickle
echo train 0.2
python dataset.py -d daily -s train \
    -k data/kw-ratio/keywords_train_02.txt \
    -p data/kw-ratio/dialogues_train_kw_02.pickle
echo train 0.4
python dataset.py -d daily -s train \
    -k data/kw-ratio/keywords_train_04.txt \
    -p data/kw-ratio/dialogues_train_kw_04.pickle \
    --max_batch_tokens=2000
echo train 0.5
python dataset.py -d daily -s train \
    -k data/kw-ratio/keywords_train_05.txt \
    -p data/kw-ratio/dialogues_train_kw_05.pickle \
    --max_batch_tokens=2000