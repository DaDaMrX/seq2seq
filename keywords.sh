python keywords.py -d daily -s test -r 0.1 -k data/kw-ratio/keywords_test_01.txt
python keywords.py -d daily -s test -r 0.2 -k data/kw-ratio/keywords_test_02.txt
python keywords.py -d daily -s test -r 0.4 -k data/kw-ratio/keywords_test_04.txt
python keywords.py -d daily -s test -r 0.5 -k data/kw-ratio/keywords_test_05.txt

echo train 0.1
python keywords.py -d daily -s train -r 0.1 -k data/kw-ratio/keywords_train_01.txt
echo train 0.2
python keywords.py -d daily -s train -r 0.2 -k data/kw-ratio/keywords_train_02.txt
echo train 0.4
python keywords.py -d daily -s train -r 0.4 -k data/kw-ratio/keywords_train_04.txt
echo train 0.5
python keywords.py -d daily -s train -r 0.5 -k data/kw-ratio/keywords_train_05.txt