# kaggle-santander-prediction
Find Rules for direct find target in Kaggle Santander Competition

run as:
$ python3 -u build_model.py -r 3 -m model_fold1.pickle
7 model built.
303/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -m model_fold2.pickle
10 model built.
201/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -m model_fold3.pickle
5 model built.
241/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -m model_fold4.pickle
8 model built.
223/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -m model_fold5.pickle
16 model built.
400/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -i 6 -m model_fold6.pickle
490 models built.
930/4459 columns filled in train:

$ python3 -u build_model.py -r 6 -i 6 -m model_fold7.pickle
64 models built.
741/4459 columns filled in train:

$ python3 -u build_model.py -r 5 -i 6 -m model_fold8.pickle
125 models built.
745/4459 columns filled in train:

$ python3 -u build_model.py -r 6 -i 6 -m model_fold9.pickle
113 models built.
618/4459 columns filled in train:

$ python3 -u build_model.py -r 4 -i 6 -m model_fold10.pickle
171 models built.
819/4459 columns filled in train:

$ python3 -u build_model.py -r 3 -i 6 -m model_fold11.pickle
425 models built.
970/4459 columns filled in train:

$ python3 -u build_model.py -n 4 -r 3 -i 6 -m model_fold12.pickle
242 models built.
635/4459 columns filled in train:

$ python3 -u build_model.py -n 4 -r 3 -i 6 -m model_fold13.pickle
236 models built.
681/4459 columns filled in train:

$ python3 count_model.py -p 96 model_tri_fold1.pickle model_tri_fold2.pickle model_tri_fold3.pickle model_tri_fold4.pickle model_tri_fold5.pickle model_fold6.pickle model_fold7.pickle model_fold8.pickle model_fold9.pickle model_fold10.pickle model_fold11.pickle model_fold12.pickle model_fold13.pickle
4991 cols read.
1912 rules read:
count train target:
1463/4459 columns filled in train:
mean/max rules in matchs is (86.9002050580998/13537).

$ python3 submit_model.py -p 120 model_tri_fold1.pickle model_tri_fold2.pickle model_tri_fold3.pickle model_tri_fold4.pickle model_tri_fold5.pickle model_fold6.pickle model_fold7.pickle model_fold8.pickle model_fold9.pickle model_fold10.pickle model_fold11.pickle model_fold12.pickle model_fold13.pickle
4991 cols read.
1912 rules read:
submit model:
2583/49342 columns filled in submission:
mean/max rules in matchs is (43.092514/991).
mean/max stds in no-allcorrect matchs is (3105618.786881/19428571).
155 matchs is no-correct std>1.
mean/max stds in no-correct matchs is (5810512.569003/19428571).
