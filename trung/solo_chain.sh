cd /scratch/users/astar/ares/suns1/trung/nlp
T="python repo/trung/train.py --no_save"
$T --name solo_roberta_adamw --model roberta-base --optim adamw --lr 2e-5 > logs/solo_roberta_adamw.log 2>&1
$T --name solo_roberta_muon --model roberta-base --optim muon --lr 5e-5 --head_lr 2e-5 > logs/solo_roberta_muon.log 2>&1
$T --name solo_deberta_adamw --model microsoft/deberta-v3-base --optim adamw --lr 2e-5 > logs/solo_deberta_adamw.log 2>&1
