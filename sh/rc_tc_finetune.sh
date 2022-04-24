for ((i = 1; i < 7; i+=1)); do
    echo "200w finetuning -${i}"
    python main_y.py --which_cuda 3 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_tc --defense_method finetune --ft_epochs 3 --ft_index -${i} > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/2000k_rc_tc_ft_-${i}.txt";
done

for ((i = 1; i < 7; i+=1)); do
    echo "full finetuning -${i}"
    python main_y.py --which_cuda 3 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_another/seed_104_11164k_10k_10k_0k_cifar10-ResNet18_0.1_imdb-TextCNN_0.01 --defense_method finetune --ft_epochs 3 --ft_index -${i} > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/full_rc_tc_ft_-${i}.txt";
done
