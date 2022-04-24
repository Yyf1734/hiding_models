for ((i = 0; i < 10; i+=1)); do
    echo "200w wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_tc --prune_ratio 0.${i} --defense_method prune > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/2000k_rc_tc_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "200w fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_tc --prune_ratio 0.${i} --defense_method prune --filter_pruning > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/2000k_rc_tc_fp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_another/seed_104_11164k_10k_10k_0k_cifar10-ResNet18_0.1_imdb-TextCNN_0.01 --prune_ratio 0.${i} --defense_method prune > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/full_rc_tc_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_another/seed_104_11164k_10k_10k_0k_cifar10-ResNet18_0.1_imdb-TextCNN_0.01 --prune_ratio 0.${i} --defense_method prune --filter_pruning > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+tc/full_rc_tc_fp_0.${i}.txt";
done