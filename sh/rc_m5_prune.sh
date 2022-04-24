for ((i = 0; i < 10; i+=1)); do
    echo "200w wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_m5 --prune_ratio 0.${i} --defense_method prune > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+M5/2000k_rc_m5_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "200w fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_m5 --prune_ratio 0.${i} --defense_method prune --filter_pruning > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+M5/2000k_rc_m5_fp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_another/11164k_4k_4k_0k_cifar10-ResNet18_0.1_speechcommand-M5_0.1 --prune_ratio 0.${i} --defense_method prune > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+M5/full_rc_m5_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 2 --test --test_path /home/myang_20210409/yyf/model_overloading/official_result/1+1_for_another/11164k_4k_4k_0k_cifar10-ResNet18_0.1_speechcommand-M5_0.1 --prune_ratio 0.${i} --defense_method prune --filter_pruning > "/home/myang_20210409/yyf/model_overloading/official_result/defense/rc+M5/full_rc_m5_fp_0.${i}.txt";
done