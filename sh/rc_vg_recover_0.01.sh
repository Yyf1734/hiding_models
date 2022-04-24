for ((i = 0; i < 10; i+=1)); do
    echo "0.01 wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path /home/myang_20210409/yyf/model_hiding/official_result/1+1_for_larger/resnet-cifar_vgg-gtsrb/rc_vg_0.01 --prune_ratio 0.${i} --defense_method prune --recover_for_pruning > "/home/myang_20210409/yyf/model_hiding/official_result/defense/rc+vg_diff_ratio/0.01_rc_m5_wp_0.${i}_recover.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "0.01 fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path /home/myang_20210409/yyf/model_hiding/official_result/1+1_for_larger/resnet-cifar_vgg-gtsrb/rc_vg_0.01 --prune_ratio 0.${i} --defense_method prune --filter_pruning --recover_for_pruning > "/home/myang_20210409/yyf/model_hiding/official_result/defense/rc+vg_diff_ratio/0.01_rc_m5_fp_0.${i}_recover.txt";
done