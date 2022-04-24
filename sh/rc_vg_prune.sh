for ((i = 0; i < 10; i+=1)); do
    echo "200w wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_gtsrb --prune_ratio 0.${i} --defense_method prune > "official_result/defense/rc+vg/2000k_rc_vg_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "200w fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_gtsrb --prune_ratio 0.${i} --defense_method prune --filter_pruning > "official_result/defense/rc+vg/2000k_rc_vg_fp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full wp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path official_result/1+1_for_another/seed_1000_11164k_4k_4k_0k_cifar10-ResNet18_0.1_gtsrb-VGG_0.01 --prune_ratio 0.${i} --defense_method prune > "official_result/defense/rc+vg/full_rc_vg_wp_0.${i}.txt";
done

for ((i = 0; i < 10; i+=1)); do
    echo "full fp pruning ratio is 0.${i}"
    python main_y.py --which_cuda 0 --test --test_path official_result/1+1_for_another/seed_1000_11164k_4k_4k_0k_cifar10-ResNet18_0.1_gtsrb-VGG_0.01 --prune_ratio 0.${i} --defense_method prune --filter_pruning > "official_result/defense/rc+vg/full_rc_vg_fp_0.${i}.txt";
done