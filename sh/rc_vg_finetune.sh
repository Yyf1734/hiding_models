
for ((i = 1; i < 7; i+=1)); do
    echo "full finetuning -${i}"
    python main_y.py --which_cuda 2 --test --test_path official_result/1+1_for_another/seed_1000_11164k_4k_4k_0k_cifar10-ResNet18_0.1_gtsrb-VGG_0.01 --defense_method finetune --ft_epochs 3 --ft_index -${i} > "official_result/defense/rc+M5/full_rc_vg_ft_-${i}.txt";
done
