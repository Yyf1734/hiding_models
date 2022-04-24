for ((i = 1; i < 7; i+=1)); do
    echo "200w finetuning -${i}"
    python main_y.py --which_cuda 3 --test --test_path official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_m5 --defense_method finetune --ft_epochs 3 --ft_index -${i} > "official_result/defense/rc+M5/2000k_rc_m5_ft_-${i}.txt";
done

for ((i = 1; i < 7; i+=1)); do
    echo "full finetuning -${i}"
    python main_y.py --which_cuda 3 --test --test_path official_result/1+1_for_larger/2000k_rc/2000k_1k_rc0.02_m5 --defense_method finetune --ft_epochs 3 --ft_index -${i} > "official_result/defense/rc+M5/full_rc_m5_ft_-${i}.txt";
done
