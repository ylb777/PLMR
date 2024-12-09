import os
if __name__ == '__main__':
    cmd = f"python run_plmr.py --aspect 0 \
            --dim_reduction_start 5 \
            --dim_reduction_end 7 \
            --lr_trans 0.000005 \
            --lr_mlp 0.00002 \
            --gpu 0 \
            --max_length 256 \
            --continuity_lambda 10 \
            --epochs 10 \
            --sparsity_lambda 10 \
            --batch_size 4 \
            --sparsity_percentage 0.2 "
    os.system(cmd)
