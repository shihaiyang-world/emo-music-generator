# 基线训练


## 预训练脚本

```shell
cd baseline

CUDA_VISIBLE_DEVICES=0 python main_cp.py --mode train --task_type ignore --path_train_data ailabs --data_root '../co-representation/' --load_dict "dictionary.pkl" --epoch 100 --batch_size 64

```