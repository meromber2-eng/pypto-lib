---
name: task-submit
description: 使用task-submit执行任务
---
所有任务都需要使用task-submit提交

例如，执行models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py
source /usr/local/Ascend/cann-9.2.0/bin/setenv.bash

cd /home/pyptouser/weijiao/workspace/pypto_projects/pypto-lib

task-submit --device auto --run 'cd "$PWD" && PYTHONPATH="$PWD:$PYTHONPATH" python models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py -p a5'
