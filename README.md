# emo-music-generator



docker run -it -d --name remote-gpu --gpus all -p 2024:22 -p 17860:7860 registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.5-ssh-fluidsynth



```

在docker中确定ssh服务已经启动
```shell
# 查看状态
/etc/init.d/ssh status
# 重启ssh服务
/etc/init.d/ssh restart
```