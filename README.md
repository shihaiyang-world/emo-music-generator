# emo-music-generator


sudo mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 10.8.50.49:/g97mfuca/devops /devops-data

docker run -it --gpus all -d --name remote-gpu -p 2024:22 -v /devops-data:/workspace docker.shuiditech.com/devops/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.5-ssh-fluidsynth



```

在docker中确定ssh服务已经启动
```shell
# 查看状态
/etc/init.d/ssh status
# 重启ssh服务
/etc/init.d/ssh restart
```