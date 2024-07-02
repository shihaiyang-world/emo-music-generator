FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.5-ssh-fluidsynth
LABEL maintainer="shihaiyang@shuidi-inc.com"

COPY ./requirements.txt /var/www/
RUN if test -e /var/www/requirements.txt; then pip install --no-cache-dir -r /var/www/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; fi

ENTRYPOINT ["python3" "webui.py"]