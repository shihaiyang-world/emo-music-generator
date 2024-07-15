import gradio as gr
from tools.i18n.i18n import I18nAuto
import re
import os
import torch
import librosa
import pandas as pd
import tempfile
from datetime import datetime

i18n = I18nAuto(language="zh_CN")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


gpt_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
sovits_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

def get_weights_names():
    return ["EfficCaps+LSTM"], ["Transformer+emo"]

SoVITS_names, GPT_names = get_weights_names()

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}

def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")

def eeg_inference(inputs):
    # 读取 CSV 文件
    df = pd.read_csv("eeg-result.csv")

    # 可以在这里对 DataFrame 进行任何处理，例如筛选、转换等
    # 示例：只提取前两列并返回
    df_processed = df.iloc[:, :2]

    # 将处理后的 DataFrame 保存为新的 CSV 文件
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_processed.to_csv(tmp_file.name, index=False)

    # 指定下载文件的文件名
    download_file_name = "processed_file.csv"

    # return (tmp_file.name, download_file_name)
    return tmp_file.name


def emo_music_inference():
    # inner_function = current_time()
    # inner_function()

    audio, sr = librosa.load(path="gen_Q1_3.mp3")
    return sr,audio


# 定义一个函数，返回当前的日期和时间。
def current_time():
    def inner():
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return f"欢迎使用,当前时间是: {current_time}"

    return inner


with gr.Blocks(title="GPT-SoVITS WebUI") as app:

    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )

    with gr.Group():
        gr.Markdown(value=i18n("脑电信号情感识别"))
        with gr.Row():
            inputs = gr.components.File(label="上传脑电文件",file_types=['.csv', '.mat'],height=100)

            GPT_dropdown = gr.Dropdown(label=i18n("EEG情感模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("音乐生成模型列表"),
                                          choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path,
                                              interactive=True)

            music_infer_btn = gr.Button(i18n("生成情感音乐"), variant="primary")

    with gr.Group():
        gr.Markdown(value=i18n("情感识别与音乐生成"))

        out_1 = gr.Textbox(label="实时状态",
                           value=current_time(),
                           every=1,
                           info="当前时间", )



        output = gr.Audio(label=i18n("生成的音乐"))

        music_infer_btn.click(fn=emo_music_inference, inputs=[], outputs=[output])
            # SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            # GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])








app.launch(
    server_name="0.0.0.0",
    share=False,
    server_port=7860
)

