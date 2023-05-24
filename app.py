import re
import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, ns, nsw, speed, is_symbol):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, is_symbol)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=ns, noise_scale_w=nsw,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_text):
        return (_clean_text(input_text, hps.data.text_cleaners), input_text) if is_symbol_input \
            else (temp_text, temp_text)

    return to_symbol_fn


models_info = [
    {
        "languages": ['日本語', '简体中文', 'English', 'Mix'],
        "description": """
                       这个模型包含赛马娘的116名角色，能合成中日英三语。\n\n
                       若需要在同一个句子中混合多种语言，使用相应的语言标记包裹句子。 （日语用[JA], 中文用[ZH], 英文用[EN]），参考Examples中的示例。
                       """,
        "model_path": "./models/G_15800.pth",
        "config_path": "./configs/modified_finetune_speaker.json",
        "examples": [['私、必ず強くなりますっ。', '特别周', '日本語', 1, False],
                     ['私も自信を持ってこの走りを貫けます。', '无声铃鹿', '日本語', 1, False],
                     ['无论做什么事情都要全力以赴！', '大和赤骥', '简体中文', 1, False],
                     ['Can you tell me how much the shirt is?',
                         '目白麦昆', 'English', 1, False],
                     ['[EN]Excuse me?[EN][JA]お帰りなさい，お兄様！[JA]', '草上飞', 'Mix', 1, False]],
    }
]

models_tts = []
models_vc = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")
    args = parser.parse_args()
    categories = ["Umamusume"]
    others = {
        "Princess Connect! Re:Dive": "https://huggingface.co/spaces/FrankZxShen/vits-fast-finetuning-pcr",
    }
    for info in models_info:
        lang = info['languages']
        examples = info['examples']
        config_path = info['config_path']
        model_path = info['model_path']
        description = info['description']
        hps = utils.get_hparams_from_file(config_path)

        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = net_g.eval()

        _ = utils.load_checkpoint(model_path, net_g, None)
        speaker_ids = hps.speakers
        speakers = list(hps.speakers.keys())
        models_tts.append((description, speakers, lang, examples,
                           hps.symbols, create_tts_fn(net_g, hps, speaker_ids),
                           create_to_symbol_fn(hps)))
        models_vc.append(
            (description, speakers, create_vc_fn(net_g, hps, speaker_ids)))

    app = gr.Blocks()
    with app:
        gr.Markdown(
            "# <center> vits-fast-fineturning-models-pcr\n"
            "## <center> Please do not generate content that could infringe upon the rights or cause harm to individuals or organizations.\n"
            "## <center> 请不要生成会对个人以及组织造成侵害的内容\n\n"
            "[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pn1xnFfdLK63gVXDwV4zCXfVeo8c-I-0?usp=sharing)\n\n"
            "[![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm-dark.svg)](https://huggingface.co/spaces/FrankZxShen/vits-fast-finetuning-pcr?duplicate=true)\n\n"
            "[![Finetune your own model](https://badgen.net/badge/icon/github?icon=github&label=Finetune%20your%20own%20model)](https://github.com/Plachtaa/VITS-fast-fine-tuning)"
        )
        gr.Markdown("# TTS&Voice Conversion for Princess Connect! Re:Dive\n\n"
                    )
        with gr.Tabs():
            for category in categories:
                with gr.TabItem(category):
                    with gr.Tab("TTS"):
                        for i, (description, speakers, lang, example, symbols, tts_fn, to_symbol_fn) in enumerate(
                                models_tts):
                            gr.Markdown(description)
                            with gr.Row():
                                with gr.Column():
                                    textbox = gr.TextArea(label="Text",
                                                          placeholder="Type your sentence here ",
                                                          value="新たなキャラを解放できるようになったようですね。", elem_id=f"tts-input")
                                    with gr.Accordion(label="Phoneme Input", open=False):
                                        temp_text_var = gr.Variable()
                                        symbol_input = gr.Checkbox(
                                            value=False, label="Symbol input")
                                        symbol_list = gr.Dataset(label="Symbol list", components=[textbox],
                                                                 samples=[[x]
                                                                          for x in symbols],
                                                                 elem_id=f"symbol-list")
                                        symbol_list_json = gr.Json(
                                            value=symbols, visible=False)
                                    symbol_input.change(to_symbol_fn,
                                                        [symbol_input, textbox,
                                                            temp_text_var],
                                                        [textbox, temp_text_var])
                                    symbol_list.click(None, [symbol_list, symbol_list_json], textbox,
                                                      _js=f"""
                                    (i, symbols, text) => {{
                                        let root = document.querySelector("body > gradio-app");
                                        if (root.shadowRoot != null)
                                            root = root.shadowRoot;
                                        let text_input = root.querySelector("#tts-input").querySelector("textarea");
                                        let startPos = text_input.selectionStart;
                                        let endPos = text_input.selectionEnd;
                                        let oldTxt = text_input.value;
                                        let result = oldTxt.substring(0, startPos) + symbols[i] + oldTxt.substring(endPos);
                                        text_input.value = result;
                                        let x = window.scrollX, y = window.scrollY;
                                        text_input.focus();
                                        text_input.selectionStart = startPos + symbols[i].length;
                                        text_input.selectionEnd = startPos + symbols[i].length;
                                        text_input.blur();
                                        window.scrollTo(x, y);
                                        text = text_input.value;
                                        return text;
                                    }}""")
                                    # select character
                                    char_dropdown = gr.Dropdown(
                                        choices=speakers, value=speakers[0], label='character')
                                    language_dropdown = gr.Dropdown(
                                        choices=lang, value=lang[0], label='language')
                                    ns = gr.Slider(
                                        label="noise_scale", minimum=0.1, maximum=1.0, step=0.1, value=0.6, interactive=True)
                                    nsw = gr.Slider(label="noise_scale_w", minimum=0.1,
                                                    maximum=1.0, step=0.1, value=0.668, interactive=True)
                                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                                label='速度 Speed')
                                with gr.Column():
                                    text_output = gr.Textbox(label="Message")
                                    audio_output = gr.Audio(
                                        label="Output Audio", elem_id="tts-audio")
                                    btn = gr.Button("Generate!")
                                    btn.click(tts_fn,
                                              inputs=[textbox, char_dropdown, language_dropdown, ns, nsw, duration_slider,
                                                      symbol_input],
                                              outputs=[text_output, audio_output])
                            gr.Examples(
                                examples=example,
                                inputs=[textbox, char_dropdown, language_dropdown,
                                        duration_slider, symbol_input],
                                outputs=[text_output, audio_output],
                                fn=tts_fn
                            )
                    with gr.Tab("Voice Conversion"):
                        for i, (description, speakers, vc_fn) in enumerate(
                                models_vc):
                            gr.Markdown("""
                                            录制或上传声音，并选择要转换的音色。
                            """)
                            with gr.Column():
                                record_audio = gr.Audio(
                                    label="record your voice", source="microphone")
                                upload_audio = gr.Audio(
                                    label="or upload audio here", source="upload")
                                source_speaker = gr.Dropdown(
                                    choices=speakers, value=speakers[0], label="source speaker")
                                target_speaker = gr.Dropdown(
                                    choices=speakers, value=speakers[0], label="target speaker")
                            with gr.Column():
                                message_box = gr.Textbox(label="Message")
                                converted_audio = gr.Audio(
                                    label='converted audio')
                            btn = gr.Button("Convert!")
                            btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
                                      outputs=[message_box, converted_audio])
            for category, link in others.items():
                with gr.TabItem(category):
                    gr.Markdown(
                        f'''
                        <center>
                          <h2>Click to Go</h2>
                          <a href="{link}">
                            <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl-dark.svg"
                          </a>
                        </center>
                        '''
                    )

    app.queue(concurrency_count=3).launch(show_api=False, share=args.share)
