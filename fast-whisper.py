import whisperx
import os
import time
import torch
import torchaudio
import numpy as np
# from noise_suppress import NoiseSuppress  # 如果需要降噪
from faster_whisper import WhisperModel
import subprocess

def audio_to_srt(
        audio_path: str,
        faster_whisper_path: str,
        whisper_model: str,
        model_dir: str,
        language: str = "zh",
        device: str = "cpu",
        output_dir: str = None,
        output_format: str = "srt",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        # VAD 相关参数
        vad_filter: bool = True,
        vad_threshold: float = 0.4,
        vad_method: str = "",  # https://github.com/Purfview/whisper-standalone-win/discussions/231
        # 音频处理
        ff_mdx_kim2: bool = False,
        # 文本处理参数
        one_word: int = 0,
        sentence: bool = False,
        max_line_width: int = 100,
        max_line_count: int = 1,
        max_comma: int = 20,
        max_comma_cent: int = 50,
        prompt: str = None,
        resample_rate=16000, # 设置重采样率
        return_char_alignments=False,
        split_by_word=True,
        remove_silences=False
):
    """
    使用 fast-whisper 将音频文件转换为 SRT 字幕。

    Args:
        audio_path: 音频文件的路径。
        faster_whisper_path: faster-whisper 模型文件的路径。
        whisper_model: whisper 模型大小(tiny, base, small, medium, large-v1, large-v2)。
        model_dir: 模型下载目录。
        language: 指定转录语言，例如："zh"。
        device: 指定运行设备，"cpu" 或 "cuda"。
        output_dir: 输出目录，默认为当前目录。
        output_format: 输出文件格式，目前只支持 "srt"。
        use_cache: 是否使用缓存。
        need_word_time_stamp: 是否需要单词时间戳。
        vad_filter: 是否使用 VAD 过滤静音部分。
        vad_threshold: VAD 阈值。
        vad_method: VAD 方法，目前未使用。
        ff_mdx_kim2: 是否使用 ff-mdx 降噪。
        one_word:  文本处理参数，用于限制单行文本的单词数量。
        sentence: 文本处理参数， 是否按句子划分
        max_line_width: 文本处理参数，最大行宽度。
        max_line_count: 文本处理参数，最大行数。
        max_comma: 文本处理参数，最大逗号数。
        max_comma_cent: 文本处理参数，最大逗号中心。
        prompt: 初始提示文本。
        resample_rate: 重采样的目标采样率。
        return_char_alignments: 是否返回字符级对齐。
        split_by_word: 是否按单词分割输出
        remove_silences: 是否移除静音片段
    """
    print(f"音频文件路径: {audio_path}")
    print(f"faster-whisper路径: {faster_whisper_path}")
    print(f"whisper模型大小: {whisper_model}")
    print(f"模型下载目录: {model_dir}")
    print(f"语言: {language}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"输出格式: {output_format}")
    print(f"使用缓存: {use_cache}")
    print(f"需要单词时间戳: {need_word_time_stamp}")
    print(f"VAD 过滤: {vad_filter}")
    print(f"VAD 阈值: {vad_threshold}")
    print(f"VAD 方法: {vad_method}")
    print(f"ff-mdx 降噪: {ff_mdx_kim2}")
    print(f"单行单词数: {one_word}")
    print(f"按句子划分: {sentence}")
    print(f"最大行宽度: {max_line_width}")
    print(f"最大行数: {max_line_count}")
    print(f"最大逗号数: {max_comma}")
    print(f"最大逗号中心: {max_comma_cent}")
    print(f"初始提示: {prompt}")
    print(f"重采样率: {resample_rate}")
    print(f"返回字符级对齐: {return_char_alignments}")
    print(f"按单词分割输出：{split_by_word}")
    print(f"移除静音片段: {remove_silences}")

    if not os.path.exists(audio_path):
        print(f"错误：音频文件 '{audio_path}' 不存在。")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not output_dir:
        output_dir = "."  # 如果没有指定输出目录，默认为当前目录

    try:
        # 加载 faster-whisper 模型
        model = WhisperModel(
            faster_whisper_path,
            device=device,
            compute_type="int8",
            download_root=model_dir
        )

        start_time = time.time()  # 记录开始时间

        # 加载音频并进行预处理
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 1. 重采样
        if sample_rate != resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
            waveform = resampler(waveform)
            sample_rate = resample_rate

        # 2. 音频归一化
        waveform = waveform / torch.max(torch.abs(waveform))
        audio = waveform.squeeze().numpy()


        segments, info = model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            vad_filter=vad_filter,
            vad_parameters=dict(threshold=vad_threshold),
        )

        result = {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                         {
                             "start": word.start,
                             "end": word.end,
                             "word": word.word
                         } for word in segment.words
                    ] if need_word_time_stamp else [],
                }
                for segment in segments
            ],
            "language": info.language,
        }

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        print(f"转录完成! 用时: {elapsed_time:.2f} 秒")
        
        model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device) # 加载对齐模型
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=return_char_alignments,
                                split_by_word=split_by_word, remove_silences=remove_silences) # 进行对齐
        
        # # 文本处理部分
        # 处理文本的逻辑, 这里是一个示例，你可以根据你的需求进行修改。
        # def format_text(segments):
        #     formatted_segments = []
        #     for segment in segments:
        #          text = segment["text"]
        #          formatted_text = text
        #
        #          if one_word > 0:
        #              words = text.split()
        #              formatted_text = " ".join(words[:one_word])
        #
        #          formatted_segments.append({
        #              "start": segment["start"],
        #              "end": segment["end"],
        #              "text": formatted_text
        #          })
        #     return formatted_segments
        #
        # result["segments"] = format_text(result["segments"])
        
        output_file_name = os.path.splitext(os.path.basename(audio_path))[0] + "." + output_format # 生成输出文件名
        output_file_path = os.path.join(output_dir, output_file_name) # 生成输出文件路径

        whisperx.utils.write_srt(result["segments"], output_file_path) # 写入SRT字幕文件

        print(f"字幕文件保存到: {output_file_path}")

    except Exception as e:
        print(f"转录过程中出现错误: {e}")
    
if __name__ == "__main__":
    audio_file = "audio.mp3"  #  替换成你的音频文件路径
    faster_whisper_path = "large-v3" # 这里实际上是模型目录，具体可以看 faster-whisper库的文档
    whisper_model = "large-v2" # whisper模型的大小
    model_dir = "./models" # 模型下载目录
    output_directory = "output_srt"  # 输出目录，可选
    language = "zh"  # 转录语言
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 根据环境选择设备
    output_format = "srt"
    use_cache = False
    need_word_time_stamp = True
    vad_filter = True
    vad_threshold = 0.4
    vad_method = ""
    ff_mdx_kim2 = False
    one_word = 0
    sentence = False
    max_line_width = 100
    max_line_count = 1
    max_comma = 20
    max_comma_cent = 50
    prompt = None
    resample_rate=16000
    return_char_alignments=False
    split_by_word=True
    remove_silences=False

    audio_to_srt(
        audio_file,
        faster_whisper_path,
        whisper_model,
        model_dir,
        language,
        device,
        output_directory,
        output_format,
        use_cache,
        need_word_time_stamp,
        vad_filter,
        vad_threshold,
        vad_method,
        ff_mdx_kim2,
        one_word,
        sentence,
        max_line_width,
        max_line_count,
        max_comma,
        max_comma_cent,
        prompt,
        resample_rate,
        return_char_alignments,
        split_by_word,
        remove_silences
    )
