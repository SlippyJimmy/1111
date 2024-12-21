import whisper
import os
import re
import torch
from whisper.utils import get_writer
from tqdm import tqdm

def load_whisper_model(model_name="base", model_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """加载 Whisper 模型."""
    try:
        model = whisper.load_model(model_name, download_root=model_dir).to(device)
        print(f"成功加载 Whisper 模型: {model_name} on {device}")
        return model
    except Exception as e:
        print(f"加载 Whisper 模型失败: {e}")
        return None

def format_time(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(segments, output_file, max_words_per_line=None):
    """将 Whisper 的 segments 列表转换为 SRT 字幕文件，支持 max_words_per_line."""
    srt_lines = []
    index = 1
    for segment in segments:
      if 'words' not in segment:
        continue # 如果此segment没有words跳过
      words_info = segment["words"]
      
      if max_words_per_line is None: # 不限制每行的单词数
          srt_lines.append(str(index))
          start_time = format_time(words_info[0]['start'])
          end_time = format_time(words_info[-1]['end'])
          text =  " ".join([word_info["word"] for word_info in words_info]).strip()
          srt_lines.append(f"{start_time} --> {end_time}")
          srt_lines.append(text)
          srt_lines.append("")
          index+=1
      else:
         
         current_line = []
         line_start_time = None
         for word_info in words_info:
            if not line_start_time:
              line_start_time = word_info["start"]
            current_line.append(word_info)

            if len(current_line) == max_words_per_line:
               start_time = format_time(line_start_time)
               end_time = format_time(current_line[-1]['end'])
               text = " ".join([word["word"] for word in current_line]).strip()
               srt_lines.append(str(index))
               srt_lines.append(f"{start_time} --> {end_time}")
               srt_lines.append(text)
               srt_lines.append("")
               index+=1
               current_line = []
               line_start_time=None
         if current_line:
            start_time = format_time(line_start_time)
            end_time = format_time(current_line[-1]['end'])
            text = " ".join([word["word"] for word in current_line]).strip()
            srt_lines.append(str(index))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")
            index+=1
    try:
      with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
        print(f"成功生成 SRT 字幕文件: {output_file}")
    except Exception as e:
      print(f"生成 SRT 字幕文件失败: {e}")
  

def transcribe_audio(
    audio_file="audio.mp3",
    model="turbo",
    output_dir=".",
    output_format="srt",
    word_timestamps=True,
    initial_prompt=None,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    max_words_per_line = None
    ):
    """转录音频文件并保存为 SRT 字幕文件."""
    try:
        if model is None:
            print("请先加载模型")
            return
        
        print(f"开始转录音频文件: {audio_file}")
        with tqdm(total=100, desc="转录进度") as pbar:
            result = model.transcribe(
                audio_file,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold
            )

        if result:
            # 提取音频文件名，作为输出文件名
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}.srt")
            if output_format =="srt":
               generate_srt(result["segments"], output_file, max_words_per_line)
            else:
                writer = get_writer(output_format, output_dir=output_dir)
                writer(result, f"{base_name}.{output_format}", max_words_per_line=max_words_per_line) # 这里传入参数
               
        else:
            print(f"转录音频文件失败: {audio_file}")
    
    except Exception as e:
        print(f"转录音频文件失败：{e}")

if __name__ == "__main__":
    audio_file = "audio.mp3"  # 替换为你的音频文件路径
    model_name = "turbo"  # 可选： "tiny", "small", "medium", "large"
    model_dir=None # 如果你需要指定模型路径，则在此修改
    output_dir = "output"  # 替换为你想要保存字幕的目录
    word_timestamps = True  # 是否需要单词级别的时间戳。
    initial_prompt=None # 初始提示
    no_speech_threshold = 0.6 # 静音阈值
    logprob_threshold = -1.0 # 转录失败阈值。
    max_words_per_line = 6 # 每行最多的单词数, 如果为 None 则不进行分行
    output_format = "srt" # or 'vtt', 'txt', 'json'
    
    
    model = load_whisper_model(model_name, model_dir)
    
    if model:
        transcribe_audio(
            audio_file,
            model,
            output_dir,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
           max_words_per_line=max_words_per_line,
            output_format=output_format
        )
