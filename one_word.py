import json
import re

def format_time(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt_one_word_timestamp(segments, output_file):
    """将 Whisper 的 segments 列表转换为 SRT 字幕文件，每词一行，且带有独立时间戳"""
    srt_lines = []
    index = 1
    for segment in segments:
      if 'words' not in segment:
         continue # 如果这个segment没有words信息，跳过。
      for word_info in segment['words']:
        start_time = format_time(word_info["start"])
        end_time = format_time(word_info["end"])
        word = word_info["word"].strip()
            
        srt_lines.append(str(index))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(word)
        srt_lines.append("")
        index += 1

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

if __name__ == "__main__":
    input_json_file = "output/audio.json" # 替换为你的JSON文件路径
    output_srt_file = "output/output.srt"  # 替换为你想要保存的 SRT 文件路径

    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_srt_one_word_timestamp(data["segments"], output_srt_file)
    print(f"已生成每行一个单词的 SRT 字幕文件: {output_srt_file}")
