import json
import re

def split_text(text, words_per_line):
    """将文本分割为最多 words_per_line 个单词的行。"""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(current_line) == words_per_line:
            lines.append(" ".join(current_line))
            current_line = []
    if current_line:
         lines.append(" ".join(current_line))
    return lines
 
def format_time(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(segments, words_per_line, output_file):
    """将 Whisper 的 segments 列表转换为 SRT 字幕文件，限制每行单词数"""
    srt_lines = []
    for i, segment in enumerate(segments):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()
        lines = split_text(text, words_per_line)
       
        for j, line in enumerate(lines):
             srt_lines.append(str(i + 1))
             if j == 0:
                 srt_lines.append(f"{start_time} --> {end_time}")
             else:
                 srt_lines.append(f"{start_time} --> {end_time}")
             srt_lines.append(line)
             srt_lines.append("")
          
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

if __name__ == "__main__":
    words_per_line = 6  # 设置每行最大单词数
    input_json_file = "output/audio.json" # 替换为你的 JSON 文件路径
    output_srt_file = "output/output.srt"  # 替换为你想要保存的 SRT 文件路径

    # 先使用 whisper 生成 json 文件
    # whisper audio.mp3 --model base --output_format json --output_dir output
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_srt(data["segments"], words_per_line, output_srt_file)
    print(f"已生成每行最多 {words_per_line} 个单词的 SRT 字幕文件: {output_srt_file}")
