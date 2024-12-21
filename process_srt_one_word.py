import json
import re

def format_time(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt_one_word(segments, output_file):
    """将 Whisper 的 segments 列表转换为 SRT 字幕文件，每行一个词"""
    srt_lines = []
    for i, segment in enumerate(segments):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()
        words = text.split()
       
        for j, word in enumerate(words):
            srt_lines.append(str(i + 1))
            if j == 0:
                srt_lines.append(f"{start_time} --> {end_time}")
            else:
                 srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(word)
            srt_lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

if __name__ == "__main__":
    input_json_file = "output/audio.json" # 替换为你的json文件路径
    output_srt_file = "output/output.srt"  # 替换为你想要保存的srt文件路径

    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_srt_one_word(data["segments"], output_srt_file)
    print(f"已生成每行一个词的 SRT 字幕文件: {output_srt_file}")
