import os
import subprocess

files=os.listdir(r"C:\Users\ayush\OneDrive\rag-based ai project\sample-videos")
for file in files:
    if not file.endswith(".webm"):
        continue

    tutorial_number=file.split(" [")[0].split(" #")[1]
    file_name=file.split(" ｜ ")[0]
    print(tutorial_number,file_name)
    subprocess.run(["ffmpeg","-i",f"sample_videos/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])
