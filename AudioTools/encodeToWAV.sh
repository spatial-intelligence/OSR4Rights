#!/bin/sh

# -y for Yes to automatically overwrite

# Audio
for filename in /home/dave/OSR4Rights/AudioTools/input/*.mp3; do
        # if no match on the above glob don't run ffmpeg 
        [ -e "$filename" ] || continue

        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done

for filename in /home/dave/OSR4Rights/AudioTools/input/*.ogg; do
        [ -e "$filename" ] || continue
        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done

for filename in /home/dave/OSR4Rights/AudioTools/input/*.flac; do
        [ -e "$filename" ] || continue
        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done

for filename in /home/dave/OSR4Rights/AudioTools/input/*.m4a; do
        [ -e "$filename" ] || continue
        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done


# Video

# sometimes youtube clips can be in this video format
for filename in /home/dave/OSR4Rights/AudioTools/input/*.webm; do
        [ -e "$filename" ] || continue
        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done


for filename in /home/dave/OSR4Rights/AudioTools/input/*.mp4; do
        [ -e "$filename" ] || continue
        ffmpeg -y -i "$filename" "${filename%.*}.wav"
done
