#!/bin/sh

for filename in /home/dave/OSR4Rights/AudioTools/input/*.mp3; do
        # if no match on the above glob don't run ffmpeg 
        [ -e "$filename" ] || continue

        ffmpeg -i "$filename" "${filename%.*}.wav"
done

# need to do flac, mp4, ogg, flac, m4a