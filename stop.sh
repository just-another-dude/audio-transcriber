#!/bin/bash
# Stop Audio Transcriber

if pgrep -f "python3.*app.py" > /dev/null; then
    pkill -f "python3.*app.py"
    echo "Audio Transcriber stopped."
else
    echo "Audio Transcriber is not running."
fi
