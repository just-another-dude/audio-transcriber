#!/bin/bash
# Stop Audio Transcriber

if pgrep -f "python.*app.py" > /dev/null; then
    pkill -f "python.*app.py"
    echo "Audio Transcriber stopped."
else
    echo "Audio Transcriber is not running."
fi
