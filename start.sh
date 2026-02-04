#!/bin/bash
# Audio Transcriber - Easy Start Script

cd "$(dirname "$0")"

# Check if already running
if pgrep -f "python3.*app.py" > /dev/null; then
    echo "App is already running!"
    echo "Access at: http://localhost:7860"
    echo ""
    echo "To stop: ./stop.sh"
    exit 0
fi

# Clear old logs
> /tmp/transcriber.log

echo "Starting Audio Transcriber..."
python3 app.py --host 127.0.0.1 &

# Wait for startup
sleep 3

if pgrep -f "python3.*app.py" > /dev/null; then
    echo ""
    echo "Audio Transcriber is running!"
    echo "Access at: http://localhost:7860"
    echo ""
    echo "Logs: tail -f /tmp/transcriber.log"
    echo "Stop: ./stop.sh"
else
    echo "Failed to start. Check logs: cat /tmp/transcriber.log"
fi
