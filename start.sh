#!/bin/bash
# Pitch Visualizer — start Flask + ngrok with one command

cd "$(dirname "$0")"

# Kill any existing instances
pkill -f "python3 app.py" 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null
pkill -f "ngrok" 2>/dev/null
sleep 1

TOKEN=$(cat .secret_token 2>/dev/null || echo "")

echo ""
echo "Starting Pitch Visualizer..."
echo ""

# Start Flask in background
python3 app.py &
FLASK_PID=$!
sleep 3

# Start ngrok tunnel in background
ngrok http 5001 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!
sleep 3

# Get the public URL from ngrok API
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    url = d['tunnels'][0]['public_url']
    print(url)
except:
    print('')
" 2>/dev/null)

echo "============================================================"
echo "  Pitch Visualizer — Share this link:"
echo ""
if [ -n "$PUBLIC_URL" ] && [ -n "$TOKEN" ]; then
    echo "  $PUBLIC_URL/view/$TOKEN"
else
    echo "  Could not get ngrok URL — check http://localhost:4040"
    echo "  Local: http://localhost:5001/view/$TOKEN"
fi
echo ""
echo "  (link stays active while this terminal is open)"
echo "============================================================"
echo ""

# Keep script alive — Ctrl+C to stop everything
trap "kill $FLASK_PID $NGROK_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait $FLASK_PID
