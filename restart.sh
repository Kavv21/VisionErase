#!/bin/bash
echo "Killing existing processes..."
pkill -9 -f "celery" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "npm run dev" 2>/dev/null || true
kill $(lsof -t -i:8000) 2>/dev/null || true
kill $(lsof -t -i:5173) 2>/dev/null || true
sleep 2

echo "Flushing Redis..."
redis-cli -p 6379 flushall

echo "Starting infrastructure..."
cd ~/visionerase
docker compose up -d redis postgres minio
sleep 3

source venv/bin/activate
set -a && source .env && source .env.local && set +a
export PYTHONPATH=/home/kavish/visionerase:/home/kavish/sam2:/home/kavish/XMem:/home/kavish/ProPainter:$PYTHONPATH

echo "All ready — now start manually in separate tabs:"
echo ""
echo "Tab 2 (API):      uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo "Tab 3 (Worker):   celery -A workers.celery_app worker --queues=segmentation,inpainting,stitching,quality --loglevel=info --concurrency=1"
echo "Tab 4 (Frontend): cd ~/visionerase/frontend && npm run dev"
