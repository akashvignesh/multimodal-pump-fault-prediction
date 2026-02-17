#!/bin/bash
# Start both API and Streamlit in the same container

# Start the FastAPI server in the background
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# Start Streamlit in the foreground
exec streamlit run app/Home.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
