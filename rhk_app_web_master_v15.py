# -*- coding: utf-8 -*-
"""
RHK Befundassistent – Web (FastAPI + Gradio mount)

Run locally:
  python rhk_app_web_master_v15.py

Render start command (recommended):
  python rhk_app_web_master_v15.py
or:
  uvicorn rhk_app_web_master_v15:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import gradio as gr
from fastapi import FastAPI
import uvicorn

from rhk.ui import build_demo
from rhk.version import APP_NAME, APP_VERSION


demo = build_demo()
# enable queue for stability in production (safe default)
try:
    demo.queue(concurrency_count=2, max_size=32)
except Exception:
    # Older gradio versions
    try:
        demo.queue()
    except Exception:
        pass

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }


# Mount gradio at root
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
