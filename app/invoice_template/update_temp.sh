#!/bin/bash
pkill -f uvicorn
/home/attnroot/anaconda3/envs/invoiceocr/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8008 --log-level warning
