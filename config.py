# config.py
# Environment setup, API client, and Ray initialization

import os
import logging
from dotenv import load_dotenv
import ray
from xai_sdk import Client

# Suppress Ray's warnings before initializing
logging.getLogger("ray").setLevel(logging.ERROR)

load_dotenv()

# Initialize Ray at startup with minimal logging
ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,
    include_dashboard=False,
    _metrics_export_port=None,
)

# API key and client
api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)
