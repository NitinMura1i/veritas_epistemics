# config.py
# Environment setup, API client, and Ray initialization

import os
import logging
import warnings
from dotenv import load_dotenv

from xai_sdk import Client

load_dotenv()

# Try to import and initialize Ray (optional - for synthetic data parallelization)
RAY_AVAILABLE = False
try:
    # Suppress Ray warnings before importing
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    os.environ["GLOG_minloglevel"] = "3"
    warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
    logging.getLogger("ray").setLevel(logging.CRITICAL)
    logging.getLogger("ray.rpc").setLevel(logging.CRITICAL)

    import ray
    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.CRITICAL,
        include_dashboard=False,
        _metrics_export_port=None,
        configure_logging=False,
        log_to_driver=False,
    )
    RAY_AVAILABLE = True
except ImportError:
    # Ray not installed - synthetic data will run sequentially
    RAY_AVAILABLE = False

# API key and client
api_key = os.getenv("XAI_API_KEY")
if api_key is None:
    raise ValueError("XAI_API_KEY not found in .env file!")

client = Client(api_key=api_key, timeout=3600)
