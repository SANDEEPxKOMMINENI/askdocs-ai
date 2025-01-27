from os import getenv
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Hugging Face configuration
HF_TOKEN = getenv("HUGGING_FACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")

# Use smaller, more efficient models for resource constraints
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller, faster model
QA_MODEL = "deepset/minilm-uncased-squad2"  # Lightweight QA model

# Device configuration
DEVICE = "cpu"  # Force CPU usage
torch.set_num_threads(4)  # Limit threads for 8GB RAM system

# Memory-optimized settings
MAX_BATCH_SIZE = 4  # Smaller batch size
CHUNK_SIZE = 384  # Reduced chunk size for memory efficiency

# Database configuration with performance settings
DATABASE_URL = "sqlite:///./askdocs.db"
SQLITE_PRAGMA = {
    "journal_mode": "WAL",  # Write-Ahead Logging for better concurrent performance
    "cache_size": -1000,    # Use up to 1MB of memory for caching
    "synchronous": "NORMAL" # Better performance while maintaining safety
}