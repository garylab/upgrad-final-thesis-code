import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# LLM and NLP
import openai
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Network analysis for cross-selling
import networkx as nx
from collections import defaultdict, Counter

# Model persistence
import joblib
import json

# Logging
from loguru import logger
import sys

import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logging
logger.add(
    "logs/recommendation_system.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="10 days"
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configuration
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
SAMPLE_FRACTION = 0.02  # Dataset sampling fraction for efficiency
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
RANDOM_STATE = 42

# File paths
FILES = {
    "aisles": os.path.join(DATASET_DIR, 'aisles.csv'),
    "departments": os.path.join(DATASET_DIR, 'departments.csv'),
    "orders": os.path.join(DATASET_DIR, 'orders.csv'),
    "order_products_prior": os.path.join(DATASET_DIR, 'order_products__prior.csv'),
    "order_products_train": os.path.join(DATASET_DIR, 'order_products__train.csv'),
    "products": os.path.join(DATASET_DIR, 'products.csv'),
    "estimate_prices": os.path.join(DATASET_DIR, 'products-with-llm-prices.csv'),
}

# Feature sets
BASE_FEATURES = [
    'order_hour_of_day', 'order_dow', 'days_since_prior_order', 'aisle_id', 
    'department_id', 'average_basket_size', 'purchase_frequency', 'product_reorder_rate',
    'price', 'user_avg_order_value', 'product_revenue_contribution', 'order_recency'
]

LLM_FEATURES = [
    'avg_semantic_similarity', 'max_semantic_similarity', 'semantic_diversity',
    'degree_centrality', 'betweenness_centrality', 'pagerank',
    'basket_expansion_score', 'cross_sell_strength', 'avg_basket_size_with_product'
]

# Model parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

# Hybrid scoring weights
HYBRID_WEIGHTS = {
    'alpha': 0.4,  # ML probability
    'beta': 0.3,   # Basket expansion
    'gamma': 0.2,  # Price/revenue
    'delta': 0.1   # Cross-sell strength
}

# Initialize OpenAI client
CLIENT = openai.OpenAI()

logger.info("Configuration loaded successfully")
logger.debug(f"Dataset directory: {DATASET_DIR}")
logger.debug(f"Sample fraction: {SAMPLE_FRACTION}")
logger.debug(f"Base features count: {len(BASE_FEATURES)}")
logger.debug(f"LLM features count: {len(LLM_FEATURES)}") 