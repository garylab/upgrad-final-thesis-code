from .config import *

def load_datasets():
    """Load all datasets from CSV files"""
    logger.info("Loading datasets...")
    
    datasets = {}
    datasets['aisles'] = pd.read_csv(FILES['aisles'])
    datasets['departments'] = pd.read_csv(FILES['departments'])
    datasets['orders'] = pd.read_csv(FILES['orders'])
    datasets['order_products_prior'] = pd.read_csv(FILES['order_products_prior'])
    datasets['order_products_train'] = pd.read_csv(FILES['order_products_train'])
    datasets['products'] = pd.read_csv(FILES['products'])
    
    logger.success("Base datasets loaded successfully")
    logger.debug(f"Loaded {len(datasets)} datasets")
    return datasets

def merge_product_prices(products_df):
    """Merge product prices from LLM-estimated prices file"""
    logger.info("Loading product prices...")
    products_with_prices = pd.read_csv(FILES['estimate_prices'])
    products_df = pd.merge(products_df, products_with_prices[['product_id', 'price']], 
                          on='product_id', how='left')
    logger.success("Product prices merged successfully")
    logger.debug(f"Products with prices: {len(products_df)}")
    return products_df

def merge_all_datasets(datasets):
    """Merge all datasets into a comprehensive dataframe"""
    logger.info("Merging datasets...")
    
    # Merge products with aisles and departments
    products_df = datasets['products'].copy()
    products_df = pd.merge(products_df, datasets['aisles'], on='aisle_id')
    products_df = pd.merge(products_df, datasets['departments'], on='department_id')
    
    # Merge product prices
    products_df = merge_product_prices(products_df)
    
    # Combine order products
    order_products_all_df = pd.concat([
        datasets['order_products_prior'], 
        datasets['order_products_train']
    ])
    
    # Merge orders with products
    merged_df = pd.merge(order_products_all_df, datasets['orders'], on='order_id', how='inner')
    merged_df = pd.merge(merged_df, products_df, on='product_id', how='inner')
    
    logger.success(f"Dataset merged successfully. Shape: {merged_df.shape}")
    logger.debug(f"Products DataFrame shape: {products_df.shape}")
    return merged_df, products_df

def handle_missing_values(merged_df):
    """Handle missing values in the merged dataset"""
    logger.info("Handling missing values...")
    
    merged_df = merged_df.copy()
    days_prior_missing = merged_df['days_since_prior_order'].isna().sum()
    price_missing = merged_df['price'].isna().sum()
    
    logger.debug(f"Missing days_since_prior_order: {days_prior_missing}")
    logger.debug(f"Missing prices: {price_missing}")
    
    merged_df['days_since_prior_order'] = merged_df['days_since_prior_order'].fillna(0)
    merged_df['price'] = merged_df['price'].fillna(merged_df['price'].median())
    
    logger.success("Missing values handled successfully")
    return merged_df

def create_base_features(merged_df):
    """Create base features for the recommendation system"""
    logger.info("Creating base features...")
    
    # User-level features
    merged_df['average_basket_size'] = merged_df.groupby('user_id')['product_id'].transform('count')
    merged_df['purchase_frequency'] = merged_df.groupby('user_id')['order_number'].transform('max')
    merged_df['user_avg_order_value'] = merged_df.groupby('user_id')['price'].transform('mean')
    merged_df['user_total_spent'] = merged_df.groupby('user_id')['price'].transform('sum')
    
    # Product-level features
    merged_df['product_reorder_rate'] = merged_df.groupby('product_id')['reordered'].transform('mean')
    merged_df['product_revenue_contribution'] = merged_df.groupby('product_id')['price'].transform('sum')
    
    # Time-based features
    merged_df['order_recency'] = merged_df.groupby('user_id')['order_number'].transform('max') - merged_df['order_number']
    merged_df['product_last_ordered'] = merged_df.groupby(['user_id', 'product_id'])['order_number'].transform('max')
    
    logger.success("Base features created successfully")
    logger.debug(f"Total features in dataset: {len(merged_df.columns)}")
    return merged_df

def sample_dataset(merged_df, sample_fraction=None):
    """Sample dataset for efficiency during development"""
    if sample_fraction is None:
        sample_fraction = SAMPLE_FRACTION
    
    logger.info(f"Sampling dataset with fraction: {sample_fraction}")
    original_size = len(merged_df)
    sampled_df = merged_df.sample(frac=sample_fraction, random_state=RANDOM_STATE)
    sampled_size = len(sampled_df)
    
    logger.success(f"Dataset sampled: {original_size:,} → {sampled_size:,} rows")
    logger.debug(f"Sampled dataset shape: {sampled_df.shape}")
    return sampled_df

def preprocess_data():
    """Main preprocessing pipeline"""
    logger.info("=== Starting Data Preprocessing Pipeline ===")
    
    # Load datasets
    datasets = load_datasets()
    
    # Merge all datasets
    merged_df, products_df = merge_all_datasets(datasets)
    
    # Handle missing values
    merged_df = handle_missing_values(merged_df)
    
    # Create base features
    merged_df = create_base_features(merged_df)
    
    # Sample dataset for efficiency
    sampled_df = sample_dataset(merged_df)
    
    logger.success("=== Data Preprocessing Pipeline Completed ===")
    return sampled_df, merged_df, products_df 