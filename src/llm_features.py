from .config import *

def generate_semantic_embeddings(products_df):
    """Generate semantic embeddings for products using sentence transformers"""
    logger.info("Generating semantic embeddings for products...")
    
    # Initialize sentence transformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.debug(f"Using embedding model: {EMBEDDING_MODEL}")
    
    # Create combined text for richer semantic understanding
    products_df['combined_text'] = (products_df['product_name'] + ' ' + 
                                   products_df['aisle'] + ' ' + 
                                   products_df['department'])
    
    # Generate embeddings (using subset for efficiency)
    unique_products = products_df[['product_id', 'combined_text']].drop_duplicates()
    logger.info(f"Generating embeddings for {len(unique_products):,} unique products...")
    
    embeddings = model.encode(unique_products['combined_text'].tolist(), 
                             show_progress_bar=True, 
                             batch_size=32)
    
    # Create embedding dataframe
    embedding_df = pd.DataFrame(embeddings)
    embedding_df['product_id'] = unique_products['product_id'].values
    embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    embedding_df.columns = embedding_cols + ['product_id']
    
    logger.success(f"Generated {embeddings.shape[1]}-dimensional embeddings successfully")
    logger.debug(f"Embedding DataFrame shape: {embedding_df.shape}")
    return embedding_df, embedding_cols

def generate_product_categories(product_names_batch):
    """Generate semantic categories for products using LLM"""
    prompt = f"""
    As a grocery retail expert, categorize these products into semantic categories that represent shopping behavior.
    Focus on how customers typically shop for these items (e.g., "healthy_snacks", "meal_prep_essentials", "comfort_food", "organic_produce").
    Return ONLY a comma-separated list of categories, one per product:

    Products:
    {"; ".join(product_names_batch)}
    """

    response = CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a grocery retail expert who understands customer shopping behavior."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    categories = [cat.strip() for cat in response.choices[0].message.content.strip().split(',')]
    return categories

def create_llm_categories(merged_df, products_df):
    """Generate LLM-based semantic categories for top products"""
    logger.info("Generating LLM-based semantic categories...")
    
    # Generate semantic categories for top products
    top_products = merged_df.groupby('product_id').size().head(200).index
    top_products_info = products_df[products_df['product_id'].isin(top_products)]
    
    logger.info(f"Processing {len(top_products_info)} top products for LLM categorization")
    logger.debug(f"Batch size: {BATCH_SIZE}")
    
    semantic_categories = {}
    total_batches = (len(top_products_info) - 1) // BATCH_SIZE + 1
    
    for i in range(0, len(top_products_info), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = top_products_info.iloc[i:i+BATCH_SIZE]
        
        try:
            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            categories = generate_product_categories(batch['product_name'].tolist())
            
            for j, (_, row) in enumerate(batch.iterrows()):
                if j < len(categories):
                    semantic_categories[row['product_id']] = categories[j]
            
            logger.info(f"Completed batch {batch_num}/{total_batches}")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            continue
    
    # Create semantic category dataframe
    semantic_cat_df = pd.DataFrame(list(semantic_categories.items()), 
                                  columns=['product_id', 'llm_category'])
    logger.success(f"Generated semantic categories for {len(semantic_cat_df)} products")
    logger.debug(f"Sample categories: {semantic_cat_df['llm_category'].head().tolist()}")
    return semantic_cat_df

def calculate_semantic_similarity_features(merged_df, embedding_df, embedding_cols):
    """Calculate semantic similarity features between frequently bought together products"""
    logger.info("Calculating semantic similarity features...")
    
    # Find frequently bought together products
    logger.debug("Finding frequently bought together products...")
    basket_pairs = []
    sample_orders = merged_df['order_id'].unique()[:10000]  # Sample for efficiency
    
    for order_id in sample_orders:
        products_in_order = merged_df[merged_df['order_id'] == order_id]['product_id'].tolist()
        if len(products_in_order) > 1:
            for i in range(len(products_in_order)):
                for j in range(i+1, len(products_in_order)):
                    basket_pairs.append((products_in_order[i], products_in_order[j]))
    
    # Count co-occurrence
    pair_counts = Counter(basket_pairs)
    frequent_pairs = {pair: count for pair, count in pair_counts.items() if count >= 5}
    
    logger.info(f"Found {len(frequent_pairs):,} frequent product pairs")
    logger.debug(f"Processed {len(sample_orders):,} orders for co-occurrence analysis")
    
    # Calculate average semantic similarity for each product
    logger.debug("Calculating semantic similarity metrics...")
    product_similarities = {}
    embedding_dict = dict(zip(embedding_df['product_id'], embedding_df[embedding_cols].values))
    
    for product_id in embedding_dict.keys():
        similarities = []
        for (p1, p2), count in frequent_pairs.items():
            if p1 == product_id and p2 in embedding_dict:
                sim = cosine_similarity([embedding_dict[p1]], [embedding_dict[p2]])[0][0]
                similarities.extend([sim] * count)
            elif p2 == product_id and p1 in embedding_dict:
                sim = cosine_similarity([embedding_dict[p1]], [embedding_dict[p2]])[0][0]
                similarities.extend([sim] * count)
        
        if similarities:
            product_similarities[product_id] = {
                'avg_semantic_similarity': np.mean(similarities),
                'max_semantic_similarity': np.max(similarities),
                'semantic_diversity': np.std(similarities)
            }
    
    similarity_df = pd.DataFrame.from_dict(product_similarities, orient='index')
    similarity_df['product_id'] = similarity_df.index
    similarity_df = similarity_df.reset_index(drop=True)
    
    logger.success(f"Calculated semantic similarity features for {len(similarity_df):,} products")
    logger.debug(f"Similarity DataFrame shape: {similarity_df.shape}")
    return similarity_df, frequent_pairs

def generate_llm_features(merged_df, products_df):
    """Main function to generate all LLM-enhanced features"""
    logger.info("=== Starting LLM Feature Generation ===")
    
    # Generate semantic embeddings
    embedding_df, embedding_cols = generate_semantic_embeddings(products_df)
    
    # Generate LLM categories
    semantic_cat_df = create_llm_categories(merged_df, products_df)
    
    # Calculate semantic similarity features
    similarity_df, frequent_pairs = calculate_semantic_similarity_features(merged_df, embedding_df, embedding_cols)
    
    logger.success("=== LLM Feature Generation Completed ===")
    
    return {
        'embedding_df': embedding_df,
        'embedding_cols': embedding_cols,
        'semantic_cat_df': semantic_cat_df,
        'similarity_df': similarity_df,
        'frequent_pairs': frequent_pairs
    } 