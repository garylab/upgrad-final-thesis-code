from .config import *

def build_cross_selling_network(frequent_pairs):
    """Build product co-occurrence network for cross-selling analysis"""
    logger.info("Building cross-selling network...")
    
    # Create product co-occurrence network
    G = nx.Graph()
    
    # Add edges with weights based on co-occurrence frequency
    for (p1, p2), weight in frequent_pairs.items():
        G.add_edge(p1, p2, weight=weight)
    
    logger.success(f"Network built successfully: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    logger.debug(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    return G

def calculate_centrality_measures(G):
    """Calculate network centrality measures for products"""
    logger.info("Calculating network centrality measures...")
    
    num_nodes = G.number_of_nodes()
    k = min(1000, num_nodes)
    
    logger.debug(f"Using k={k} for betweenness centrality calculation")
    
    # Calculate centrality measures
    logger.debug("Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    logger.debug("Computing betweenness centrality...")
    betweenness_centrality = nx.betweenness_centrality(G, k=k)  # Sample for efficiency
    
    logger.debug("Computing PageRank...")
    pagerank = nx.pagerank(G, max_iter=50)
    
    centrality_measures = {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'pagerank': pagerank
    }
    
    # Create centrality dataframe
    centrality_df = pd.DataFrame(centrality_measures)
    centrality_df['product_id'] = centrality_df.index
    centrality_df = centrality_df.reset_index(drop=True)
    
    logger.success("Network centrality measures calculated successfully")
    logger.debug(f"Centrality DataFrame shape: {centrality_df.shape}")
    
    # Log top products by each measure
    for measure in ['degree_centrality', 'betweenness_centrality', 'pagerank']:
        top_product = centrality_df.nlargest(1, measure).iloc[0]
        logger.debug(f"Top product by {measure}: ID {top_product['product_id']} (score: {top_product[measure]:.4f})")
    
    return centrality_df

def calculate_basket_expansion_metrics(merged_df, G):
    """Calculate basket expansion potential scoring"""
    logger.info("Calculating basket expansion potential...")
    
    # Calculate basket expansion metrics
    logger.debug("Computing basic aggregation metrics...")
    basket_metrics = merged_df.groupby('product_id').agg({
        'order_id': 'nunique',  # Number of unique orders
        'user_id': 'nunique',   # Number of unique users
        'price': 'mean',        # Average price
        'reordered': 'mean'     # Reorder rate
    }).rename(columns={
        'order_id': 'order_frequency',
        'user_id': 'user_reach',
        'price': 'avg_price',
        'reordered': 'reorder_rate'
    })
    
    # Calculate average basket size when this product is present
    logger.debug("Computing basket size associations...")
    basket_sizes = merged_df.groupby('order_id').size()
    product_basket_sizes = merged_df.merge(
        basket_sizes.to_frame('basket_size'), 
        left_on='order_id', 
        right_index=True
    ).groupby('product_id')['basket_size'].mean()
    
    basket_metrics['avg_basket_size_with_product'] = product_basket_sizes
    
    # Calculate cross-selling strength
    logger.debug("Computing cross-selling strength...")
    cross_sell_strength = {}
    for product_id in G.nodes():
        neighbors = list(G.neighbors(product_id))
        if neighbors:
            # Weight by edge strength and neighbor importance
            strength = sum(G[product_id][neighbor]['weight'] for neighbor in neighbors)
            cross_sell_strength[product_id] = strength / len(neighbors)  # Average strength
        else:
            cross_sell_strength[product_id] = 0
    
    basket_metrics['cross_sell_strength'] = pd.Series(cross_sell_strength)
    
    logger.success(f"Basket expansion metrics calculated for {len(basket_metrics):,} products")
    logger.debug(f"Metrics shape: {basket_metrics.shape}")
    return basket_metrics

def normalize_and_score_basket_metrics(basket_metrics):
    """Normalize metrics and calculate composite basket expansion score"""
    logger.info("Calculating composite basket expansion scores...")
    
    # Normalize metrics
    logger.debug("Normalizing individual metrics...")
    normalization_cols = ['order_frequency', 'user_reach', 'avg_price', 'reorder_rate', 
                         'avg_basket_size_with_product', 'cross_sell_strength']
    
    for col in normalization_cols:
        if col in basket_metrics.columns:
            min_val = basket_metrics[col].min()
            max_val = basket_metrics[col].max()
            
            if max_val > min_val:
                basket_metrics[f'{col}_norm'] = (basket_metrics[col] - min_val) / (max_val - min_val)
            else:
                basket_metrics[f'{col}_norm'] = 0
                logger.warning(f"Column {col} has no variance, setting normalized values to 0")
    
    # Composite basket expansion score
    logger.debug("Computing composite basket expansion score...")
    basket_metrics['basket_expansion_score'] = (
        0.25 * basket_metrics['avg_basket_size_with_product_norm'] +
        0.25 * basket_metrics['cross_sell_strength_norm'] +
        0.20 * basket_metrics['reorder_rate_norm'] +
        0.15 * basket_metrics['avg_price_norm'] +
        0.15 * basket_metrics['user_reach_norm']
    )
    
    basket_metrics['product_id'] = basket_metrics.index
    basket_metrics = basket_metrics.reset_index(drop=True)
    
    # Log statistics
    score_stats = basket_metrics['basket_expansion_score'].describe()
    logger.success(f"Basket expansion scoring completed for {len(basket_metrics):,} products")
    logger.debug(f"Score statistics - Mean: {score_stats['mean']:.4f}, Std: {score_stats['std']:.4f}")
    logger.debug(f"Score range: {score_stats['min']:.4f} to {score_stats['max']:.4f}")
    
    # Log top products
    top_products = basket_metrics.nlargest(5, 'basket_expansion_score')
    logger.debug(f"Top 5 products by expansion score: {top_products['product_id'].tolist()}")
    
    return basket_metrics

def analyze_basket_expansion(merged_df, frequent_pairs):
    """Main function to perform basket expansion analysis"""
    logger.info("=== Starting Basket Expansion Analysis ===")
    
    # Build cross-selling network
    G = build_cross_selling_network(frequent_pairs)
    
    # Calculate centrality measures
    centrality_df = calculate_centrality_measures(G)
    
    # Calculate basket expansion metrics
    basket_metrics = calculate_basket_expansion_metrics(merged_df, G)
    
    # Normalize and score metrics
    basket_metrics = normalize_and_score_basket_metrics(basket_metrics)
    
    logger.success("=== Basket Expansion Analysis Completed ===")
    
    return {
        'network': G,
        'centrality_df': centrality_df,
        'basket_metrics': basket_metrics
    } 