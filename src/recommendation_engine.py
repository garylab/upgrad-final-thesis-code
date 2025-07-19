from .config import *

def calculate_hybrid_score(ml_probability, basket_expansion_score, price, cross_sell_strength, 
                          alpha=None, beta=None, gamma=None, delta=None):
    """
    Calculate hybrid recommendation score combining:
    - ML prediction probability
    - Basket expansion potential
    - Price/revenue potential
    - Cross-selling strength
    """
    # Use default weights if not provided
    weights = HYBRID_WEIGHTS
    if alpha is not None:
        weights['alpha'] = alpha
    if beta is not None:
        weights['beta'] = beta
    if gamma is not None:
        weights['gamma'] = gamma
    if delta is not None:
        weights['delta'] = delta
    
    # Normalize price (higher price = higher potential revenue)
    price_norm = (price - np.min(price)) / (np.max(price) - np.min(price)) if np.max(price) > np.min(price) else 0

    hybrid_score = (
        weights['alpha'] * ml_probability +
        weights['beta'] * basket_expansion_score +
        weights['gamma'] * price_norm +
        weights['delta'] * cross_sell_strength
    )

    return hybrid_score

def create_revenue_analysis_data(model_results, enhanced_df):
    """Create dataset for revenue impact analysis"""
    logger.info("Creating revenue analysis data...")
    
    # Get test data with all features - handle different data types
    train_data = model_results['train_data']
    enhanced_results = model_results['enhanced_results']
    
    logger.debug("Mapping test indices to original dataframe...")
    if hasattr(train_data['enhanced']['X_test'], 'index'):
        test_indices = enhanced_df.index.get_indexer_for(train_data['enhanced']['X_test'].index)
    else:
        test_indices = range(len(train_data['enhanced']['X_test']))

    # Get corresponding rows from enhanced_df
    test_data = enhanced_df.iloc[test_indices].copy()
    
    logger.debug("Calculating hybrid scores...")
    # Calculate hybrid scores
    hybrid_scores = calculate_hybrid_score(
        ml_probability=enhanced_results['probabilities'],
        basket_expansion_score=test_data['basket_expansion_score'].fillna(0).values,
        price=test_data['price'].fillna(0).values,
        cross_sell_strength=test_data['cross_sell_strength'].fillna(0).values
    )

    logger.success(f"Calculated hybrid scores for {len(hybrid_scores):,} test samples")
    
    # Create revenue analysis dataframe
    revenue_analysis = pd.DataFrame({
        'actual_reorder': train_data['enhanced']['y_test'].values,
        'ml_probability': enhanced_results['probabilities'],
        'hybrid_score': hybrid_scores,
        'price': test_data['price'].fillna(0).values,
        'basket_expansion_score': test_data['basket_expansion_score'].fillna(0).values
    })
    
    logger.debug(f"Revenue analysis DataFrame shape: {revenue_analysis.shape}")
    return revenue_analysis

def calculate_revenue_metrics(df, score_column, top_k=1000):
    """Calculate revenue metrics for different strategies"""
    # Sort by score and take top K recommendations
    top_recommendations = df.nlargest(top_k, score_column)

    # Calculate metrics
    precision = top_recommendations['actual_reorder'].mean()
    total_revenue = (top_recommendations['actual_reorder'] * top_recommendations['price']).sum()
    avg_basket_expansion = top_recommendations['basket_expansion_score'].mean()

    return {
        'precision': precision,
        'total_revenue': total_revenue,
        'avg_revenue_per_rec': total_revenue / top_k,
        'avg_basket_expansion': avg_basket_expansion
    }

def analyze_revenue_impact(revenue_analysis):
    """Analyze revenue impact of different recommendation strategies"""
    logger.info("Analyzing revenue impact...")
    
    # Compare strategies
    logger.debug("Calculating ML-only metrics...")
    ml_only_metrics = calculate_revenue_metrics(revenue_analysis, 'ml_probability')
    
    logger.debug("Calculating hybrid strategy metrics...")
    hybrid_metrics = calculate_revenue_metrics(revenue_analysis, 'hybrid_score')

    logger.info("=== REVENUE IMPACT COMPARISON ===")
    logger.info("ML-Only Strategy:")
    for metric, value in ml_only_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("Hybrid Strategy (LLM-Enhanced):")
    for metric, value in hybrid_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("Improvement with Hybrid Approach:")
    improvement_details = {}
    for metric in ml_only_metrics.keys():
        improvement = ((hybrid_metrics[metric] - ml_only_metrics[metric]) / ml_only_metrics[metric]) * 100
        improvement_details[metric] = improvement
        logger.info(f"  {metric}: {improvement:+.2f}%")
    
    # Calculate revenue improvement
    revenue_improvement = ((hybrid_metrics['total_revenue'] - ml_only_metrics['total_revenue']) / ml_only_metrics['total_revenue']) * 100
    
    logger.success(f"Revenue analysis completed. Overall revenue improvement: {revenue_improvement:+.2f}%")
    
    return {
        'ml_only_metrics': ml_only_metrics,
        'hybrid_metrics': hybrid_metrics,
        'revenue_improvement': revenue_improvement,
        'improvement_details': improvement_details
    }

def generate_cross_sell_recommendations(user_basket, G, enhanced_df, top_n=5):
    """
    Generate cross-selling recommendations based on current basket
    using network analysis and hybrid scoring
    """
    recommendations = []

    # Find products frequently bought with items in current basket
    candidate_products = set()
    for product_id in user_basket:
        if product_id in G:
            neighbors = list(G.neighbors(product_id))
            candidate_products.update(neighbors)

    # Remove products already in basket
    candidate_products = candidate_products - set(user_basket)

    # Score candidates
    for product_id in candidate_products:
        # Get product features
        product_data = enhanced_df[enhanced_df['product_id'] == product_id]
        if len(product_data) > 0:
            product_data = product_data.iloc[0]

            # Calculate network strength with current basket
            network_strength = sum(G[product_id][basket_item]['weight'] 
                                 for basket_item in user_basket 
                                 if basket_item in G and G.has_edge(product_id, basket_item))

            # Get hybrid score components
            basket_exp_score = getattr(product_data, 'basket_expansion_score', 0)
            price = getattr(product_data, 'price', 0)
            cross_sell_str = getattr(product_data, 'cross_sell_strength', 0)

            # Calculate recommendation score
            rec_score = (
                0.4 * network_strength +
                0.3 * basket_exp_score +
                0.2 * (price / enhanced_df['price'].max()) +  # Normalized price
                0.1 * cross_sell_str
            )

            recommendations.append({
                'product_id': product_id,
                'product_name': getattr(product_data, 'product_name', 'Unknown'),
                'price': price,
                'recommendation_score': rec_score,
                'network_strength': network_strength
            })

    # Sort by recommendation score and return top N
    recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
    return recommendations[:top_n]

def demo_cross_sell_recommendations(enhanced_df, G):
    """Generate and display example cross-selling recommendations"""
    logger.info("=== Demonstrating Cross-Selling Recommendations ===")
    
    # Example: Generate recommendations for a sample basket
    sample_user_orders = enhanced_df[enhanced_df['user_id'] == enhanced_df['user_id'].iloc[0]]
    sample_basket = sample_user_orders['product_id'].tolist()[:3]  # First 3 products

    logger.info(f"Sample basket contains products: {sample_basket}")
    
    try:
        cross_sell_recs = generate_cross_sell_recommendations(sample_basket, G, enhanced_df)
        
        if cross_sell_recs:
            logger.info("Cross-selling recommendations:")
            for i, rec in enumerate(cross_sell_recs, 1):
                logger.info(f"  {i}. {rec['product_name']} (ID: {rec['product_id']})")
                logger.info(f"     Price: ${rec['price']:.2f} | Score: {rec['recommendation_score']:.4f} | Network: {rec['network_strength']:.2f}")
        else:
            logger.warning("No cross-selling recommendations found for this basket")
            
    except Exception as e:
        logger.error(f"Error generating cross-sell recommendations: {e}")

def save_model_artifacts(model_results, revenue_analysis_results):
    """Save enhanced model and comprehensive results"""
    logger.info("Saving model artifacts and results...")
    
    enhanced_results = model_results['enhanced_results']
    base_results = model_results['base_results']
    feature_importance = model_results['feature_importance']
    improvements = model_results['improvements']
    
    try:
        # Save the enhanced model
        logger.debug("Saving trained model...")
        model_artifacts = {
            'model': enhanced_results['model'],
            'feature_names': BASE_FEATURES + LLM_FEATURES,
            'scaler': None,  # Add if used
            'performance_metrics': {
                'accuracy': enhanced_results['accuracy'],
                'auc': enhanced_results['auc'],
                'revenue_improvement': revenue_analysis_results['revenue_improvement']
            }
        }

        joblib.dump(model_artifacts, 'llm_enhanced_recommendation_model.pkl')
        logger.debug("Model saved to: llm_enhanced_recommendation_model.pkl")

        # Save comprehensive results
        logger.debug("Saving comprehensive results...")
        results = {
            'base_model_performance': {
                'accuracy': float(base_results['accuracy']),
                'auc': float(base_results['auc'])
            },
            'enhanced_model_performance': {
                'accuracy': float(enhanced_results['accuracy']),
                'auc': float(enhanced_results['auc'])
            },
            'improvements': {
                'accuracy_improvement_pct': float(improvements['accuracy_improvement']),
                'auc_improvement_pct': float(improvements['auc_improvement']),
                'revenue_improvement_pct': float(revenue_analysis_results['revenue_improvement'])
            },
            'feature_importance': feature_importance.to_dict('records'),
            'revenue_metrics': {
                'ml_only': revenue_analysis_results['ml_only_metrics'],
                'hybrid': revenue_analysis_results['hybrid_metrics']
            }
        }

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        results = deep_convert(results)

        with open('llm_enhanced_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug("Results saved to: llm_enhanced_results.json")
        logger.success("✅ Model artifacts and results saved successfully:")
        logger.success("   • llm_enhanced_recommendation_model.pkl")
        logger.success("   • llm_enhanced_results.json")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise

def run_recommendation_engine(model_results, enhanced_df, basket_analysis_dict):
    """Main function to run the recommendation engine and analysis"""
    logger.info("=== Starting Recommendation Engine ===")
    
    # Create revenue analysis data
    revenue_analysis = create_revenue_analysis_data(model_results, enhanced_df)
    
    # Analyze revenue impact
    revenue_analysis_results = analyze_revenue_impact(revenue_analysis)
    
    # Demo cross-selling recommendations
    demo_cross_sell_recommendations(enhanced_df, basket_analysis_dict['network'])
    
    # Save model artifacts
    save_model_artifacts(model_results, revenue_analysis_results)
    
    logger.success("=== Recommendation Engine Completed ===")
    
    return {
        'revenue_analysis': revenue_analysis,
        'revenue_analysis_results': revenue_analysis_results
    } 