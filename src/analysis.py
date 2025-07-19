from .config import *

def create_comprehensive_visualization(model_results, recommendation_results):
    """Create comprehensive visualization of all results"""
    logger.info("Creating comprehensive visualizations...")
    
    feature_importance = model_results['feature_importance']
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    revenue_analysis_results = recommendation_results['revenue_analysis_results']
    
    try:
        # Create comprehensive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance by Category', 'Top 15 Most Important Features',
                           'Model Performance Comparison', 'Revenue Impact by Strategy'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Plot 1: Feature importance by category
        category_importance = feature_importance.groupby('category')['importance'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=category_importance['category'], y=category_importance['importance'],
                   name='Importance by Category', marker_color=['skyblue', 'lightcoral']),
            row=1, col=1
        )

        # Plot 2: Top features
        top_features = feature_importance.head(15)
        colors = ['lightcoral' if cat == 'LLM-Enhanced' else 'skyblue' for cat in top_features['category']]
        fig.add_trace(
            go.Bar(y=top_features['feature'], x=top_features['importance'],
                   orientation='h', marker_color=colors, name='Feature Importance'),
            row=1, col=2
        )

        # Plot 3: Model performance comparison
        models = ['Base Model', 'Enhanced Model']
        accuracies = [base_results['accuracy'], enhanced_results['accuracy']]
        aucs = [base_results['auc'], enhanced_results['auc']]

        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=aucs, name='AUC', marker_color='lightgreen'),
            row=2, col=1
        )

        # Plot 4: Revenue impact
        strategies = ['ML-Only', 'Hybrid (LLM-Enhanced)']
        revenues = [
            revenue_analysis_results['ml_only_metrics']['total_revenue'], 
            revenue_analysis_results['hybrid_metrics']['total_revenue']
        ]
        fig.add_trace(
            go.Bar(x=strategies, y=revenues, name='Total Revenue', marker_color='gold'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="LLM-Enhanced Product Recommendation System - Comprehensive Analysis",
            showlegend=False
        )

        fig.show()
        logger.success("Comprehensive visualization created successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def print_feature_importance_analysis(feature_importance):
    """Print detailed feature importance analysis"""
    logger.info("=== Feature Importance Analysis ===")
    
    # Top 20 features
    logger.info("Top 20 Most Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']} ({row['category']}): {row['importance']:.4f}")
    
    # Analyze LLM vs Traditional features
    llm_importance = feature_importance[feature_importance['category'] == 'LLM-Enhanced']['importance'].sum()
    traditional_importance = feature_importance[feature_importance['category'] == 'Traditional']['importance'].sum()
    total_importance = feature_importance['importance'].sum()
    
    logger.info("=== Feature Category Analysis ===")
    logger.info(f"LLM-Enhanced features contribution: {(llm_importance/total_importance)*100:.2f}%")
    logger.info(f"Traditional features contribution: {(traditional_importance/total_importance)*100:.2f}%")
    
    # Feature count analysis
    llm_count = len(feature_importance[feature_importance['category'] == 'LLM-Enhanced'])
    traditional_count = len(feature_importance[feature_importance['category'] == 'Traditional'])
    
    logger.debug(f"LLM-Enhanced features count: {llm_count}")
    logger.debug(f"Traditional features count: {traditional_count}")

def print_thesis_contributions_summary(model_results, recommendation_results):
    """Print comprehensive thesis contributions summary"""
    
    improvements = model_results['improvements']
    revenue_improvement = recommendation_results['revenue_analysis_results']['revenue_improvement']
    hybrid_metrics = recommendation_results['revenue_analysis_results']['hybrid_metrics']
    
    logger.info("="*60)
    logger.info("           THESIS CONTRIBUTIONS SUMMARY")
    logger.info("="*60)

    logger.info("🎯 APPROACH #1: LLM-ENHANCED FEATURE ENGINEERING")
    logger.info("   • Generated semantic embeddings for products")
    logger.info("   • Created LLM-generated product categories")
    logger.info("   • Calculated semantic similarity features for cross-selling")
    logger.info(f"   • Model accuracy improvement: +{improvements['accuracy_improvement']:.2f}%")
    logger.info(f"   • AUC improvement: +{improvements['auc_improvement']:.2f}%")

    logger.info("🚀 APPROACH #2: HYBRID SCORING FOR BASKET EXPANSION")
    logger.info("   • Built co-occurrence network for cross-selling analysis")
    logger.info("   • Calculated basket expansion scores for revenue optimization")
    logger.info(f"   • Revenue improvement over ML-only: +{revenue_improvement:.2f}%")
    logger.info("   • Enhanced cross-selling recommendation engine")

    logger.info("📊 KEY PERFORMANCE METRICS:")
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    logger.info(f"   • Base Model: {base_results['accuracy']:.4f} accuracy, {base_results['auc']:.4f} AUC")
    logger.info(f"   • Enhanced Model: {enhanced_results['accuracy']:.4f} accuracy, {enhanced_results['auc']:.4f} AUC")
    logger.info(f"   • Revenue per recommendation: ${hybrid_metrics['avg_revenue_per_rec']:.2f}")
    logger.info(f"   • Basket expansion potential: {hybrid_metrics['avg_basket_expansion']:.4f}")

    logger.info("🎓 THESIS INNOVATION HIGHLIGHTS:")
    logger.info("   ✓ Novel application of LLMs for grocery recommendation feature engineering")
    logger.info("   ✓ Semantic product understanding beyond traditional categorical encoding")
    logger.info("   ✓ Network analysis for cross-selling opportunity identification")
    logger.info("   ✓ Revenue-focused hybrid scoring system for business impact")
    logger.info("   ✓ Measurable improvements in both accuracy and revenue potential")

    logger.info("💡 FUTURE RESEARCH DIRECTIONS:")
    logger.info("   • Real-time LLM feature generation for new products")
    logger.info("   • Multi-modal embeddings incorporating product images")
    logger.info("   • Personalized semantic similarity based on user preferences")
    logger.info("   • Dynamic basket expansion scoring with temporal patterns")

    logger.info("="*60)
    logger.success("🎉 LLM-Enhanced recommendation system successfully implemented!")

def generate_performance_summary(model_results, recommendation_results):
    """Generate a concise performance summary"""
    logger.debug("Generating performance summary...")
    
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    improvements = model_results['improvements']
    revenue_analysis_results = recommendation_results['revenue_analysis_results']
    
    summary = {
        'model_performance': {
            'base_accuracy': base_results['accuracy'],
            'enhanced_accuracy': enhanced_results['accuracy'],
            'accuracy_improvement_pct': improvements['accuracy_improvement'],
            'base_auc': base_results['auc'],
            'enhanced_auc': enhanced_results['auc'],
            'auc_improvement_pct': improvements['auc_improvement']
        },
        'revenue_impact': {
            'ml_only_revenue': revenue_analysis_results['ml_only_metrics']['total_revenue'],
            'hybrid_revenue': revenue_analysis_results['hybrid_metrics']['total_revenue'],
            'revenue_improvement_pct': revenue_analysis_results['revenue_improvement'],
            'hybrid_avg_revenue_per_rec': revenue_analysis_results['hybrid_metrics']['avg_revenue_per_rec']
        }
    }
    
    logger.debug("Performance summary generated successfully")
    return summary

def log_detailed_metrics(model_results, recommendation_results):
    """Log detailed performance metrics for debugging"""
    logger.debug("=== Detailed Performance Metrics ===")
    
    # Model metrics
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    
    logger.debug(f"Base model accuracy: {base_results['accuracy']:.6f}")
    logger.debug(f"Base model AUC: {base_results['auc']:.6f}")
    logger.debug(f"Enhanced model accuracy: {enhanced_results['accuracy']:.6f}")
    logger.debug(f"Enhanced model AUC: {enhanced_results['auc']:.6f}")
    
    # Revenue metrics
    revenue_results = recommendation_results['revenue_analysis_results']
    logger.debug(f"ML-only total revenue: {revenue_results['ml_only_metrics']['total_revenue']:.2f}")
    logger.debug(f"Hybrid total revenue: {revenue_results['hybrid_metrics']['total_revenue']:.2f}")
    logger.debug(f"ML-only precision: {revenue_results['ml_only_metrics']['precision']:.4f}")
    logger.debug(f"Hybrid precision: {revenue_results['hybrid_metrics']['precision']:.4f}")

def run_comprehensive_analysis(model_results, recommendation_results):
    """Run comprehensive analysis and visualization"""
    logger.info("=== Starting Comprehensive Analysis ===")
    
    try:
        # Create visualizations
        create_comprehensive_visualization(model_results, recommendation_results)
        
        # Print feature importance analysis
        print_feature_importance_analysis(model_results['feature_importance'])
        
        # Log detailed metrics for debugging
        log_detailed_metrics(model_results, recommendation_results)
        
        # Print thesis contributions summary
        print_thesis_contributions_summary(model_results, recommendation_results)
        
        # Generate performance summary
        performance_summary = generate_performance_summary(model_results, recommendation_results)
        
        logger.success("=== Comprehensive Analysis Completed ===")
        return performance_summary
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise 