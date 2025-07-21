from .config import *

def create_matplotlib_fallback_visualization(model_results, recommendation_results):
    """Create fallback visualization using matplotlib if plotly fails"""
    logger.info("Creating matplotlib fallback visualization...")
    
    feature_importance = model_results['feature_importance']
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    revenue_analysis_results = recommendation_results['revenue_analysis_results']
    
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Create a 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LLM-Enhanced Product Recommendation System - Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature importance by category
    category_importance = feature_importance.groupby('category')['importance'].sum()
    colors = ['skyblue', 'lightcoral']
    ax1.bar(category_importance.index, category_importance.values, color=colors)
    ax1.set_title('Feature Importance by Category')
    ax1.set_ylabel('Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Top 10 features
    top_features = feature_importance.head(10)
    colors = ['lightcoral' if cat == 'LLM-Enhanced' else 'skyblue' for cat in top_features['category']]
    ax2.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'], fontsize=8)
    ax2.set_title('Top 10 Most Important Features')
    ax2.set_xlabel('Importance')
    ax2.invert_yaxis()
    
    # Plot 3: Model performance comparison
    models = ['Base Model', 'Enhanced Model']
    accuracies = [base_results['accuracy'], enhanced_results['accuracy']]
    aucs = [base_results['auc'], enhanced_results['auc']]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x_pos - width/2, accuracies, width, label='Accuracy', color='lightblue')
    ax3.bar(x_pos + width/2, aucs, width, label='AUC', color='lightgreen')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Plot 4: Revenue impact
    strategies = ['ML-Only', 'Hybrid\n(LLM-Enhanced)']
    revenues = [
        revenue_analysis_results['ml_only_metrics']['total_revenue'], 
        revenue_analysis_results['hybrid_metrics']['total_revenue']
    ]
    
    ax4.bar(strategies, revenues, color='gold')
    ax4.set_title('Revenue Impact by Strategy')
    ax4.set_ylabel('Total Revenue')
    ax4.tick_params(axis='x', rotation=0)
    
    # Format revenue values
    for i, v in enumerate(revenues):
        ax4.text(i, v + max(revenues) * 0.01, f'${v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the matplotlib figure
    plt.savefig("output/comprehensive_analysis_matplotlib.png", dpi=300, bbox_inches='tight')
    plt.savefig("output/comprehensive_analysis_matplotlib.svg", bbox_inches='tight')
    
    logger.success("Saved: output/comprehensive_analysis_matplotlib.png")
    logger.success("Saved: output/comprehensive_analysis_matplotlib.svg")
    
    plt.close()  # Clean up
    logger.success("Matplotlib fallback visualization completed")

def create_individual_visualizations(model_results, recommendation_results):
    """Create individual visualization files for each analysis component"""
    logger.info("Creating individual visualization files...")
    
    feature_importance = model_results['feature_importance']
    base_results = model_results['base_results']
    enhanced_results = model_results['enhanced_results']
    revenue_analysis_results = recommendation_results['revenue_analysis_results']
    
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    try:
        # 1. Feature Importance by Category
        fig1 = go.Figure()
        category_importance = feature_importance.groupby('category')['importance'].sum().reset_index()
        fig1.add_trace(go.Bar(
            x=category_importance['category'], 
            y=category_importance['importance'],
            marker_color=['skyblue', 'lightcoral'],
            name='Feature Importance'
        ))
        fig1.update_layout(
            title="Feature Importance by Category",
            xaxis_title="Feature Category",
            yaxis_title="Total Importance",
            width=800, height=600
        )
        fig1.write_image("output/feature_importance_by_category.png", width=800, height=600, scale=2)
        fig1.write_html("output/feature_importance_by_category.html")
        logger.success("Saved: output/feature_importance_by_category.png")
        
        # 2. Top Features
        fig2 = go.Figure()
        top_features = feature_importance.head(15)
        colors = ['lightcoral' if cat == 'LLM-Enhanced' else 'skyblue' for cat in top_features['category']]
        fig2.add_trace(go.Bar(
            y=top_features['feature'], 
            x=top_features['importance'],
            orientation='h',
            marker_color=colors,
            name='Feature Importance'
        ))
        fig2.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            width=800, height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig2.write_image("output/top_features.png", width=800, height=600, scale=2)
        fig2.write_html("output/top_features.html")
        logger.success("Saved: output/top_features.png")
        
        # 3. Model Performance Comparison
        fig3 = go.Figure()
        models = ['Base Model', 'Enhanced Model']
        accuracies = [base_results['accuracy'], enhanced_results['accuracy']]
        aucs = [base_results['auc'], enhanced_results['auc']]
        
        fig3.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'))
        fig3.add_trace(go.Bar(x=models, y=aucs, name='AUC', marker_color='lightgreen'))
        fig3.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            width=800, height=600,
            barmode='group'
        )
        fig3.write_image("output/model_performance.png", width=800, height=600, scale=2)
        fig3.write_html("output/model_performance.html")
        logger.success("Saved: output/model_performance.png")
        
        # 4. Revenue Impact
        fig4 = go.Figure()
        strategies = ['ML-Only', 'Hybrid (LLM-Enhanced)']
        revenues = [
            revenue_analysis_results['ml_only_metrics']['total_revenue'], 
            revenue_analysis_results['hybrid_metrics']['total_revenue']
        ]
        
        fig4.add_trace(go.Bar(
            x=strategies, 
            y=revenues, 
            marker_color='gold',
            name='Total Revenue',
            text=[f'${r:.0f}' for r in revenues],
            textposition='auto'
        ))
        fig4.update_layout(
            title="Revenue Impact by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Total Revenue ($)",
            width=800, height=600
        )
        fig4.write_image("output/revenue_impact.png", width=800, height=600, scale=2)
        fig4.write_html("output/revenue_impact.html")
        logger.success("Saved: output/revenue_impact.png")
        
        logger.success("All individual visualizations created successfully")
        
    except Exception as e:
        logger.error(f"Error creating individual visualizations: {e}")
        logger.warning("Some individual visualizations may not have been created")

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

        # Save as image files
        logger.info("Saving visualization as image files...")
        
        # Create output directory
        import os
        os.makedirs("output", exist_ok=True)
        
        # Save as PNG (high quality)
        fig.write_image("output/comprehensive_analysis.png", width=1600, height=800, scale=2)
        logger.success("Saved: output/comprehensive_analysis.png")
        
        # Save as SVG (vector format)
        fig.write_image("output/comprehensive_analysis.svg", width=1600, height=800)
        logger.success("Saved: output/comprehensive_analysis.svg")
        
        # Save as HTML (interactive)
        fig.write_html("output/comprehensive_analysis.html")
        logger.success("Saved: output/comprehensive_analysis.html")
        
        # Also show interactively if possible
        try:
            fig.show()
            logger.debug("Interactive visualization displayed")
        except:
            logger.debug("Interactive display not available, saved to files instead")
        
        logger.success("Comprehensive visualization created and saved successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        # Try to save a basic matplotlib version as fallback
        try:
            logger.warning("Attempting to create fallback matplotlib visualization...")
            create_matplotlib_fallback_visualization(model_results, recommendation_results)
        except Exception as fallback_error:
            logger.error(f"Fallback visualization also failed: {fallback_error}")

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
        # Create comprehensive visualization
        create_comprehensive_visualization(model_results, recommendation_results)
        
        # Create individual visualizations
        create_individual_visualizations(model_results, recommendation_results)
        
        # Print feature importance analysis
        print_feature_importance_analysis(model_results['feature_importance'])
        
        # Log detailed metrics for debugging
        log_detailed_metrics(model_results, recommendation_results)
        
        # Print thesis contributions summary
        print_thesis_contributions_summary(model_results, recommendation_results)
        
        # Generate performance summary
        performance_summary = generate_performance_summary(model_results, recommendation_results)
        
        # Log output summary
        logger.info("=== Output Files Generated ===")
        logger.info("📊 Visualizations:")
        logger.info("   • output/comprehensive_analysis.png - Complete analysis dashboard")
        logger.info("   • output/feature_importance_by_category.png - Feature category breakdown")
        logger.info("   • output/top_features.png - Top 15 important features")
        logger.info("   • output/model_performance.png - Model comparison")
        logger.info("   • output/revenue_impact.png - Revenue analysis")
        logger.info("📝 Interactive Files:")
        logger.info("   • output/*.html - Interactive versions of all charts")
        logger.info("🔍 Vector Graphics:")
        logger.info("   • output/*.svg - Scalable vector versions")
        
        logger.success("=== Comprehensive Analysis Completed ===")
        return performance_summary
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise 