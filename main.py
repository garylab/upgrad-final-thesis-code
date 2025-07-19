#!/usr/bin/env python3
"""
LLM-Enhanced Product Recommendation System
Main execution script for thesis project

This script orchestrates the entire pipeline:
1. Data preprocessing and feature engineering
2. LLM-enhanced feature generation (semantic embeddings, categories)
3. Basket expansion analysis with network analysis
4. Model training and evaluation
5. Hybrid recommendation engine
6. Comprehensive analysis and visualization
"""

from src.config import *
from src.data_preprocessing import preprocess_data
from src.llm_features import generate_llm_features
from src.basket_analysis import analyze_basket_expansion
from src.model_training import train_and_evaluate_models, merge_llm_features
from src.recommendation_engine import run_recommendation_engine
from src.analysis import run_comprehensive_analysis

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("    LLM-ENHANCED PRODUCT RECOMMENDATION SYSTEM")
    logger.info("              Thesis Implementation")
    logger.info("="*80)
    
    try:
        # Step 1: Data Preprocessing
        logger.info("🔄 STEP 1: DATA PREPROCESSING")
        sampled_df, merged_df, products_df = preprocess_data()
        
        # Step 2: LLM Feature Generation
        logger.info("🤖 STEP 2: LLM FEATURE GENERATION")
        llm_features_dict = generate_llm_features(merged_df, products_df)
        
        # Step 3: Basket Expansion Analysis
        logger.info("🛒 STEP 3: BASKET EXPANSION ANALYSIS")
        basket_analysis_dict = analyze_basket_expansion(
            merged_df, 
            llm_features_dict['frequent_pairs']
        )
        
        # Step 4: Merge LLM Features with Dataset
        logger.info("🔗 STEP 4: MERGING LLM FEATURES")
        enhanced_df = merge_llm_features(
            sampled_df, 
            llm_features_dict, 
            basket_analysis_dict
        )
        
        # Step 5: Model Training and Evaluation
        logger.info("🎯 STEP 5: MODEL TRAINING AND EVALUATION")
        model_results = train_and_evaluate_models(enhanced_df)
        
        # Step 6: Recommendation Engine
        logger.info("🚀 STEP 6: RECOMMENDATION ENGINE")
        recommendation_results = run_recommendation_engine(
            model_results, 
            enhanced_df, 
            basket_analysis_dict
        )
        
        # Step 7: Comprehensive Analysis
        logger.info("📊 STEP 7: COMPREHENSIVE ANALYSIS")
        performance_summary = run_comprehensive_analysis(
            model_results, 
            recommendation_results
        )
        
        # Success message
        logger.success("="*80)
        logger.success("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.success("="*80)
        
        return {
            'performance_summary': performance_summary,
            'model_results': model_results,
            'recommendation_results': recommendation_results,
            'enhanced_df': enhanced_df
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR: Pipeline failed with error: {str(e)}")
        logger.error("Please check the error details and try again.")
        raise

def run_quick_demo():
    """Run a quick demo with reduced data for testing"""
    logger.info("Running quick demo with reduced dataset...")
    
    # Temporarily reduce sample fraction for quick testing
    global SAMPLE_FRACTION
    original_sample = SAMPLE_FRACTION
    SAMPLE_FRACTION = 0.005  # Even smaller sample for demo
    
    try:
        results = main()
        logger.success("🎉 Quick demo completed successfully!")
        return results
    finally:
        # Restore original sample fraction
        SAMPLE_FRACTION = original_sample


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        results = run_quick_demo()
    else:
        results = main()
    
    logger.info("To run a quick demo, use: python main.py --demo")
    logger.info("For full analysis, use: python main.py") 