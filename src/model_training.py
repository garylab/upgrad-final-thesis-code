from .config import *

def merge_llm_features(sampled_df, llm_features_dict, basket_analysis_dict):
    """Merge LLM-enhanced features with the sampled dataset"""
    logger.info("Merging LLM-enhanced features...")
    
    # Merge LLM features
    enhanced_df = sampled_df.copy()
    original_shape = enhanced_df.shape
    
    logger.debug("Merging semantic categories...")
    enhanced_df = enhanced_df.merge(llm_features_dict['semantic_cat_df'], on='product_id', how='left')
    
    logger.debug("Merging similarity features...")
    enhanced_df = enhanced_df.merge(llm_features_dict['similarity_df'], on='product_id', how='left')
    
    logger.debug("Merging centrality features...")
    enhanced_df = enhanced_df.merge(basket_analysis_dict['centrality_df'], on='product_id', how='left')
    
    logger.debug("Merging basket metrics...")
    enhanced_df = enhanced_df.merge(
        basket_analysis_dict['basket_metrics'][['product_id', 'basket_expansion_score', 
                                              'cross_sell_strength', 'avg_basket_size_with_product']], 
        on='product_id', how='left'
    )
    
    # Handle missing values for LLM features
    logger.debug("Handling missing values for LLM features...")
    missing_counts = {}
    for feature in LLM_FEATURES:
        if feature in enhanced_df.columns:
            missing_count = enhanced_df[feature].isna().sum()
            missing_counts[feature] = missing_count
            enhanced_df[feature] = enhanced_df[feature].fillna(0)
    
    logger.success("LLM features merged successfully")
    logger.debug(f"Dataset shape: {original_shape} → {enhanced_df.shape}")
    logger.debug(f"Missing values filled: {sum(missing_counts.values()):,} total")
    return enhanced_df

def prepare_training_data(enhanced_df):
    """Prepare training datasets with base and enhanced features"""
    logger.info("Preparing training datasets...")
    
    # Prepare feature sets
    all_features = BASE_FEATURES + LLM_FEATURES
    target = 'reordered'
    
    # Check feature availability
    available_base = [f for f in BASE_FEATURES if f in enhanced_df.columns]
    available_llm = [f for f in LLM_FEATURES if f in enhanced_df.columns]
    
    logger.debug(f"Available base features: {len(available_base)}/{len(BASE_FEATURES)}")
    logger.debug(f"Available LLM features: {len(available_llm)}/{len(LLM_FEATURES)}")
    
    if len(available_base) != len(BASE_FEATURES):
        missing_base = set(BASE_FEATURES) - set(available_base)
        logger.warning(f"Missing base features: {missing_base}")
    
    if len(available_llm) != len(LLM_FEATURES):
        missing_llm = set(LLM_FEATURES) - set(available_llm)
        logger.warning(f"Missing LLM features: {missing_llm}")
    
    # Prepare datasets
    X_base = enhanced_df[available_base].fillna(0)
    X_enhanced = enhanced_df[available_base + available_llm].fillna(0)
    y = enhanced_df[target]
    
    # Log target distribution
    target_dist = y.value_counts(normalize=True)
    logger.info(f"Base features: {len(available_base)}")
    logger.info(f"Enhanced features: {len(available_base + available_llm)}")
    logger.info(f"Target variable distribution:")
    for value, proportion in target_dist.items():
        logger.info(f"  {value}: {proportion:.4f}")
    
    return X_base, X_enhanced, y, available_base + available_llm

def apply_smote_and_split(X_base, X_enhanced, y):
    """Apply SMOTE for class balancing and split data"""
    logger.info("Applying SMOTE and splitting data...")
    
    original_dist = y.value_counts()
    logger.debug(f"Original class distribution: {dict(original_dist)}")
    
    # Balance data using SMOTE
    logger.debug("Applying SMOTE to base features...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_base_resampled, y_base_resampled = smote.fit_resample(X_base, y)
    
    logger.debug("Applying SMOTE to enhanced features...")
    X_enhanced_resampled, y_enhanced_resampled = smote.fit_resample(X_enhanced, y)
    
    balanced_dist = pd.Series(y_base_resampled).value_counts()
    logger.debug(f"Balanced class distribution: {dict(balanced_dist)}")
    
    # Split data
    logger.debug("Splitting data into train/test sets...")
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
        X_base_resampled, y_base_resampled, test_size=0.2, random_state=RANDOM_STATE)
    
    X_enh_train, X_enh_test, y_enh_train, y_enh_test = train_test_split(
        X_enhanced_resampled, y_enhanced_resampled, test_size=0.2, random_state=RANDOM_STATE)
    
    logger.success("Data preparation completed successfully")
    logger.debug(f"Base training set: {X_base_train.shape}")
    logger.debug(f"Enhanced training set: {X_enh_train.shape}")
    logger.debug(f"Test set size: {X_base_test.shape[0]:,} samples")
    
    return {
        'base': {
            'X_train': X_base_train,
            'X_test': X_base_test,
            'y_train': y_base_train,
            'y_test': y_base_test
        },
        'enhanced': {
            'X_train': X_enh_train,
            'X_test': X_enh_test,
            'y_train': y_enh_train,
            'y_test': y_enh_test
        }
    }

def train_base_model(train_data):
    """Train XGBoost model with base features"""
    logger.info("Training model with BASE features...")
    
    # XGBoost with base features
    logger.debug(f"XGBoost parameters: {XGB_PARAMS}")
    xgb_base = XGBClassifier(**XGB_PARAMS)
    
    logger.debug("Fitting base model...")
    xgb_base.fit(train_data['base']['X_train'], train_data['base']['y_train'])
    
    # Predictions
    logger.debug("Generating predictions...")
    y_pred_base = xgb_base.predict(train_data['base']['X_test'])
    y_prob_base = xgb_base.predict_proba(train_data['base']['X_test'])[:, 1]
    
    # Metrics
    accuracy_base = accuracy_score(train_data['base']['y_test'], y_pred_base)
    auc_base = roc_auc_score(train_data['base']['y_test'], y_prob_base)
    
    logger.success(f"Base Model trained - Accuracy: {accuracy_base:.4f}, AUC: {auc_base:.4f}")
    
    return {
        'model': xgb_base,
        'predictions': y_pred_base,
        'probabilities': y_prob_base,
        'accuracy': accuracy_base,
        'auc': auc_base
    }

def train_enhanced_model(train_data):
    """Train XGBoost model with enhanced features"""
    logger.info("Training model with ENHANCED features...")
    
    # XGBoost with enhanced features
    logger.debug(f"XGBoost parameters: {XGB_PARAMS}")
    xgb_enhanced = XGBClassifier(**XGB_PARAMS)
    
    logger.debug("Fitting enhanced model...")
    xgb_enhanced.fit(train_data['enhanced']['X_train'], train_data['enhanced']['y_train'])
    
    # Predictions
    logger.debug("Generating predictions...")
    y_pred_enhanced = xgb_enhanced.predict(train_data['enhanced']['X_test'])
    y_prob_enhanced = xgb_enhanced.predict_proba(train_data['enhanced']['X_test'])[:, 1]
    
    # Metrics
    accuracy_enhanced = accuracy_score(train_data['enhanced']['y_test'], y_pred_enhanced)
    auc_enhanced = roc_auc_score(train_data['enhanced']['y_test'], y_prob_enhanced)
    
    logger.success(f"Enhanced Model trained - Accuracy: {accuracy_enhanced:.4f}, AUC: {auc_enhanced:.4f}")
    
    return {
        'model': xgb_enhanced,
        'predictions': y_pred_enhanced,
        'probabilities': y_prob_enhanced,
        'accuracy': accuracy_enhanced,
        'auc': auc_enhanced
    }

def calculate_feature_importance(enhanced_model, all_features):
    """Calculate and categorize feature importance"""
    logger.info("Analyzing feature importance...")
    
    # Analyze feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': enhanced_model['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Categorize features
    feature_importance['category'] = feature_importance['feature'].apply(
        lambda x: 'LLM-Enhanced' if x in LLM_FEATURES else 'Traditional'
    )
    
    # Log top features
    logger.debug("Top 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.debug(f"  {row['feature']} ({row['category']}): {row['importance']:.4f}")
    
    # Category analysis
    category_importance = feature_importance.groupby('category')['importance'].sum()
    total_importance = feature_importance['importance'].sum()
    
    for category, importance in category_importance.items():
        percentage = (importance / total_importance) * 100
        logger.info(f"{category} features contribute {percentage:.2f}% of total importance")
    
    logger.success("Feature importance analysis completed")
    return feature_importance

def train_and_evaluate_models(enhanced_df):
    """Main function to train and evaluate both base and enhanced models"""
    logger.info("=== Starting Model Training and Evaluation ===")
    
    # Prepare training data
    X_base, X_enhanced, y, all_features = prepare_training_data(enhanced_df)
    
    # Apply SMOTE and split
    train_data = apply_smote_and_split(X_base, X_enhanced, y)
    
    # Train models
    base_results = train_base_model(train_data)
    enhanced_results = train_enhanced_model(train_data)
    
    # Calculate feature importance
    feature_importance = calculate_feature_importance(enhanced_results, all_features)
    
    # Calculate improvements
    accuracy_improvement = ((enhanced_results['accuracy'] - base_results['accuracy']) / base_results['accuracy']) * 100
    auc_improvement = ((enhanced_results['auc'] - base_results['auc']) / base_results['auc']) * 100
    
    logger.success("=== Model Training and Evaluation Completed ===")
    logger.info(f"Model Performance Improvements:")
    logger.info(f"  Accuracy improvement: +{accuracy_improvement:.2f}%")
    logger.info(f"  AUC improvement: +{auc_improvement:.2f}%")
    
    return {
        'base_results': base_results,
        'enhanced_results': enhanced_results,
        'feature_importance': feature_importance,
        'train_data': train_data,
        'improvements': {
            'accuracy_improvement': accuracy_improvement,
            'auc_improvement': auc_improvement
        }
    } 