import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm


class LLMFeatures:
    def __init__(self, llm_api_key=None):
        """Initialize with minimal, high-impact features only"""
        pass
    
    def extract_core_features(self, product_name, department, aisle, price):
        """
        Extract only the 5 most impactful semantic features
        """
        prompt = f"""
        Product: {product_name}
        Department: {department}  
        Aisle: {aisle}
        Price: ${price:.2f}

        Extract these 5 features (score 0-10) and return as JSON:

        1. premium_score: How premium/high-quality is this? (0=basic/generic, 10=premium/brand)
        2. necessity_score: How essential/necessary? (0=luxury/optional, 10=daily necessity)
        3. impulse_score: Impulse purchase likelihood? (0=planned purchase, 10=impulse buy)
        4. cross_sell_score: Likely to trigger more purchases? (0=standalone, 10=leads to many items)
        5. product_type: "food" or "non_food"

        Return only valid JSON.
        """
        
        try:
            response = self.call_llm_api(prompt)
            return self.parse_llm_response(response)
        except Exception as e:
            print(f"Error processing {product_name}: {e}")
            return self.get_default_features()
    
    def call_llm_api(self, prompt):
        """Replace with your LLM API call"""
        # Mock response for demonstration
        return """{
            "premium_score": 6.5,
            "necessity_score": 7.0,
            "impulse_score": 4.5,
            "cross_sell_score": 6.0,
            "product_type": "food"
        }"""
    
    def parse_llm_response(self, response):
        """Parse LLM response"""
        try:
            return json.loads(response)
        except:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    return self.get_default_features()
            return self.get_default_features()
    
    def get_default_features(self):
        """Default values if LLM fails"""
        return {
            "premium_score": 5.0,
            "necessity_score": 5.0,
            "impulse_score": 5.0,
            "cross_sell_score": 5.0,
            "product_type": "food"
        }
    
    def add_efficient_features(self, df, sample_size=None):
        """
        Add only the most efficient features for maximum impact
        """
        print("Adding efficient LLM features...")
        
        if sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        # Step 1: Core price features (3 features)
        print("Adding price features...")
        df_sample['price_percentile'] = df_sample.groupby('department')['price'].rank(pct=True)
        df_sample['is_premium_priced'] = (df_sample['price_percentile'] > 0.7).astype(int)
        df_sample['price_vs_dept_avg'] = df_sample['price'] / df_sample.groupby('department')['price'].transform('mean')
        
        # Step 2: User behavior features (2 features)
        print("Adding user features...")
        user_avg_price = df_sample.groupby('user_id')['price'].mean()
        df_sample['user_premium_tendency'] = df_sample['user_id'].map(user_avg_price)
        df_sample['user_vs_product_price'] = df_sample['price'] / df_sample['user_premium_tendency']
        
        # Step 3: LLM semantic features (4 core features)
        print("Extracting LLM features...")
        unique_products = df_sample[['product_id', 'product_name', 'department', 'aisle', 'price']].drop_duplicates()
        
        semantic_features_list = []
        for idx, row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Processing"):
            features = self.extract_core_features(
                row['product_name'], 
                row['department'], 
                row['aisle'], 
                row['price']
            )
            features['product_id'] = row['product_id']
            semantic_features_list.append(features)
        
        semantic_df = pd.DataFrame(semantic_features_list)
        df_enhanced = df_sample.merge(semantic_df, on='product_id', how='left')
        
        # Step 4: Create 3 high-impact derived features
        print("Creating derived features...")
        # Revenue potential = premium score × necessity (premium necessities = high revenue)
        df_enhanced['revenue_potential'] = df_enhanced['premium_score'] * df_enhanced['necessity_score']
        
        # Basket expansion = cross-sell × impulse (items that lead to bigger baskets)
        df_enhanced['basket_expansion_score'] = df_enhanced['cross_sell_score'] * (10 - df_enhanced['impulse_score'])
        
        # User-product fit = user premium tendency × product premium score
        df_enhanced['user_product_fit'] = df_enhanced['user_premium_tendency'] * df_enhanced['premium_score'] / 100
        
        # Step 5: One binary categorical feature
        df_enhanced['is_food'] = (df_enhanced['product_type'] == 'food').astype(int)
        
        print(f"Added {len(df_enhanced.columns) - len(df.columns)} efficient features")
        print("New features:", [col for col in df_enhanced.columns if col not in df.columns])
        
        return df_enhanced
    
    def get_feature_list(self):
        """Return list of all new features created"""
        return [
            # Price features (3)
            'price_percentile', 'is_premium_priced', 'price_vs_dept_avg',
            
            # User features (2)  
            'user_premium_tendency', 'user_vs_product_price',
            
            # LLM semantic features (4)
            'premium_score', 'necessity_score', 'impulse_score', 'cross_sell_score',
            
            # Derived features (3)
            'revenue_potential', 'basket_expansion_score', 'user_product_fit',
            
            # Categorical (1)
            'is_food'
        ]

# Usage:
def main():
    # Initialize
    feature_engineer = EfficientLLMFeatures()
    
    # Add features to your dataframe
    # df_enhanced = feature_engineer.add_efficient_features(df)
    
    # Get list of new features for your ML models
    new_features = feature_engineer.get_feature_list()
    print(f"Total new features: {len(new_features)}")
    
    # Use in your Random Forest/XGBoost
    # X = df_enhanced[original_features + new_features]
    # model.fit(X, y)

if __name__ == "__main__":
    main()