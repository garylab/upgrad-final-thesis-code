import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
from openai import OpenAI
import os
from pydantic import BaseModel


class NewFeatures(BaseModel):
    premium_score: float
    necessity_score: float
    impulse_score: float
    cross_sell_score: float
    is_food: int
    price: float



class LLMFeatures:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_core_features(self, product_name, department, aisle) -> NewFeatures:
        """
        Extract only the 5 most impactful semantic features + price
        """
        prompt = f"""
        Product: {product_name}
        Department: {department}
        Aisle: {aisle}

        Extract these 6 features and return as JSON:

        1. premium_score: How premium/high-quality is this? (0=basic/generic, 10=premium/brand)
        2. necessity_score: How essential/necessary? (0=luxury/optional, 10=daily necessity)
        3. impulse_score: Impulse purchase likelihood? (0=planned purchase, 10=impulse buy)
        4. cross_sell_score: Likely to trigger more purchases? (0=standalone, 10=leads to many items)
        5. is_food: 1 if the product is food, 0 otherwise
        6. price: Estimated price in USD (realistic grocery store price)

        Return only valid JSON.
        """
        
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an e-commerce expert estimating product features and prices."},
                    {"role": "user", "content": prompt}
                ],
                response_format=NewFeatures
            )
            
            if response.choices[0].message.parsed is None:
                print(f"Warning: Failed to parse response for {product_name}")
                print(f"Raw response: {response.choices[0].message.content}")
                # Return default values
                return NewFeatures(
                    premium_score=5.0,
                    necessity_score=5.0, 
                    impulse_score=5.0,
                    cross_sell_score=5.0,
                    is_food=0,
                    price=3.99
                )
            
            return response.choices[0].message.parsed
            
        except Exception as e:
            print(f"Error processing {product_name}: {e}")
            # Return default values
            return NewFeatures(
                premium_score=5.0,
                necessity_score=5.0, 
                impulse_score=5.0,
                cross_sell_score=5.0,
                is_food=0,
                price=3.99
            )
    
    
    
    def add_efficient_features(self, sample_size=None):
        """
        Add only the most efficient features for maximum impact
        """
        print("Adding efficient LLM features...")
        
        df = pd.read_csv('dataset/products.csv')
        if sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        # Load and merge department and aisle data
        print("Loading department and aisle data...")
        departments_df = pd.read_csv('dataset/departments.csv')
        aisles_df = pd.read_csv('dataset/aisles.csv')
        
        # Merge with the main dataframe
        df_sample = df_sample.merge(departments_df, on='department_id', how='left')
        df_sample = df_sample.merge(aisles_df, on='aisle_id', how='left')
        
        # Step 1: LLM semantic features (including price estimation)
        print("Extracting LLM features (including price)...")
        unique_products = df_sample[['product_id', 'product_name', 'department', 'aisle']].drop_duplicates()
        
        semantic_features_list = []
        for idx, row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Processing"):
            features = self.extract_core_features(
                row['product_name'],
                row['department'], 
                row['aisle']
            )
            # Create dictionary with product_id and features
            feature_dict = features.model_dump()
            feature_dict['product_id'] = row['product_id']
            semantic_features_list.append(feature_dict)
        
        semantic_df = pd.DataFrame(semantic_features_list)
        df_enhanced = df_sample.merge(semantic_df, on='product_id', how='left')
        
        # Step 2: Core price features using LLM-extracted price (3 features)
        print("Adding price features using LLM price...")
        df_enhanced['price_percentile'] = df_enhanced.groupby('department_id')['price'].rank(pct=True)
        df_enhanced['is_premium_priced'] = (df_enhanced['price_percentile'] > 0.7).astype(int)
        df_enhanced['price_vs_dept_avg'] = df_enhanced['price'] / df_enhanced.groupby('department_id')['price'].transform('mean')
        
        # Step 3: Create 3 high-impact derived features
        print("Creating derived features...")
        # Revenue potential = premium score × necessity (premium necessities = high revenue)
        df_enhanced['revenue_potential'] = df_enhanced['premium_score'] * df_enhanced['necessity_score']
        
        # Basket expansion = cross-sell × impulse (items that lead to bigger baskets)
        df_enhanced['basket_expansion_score'] = df_enhanced['cross_sell_score'] * (10 - df_enhanced['impulse_score'])
        
        # Price-premium alignment = LLM price vs actual department percentile
        df_enhanced['price_premium_alignment'] = df_enhanced['premium_score'] * df_enhanced['price_percentile']
        
        print(f"Added {len(df_enhanced.columns) - len(df.columns)} efficient features")
        print("New features:", [col for col in df_enhanced.columns if col not in df.columns])

        return df_enhanced
    
    def get_feature_list(self):
        """Return list of all new features created"""
        return [
            # Text categorical features from merging (2)
            'department', 'aisle',
            
            # Price features (3)
            'price_percentile', 'is_premium_priced', 'price_vs_dept_avg',
            
            # LLM semantic features (5 - including extracted price)
            'premium_score', 'necessity_score', 'impulse_score', 'cross_sell_score', 'price',
            
            # Derived features (3)
            'revenue_potential', 'basket_expansion_score', 'price_premium_alignment',
            
            # Categorical (1)
            'is_food'
        ]

# Usage:
def main():
    # Initialize
    feature_engineer = LLMFeatures()
    
    df_enhanced = feature_engineer.add_efficient_features(sample_size=5)

    df_enhanced.to_csv('dataset/products_enhanced_5.csv', index=False)
    # Get list of new features for your ML models
    new_features = feature_engineer.get_feature_list()
    print(f"Total new features: {len(new_features)}")

    # Use in your Random Forest/XGBoost
    # X = df_enhanced[original_features + new_features]
    # model.fit(X, y)

if __name__ == "__main__":
    main()