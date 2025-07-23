import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
from openai import OpenAI
import os
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class NewFeatures(BaseModel):
    premium_score: float
    necessity_score: float
    impulse_score: float
    cross_sell_score: float
    is_food: int
    price: float



class LLMFeatures:
    def __init__(self, max_workers=5):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    def extract_core_features(self, product_name, department, aisle) -> NewFeatures:
        """
        Extract only the core LLM features defined in NewFeatures
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
    
    def process_single_product(self, row):
        """Process a single product - used for threading"""
        features = self.extract_core_features(
            row['product_name'],
            row['department'], 
            row['aisle']
        )
        feature_dict = features.model_dump()
        feature_dict['product_id'] = row['product_id']
        return feature_dict
    
    def add_efficient_features(self, sample_size=None):
        """
        Extract only LLM features and save to llm_features.csv
        """
        print("Extracting LLM features...")
        
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
        
        # Get unique products for LLM processing
        unique_products = df_sample[['product_id', 'product_name', 'department', 'aisle']].drop_duplicates()
        
        print(f"Processing {len(unique_products)} unique products with {self.max_workers} threads...")
        
        # Multi-threaded LLM feature extraction
        semantic_features_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self.process_single_product, row): row 
                for idx, row in unique_products.iterrows()
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="LLM Processing"):
                try:
                    result = future.result()
                    semantic_features_list.append(result)
                except Exception as e:
                    row = future_to_row[future]
                    print(f"Error processing product {row['product_id']}: {e}")
        
        # Create DataFrame with LLM features
        llm_features_df = pd.DataFrame(semantic_features_list)
        
        # Save LLM features to CSV
        llm_features_df.to_csv('dataset/products_llm_features.csv', index=False)
        print(f"Saved {len(llm_features_df)} products with LLM features to 'dataset/products_llm_features.csv'")
        print("LLM features:", [col for col in llm_features_df.columns if col != 'product_id'])
        
        return llm_features_df
    
    def add_derived_features(self, df, llm_features_df=None):
        """
        Add derived features dynamically before training
        """
        print("Adding derived features...")
        
        # Load LLM features if not provided
        if llm_features_df is None:
            if os.path.exists('dataset/llm_features.csv'):
                llm_features_df = pd.read_csv('dataset/llm_features.csv')
                print("Loaded existing LLM features from llm_features.csv")
            else:
                raise FileNotFoundError("LLM features file not found. Run add_efficient_features() first.")
        
        # Merge with input dataframe
        df_enhanced = df.merge(llm_features_df, on='product_id', how='left')
        
        # Price-based features (3 features)
        print("Adding price features...")
        df_enhanced['price_percentile'] = df_enhanced.groupby('department_id')['price'].rank(pct=True)
        df_enhanced['is_premium_priced'] = (df_enhanced['price_percentile'] > 0.7).astype(int)
        df_enhanced['price_vs_dept_avg'] = df_enhanced['price'] / df_enhanced.groupby('department_id')['price'].transform('mean')
        
        # Derived semantic features (3 features)
        print("Adding derived semantic features...")
        df_enhanced['revenue_potential'] = df_enhanced['premium_score'] * df_enhanced['necessity_score']
        df_enhanced['basket_expansion_score'] = df_enhanced['cross_sell_score'] * (10 - df_enhanced['impulse_score'])
        df_enhanced['price_premium_alignment'] = df_enhanced['premium_score'] * df_enhanced['price_percentile']
        
        new_features = [
            'premium_score', 'necessity_score', 'impulse_score', 'cross_sell_score', 'is_food', 'price',
            'price_percentile', 'is_premium_priced', 'price_vs_dept_avg', 
            'revenue_potential', 'basket_expansion_score', 'price_premium_alignment'
        ]
        
        print(f"Added {len(new_features)} total features")
        print("All new features:", new_features)
        
        return df_enhanced
    
    def get_llm_feature_names(self):
        """Get list of LLM feature names"""
        return ['premium_score', 'necessity_score', 'impulse_score', 'cross_sell_score', 'is_food', 'price']
    
    def get_derived_feature_names(self):
        """Get list of derived feature names"""
        return [
            'price_percentile', 'is_premium_priced', 'price_vs_dept_avg', 
            'revenue_potential', 'basket_expansion_score', 'price_premium_alignment'
        ]
    
    def get_all_feature_names(self):
        """Get list of all feature names"""
        return self.get_llm_feature_names() + self.get_derived_feature_names()


# Usage:
def main():
    # Initialize with 5 concurrent threads
    feature_engineer = LLMFeatures(max_workers=10)
    
    # Step 1: Extract LLM features and save to llm_features.csv (test with small sample)
    llm_features_df = feature_engineer.add_efficient_features()
    
    # Step 2: Example of how to use derived features before training
    print("\n" + "="*50)
    print("Example: Adding derived features before training")
    
    # Load original products data
    products_df = pd.read_csv('dataset/products.csv')
    
    # Add all features dynamically
    df_with_all_features = feature_engineer.add_derived_features(products_df)
    
    # Save complete dataset
    df_with_all_features.to_csv('dataset/products_with_all_features.csv', index=False)
    print(f"Saved complete dataset with all features to 'dataset/products_with_all_features.csv'")


if __name__ == "__main__":
    main()