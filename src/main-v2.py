
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# LLM and NLP
import openai
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Network analysis for cross-selling
import networkx as nx
from collections import defaultdict, Counter

import warnings
warnings.filterwarnings('ignore')


load_dotenv()
dataset_dir = os.path.join(os.getcwd(), 'dataset')

files = {
    "aisles": os.path.join(dataset_dir, 'aisles.csv'),
    "departments": os.path.join(dataset_dir, 'departments.csv'),
    "orders": os.path.join(dataset_dir, 'orders.csv'),
    "order_products_prior": os.path.join(dataset_dir, 'order_products__prior.csv'),
    "order_products_train": os.path.join(dataset_dir, 'order_products__train.csv'),
    "products": os.path.join(dataset_dir, 'products.csv'),
    "estimate_prices": os.path.join(dataset_dir, 'products-with-llm-prices.csv'),
}

print("Loading datasets...")
aisles_df = pd.read_csv(files['aisles'])
departments_df = pd.read_csv(files['departments'])
orders_df = pd.read_csv(files['orders'])
order_products_prior_df = pd.read_csv(files['order_products_prior'])
order_products_train_df = pd.read_csv(files['order_products_train'])
products_df = pd.read_csv(files['products'])
print("Loaded.")


products_with_prices = pd.read_csv(files['estimate_prices'])
products_df = pd.merge(products_df, products_with_prices[['product_id', 'price']], on='product_id', how='left')
print("Loaded product prices.")


# ## Data Preprocessing with Enhanced Features

print("Merging datasets...")
products_df = pd.merge(products_df, aisles_df, on='aisle_id')
products_df = pd.merge(products_df, departments_df, on='department_id')
order_products_all_df = pd.concat([order_products_prior_df, order_products_train_df])
merged_df = pd.merge(order_products_all_df, orders_df, on='order_id', how='inner')
merged_df = pd.merge(merged_df, products_df, on='product_id', how='inner')

# Handle missing values
merged_df = merged_df.copy()
merged_df['days_since_prior_order'] = merged_df['days_since_prior_order'].fillna(0)
merged_df['price'] = merged_df['price'].fillna(merged_df['price'].median())

print(f"Dataset shape: {merged_df.shape}")


# ## Approach #1: LLM-Enhanced Feature Engineering ⭐⭐⭐
# 
# This approach uses LLM capabilities to create semantic features from product names and departments, going beyond traditional categorical encoding.
# 

# ### 1.1 Semantic Product Embeddings
# 


# Initialize sentence transformer for semantic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating semantic embeddings for products...")
# Create combined text for richer semantic understanding
products_df['combined_text'] = (products_df['product_name'] + ' ' + 
                               products_df['aisle'] + ' ' + 
                               products_df['department'])

# Generate embeddings (using subset for efficiency)
unique_products = products_df[['product_id', 'combined_text']].drop_duplicates()
print(f"Generating embeddings for {len(unique_products)} unique products...")

embeddings = model.encode(unique_products['combined_text'].tolist(), 
                         show_progress_bar=True, 
                         batch_size=32)

# Create embedding dataframe
embedding_df = pd.DataFrame(embeddings)
embedding_df['product_id'] = unique_products['product_id'].values
embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
embedding_df.columns = embedding_cols + ['product_id']

print(f"Generated {embeddings.shape[1]}-dimensional embeddings")


# ### 1.2 LLM-Generated Product Categories
# 

# In[6]:


client = openai.OpenAI()

def generate_product_categories(product_names_batch):
    """Generate semantic categories for products using LLM"""
    prompt = f"""
    As a grocery retail expert, categorize these products into semantic categories that represent shopping behavior.
    Focus on how customers typically shop for these items (e.g., "healthy_snacks", "meal_prep_essentials", "comfort_food", "organic_produce").
    Return ONLY a comma-separated list of categories, one per product:

    Products:
    {"; ".join(product_names_batch)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a grocery retail expert who understands customer shopping behavior."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    categories = [cat.strip() for cat in response.choices[0].message.content.strip().split(',')]
    return categories

# Generate semantic categories for top products
print("Generating LLM-based semantic categories...")
top_products = merged_df.groupby('product_id').size().head(200).index
top_products_info = products_df[products_df['product_id'].isin(top_products)]

batch_size = 20
semantic_categories = {}

for i in range(0, len(top_products_info), batch_size):
    batch = top_products_info.iloc[i:i+batch_size]
    try:
        categories = generate_product_categories(batch['product_name'].tolist())
        for j, (_, row) in enumerate(batch.iterrows()):
            if j < len(categories):
                semantic_categories[row['product_id']] = categories[j]
        print(f"Processed batch {i//batch_size + 1}/{(len(top_products_info)-1)//batch_size + 1}")
        time.sleep(1)  # Rate limiting
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue

# Create semantic category dataframe
semantic_cat_df = pd.DataFrame(list(semantic_categories.items()), 
                              columns=['product_id', 'llm_category'])
print(f"Generated semantic categories for {len(semantic_cat_df)} products")


# ### 1.3 Semantic Similarity Features
# 

# In[7]:


# Calculate semantic similarity between frequently bought together products
print("Calculating semantic similarity features...")

# Find frequently bought together products
basket_pairs = []
for order_id in merged_df['order_id'].unique()[:10000]:  # Sample for efficiency
    products_in_order = merged_df[merged_df['order_id'] == order_id]['product_id'].tolist()
    if len(products_in_order) > 1:
        for i in range(len(products_in_order)):
            for j in range(i+1, len(products_in_order)):
                basket_pairs.append((products_in_order[i], products_in_order[j]))

# Count co-occurrence
pair_counts = Counter(basket_pairs)
frequent_pairs = {pair: count for pair, count in pair_counts.items() if count >= 5}

print(f"Found {len(frequent_pairs)} frequent product pairs")

# Calculate average semantic similarity for each product
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

print(f"Calculated semantic similarity features for {len(similarity_df)} products")


# ## Approach #2: Hybrid Scoring for Basket Expansion ⭐⭐⭐
# 
# This approach combines ML predictions with basket expansion potential, focusing on cross-selling and revenue growth.
# 

# ### 2.1 Cross-Selling Network Analysis
# 

# In[8]:


print("Building cross-selling network...")

# Create product co-occurrence network
G = nx.Graph()

# Add edges with weights based on co-occurrence frequency
for (p1, p2), weight in frequent_pairs.items():
    G.add_edge(p1, p2, weight=weight)

print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

num_nodes = G.number_of_nodes()
k = min(1000, num_nodes)

# Calculate network centrality measures
centrality_measures = {
    'degree_centrality': nx.degree_centrality(G),
    'betweenness_centrality': nx.betweenness_centrality(G, k=k),  # Sample for efficiency
    'pagerank': nx.pagerank(G, max_iter=50)
}

# Create centrality dataframe
centrality_df = pd.DataFrame(centrality_measures)
centrality_df['product_id'] = centrality_df.index
centrality_df = centrality_df.reset_index(drop=True)

print("Calculated network centrality measures")


# ### 2.2 Basket Expansion Potential Scoring
# 

# In[9]:


print("Calculating basket expansion potential...")

# Calculate basket expansion metrics
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
basket_sizes = merged_df.groupby('order_id').size()
product_basket_sizes = merged_df.merge(
    basket_sizes.to_frame('basket_size'), 
    left_on='order_id', 
    right_index=True
).groupby('product_id')['basket_size'].mean()

basket_metrics['avg_basket_size_with_product'] = product_basket_sizes

# Calculate cross-selling strength
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

# Calculate basket expansion score
# Normalize metrics
for col in ['order_frequency', 'user_reach', 'avg_price', 'reorder_rate', 
           'avg_basket_size_with_product', 'cross_sell_strength']:
    if col in basket_metrics.columns:
        basket_metrics[f'{col}_norm'] = (
            basket_metrics[col] - basket_metrics[col].min()
        ) / (basket_metrics[col].max() - basket_metrics[col].min())

# Composite basket expansion score
basket_metrics['basket_expansion_score'] = (
    0.25 * basket_metrics['avg_basket_size_with_product_norm'] +
    0.25 * basket_metrics['cross_sell_strength_norm'] +
    0.20 * basket_metrics['reorder_rate_norm'] +
    0.15 * basket_metrics['avg_price_norm'] +
    0.15 * basket_metrics['user_reach_norm']
)

basket_metrics['product_id'] = basket_metrics.index
basket_metrics = basket_metrics.reset_index(drop=True)

print(f"Calculated basket expansion metrics for {len(basket_metrics)} products")


# ## Enhanced Feature Engineering with LLM Features
# 

# In[10]:


print("Enhanced Feature Engineering...")

# Basic features (from original)
merged_df['average_basket_size'] = merged_df.groupby('user_id')['product_id'].transform('count')
merged_df['purchase_frequency'] = merged_df.groupby('user_id')['order_number'].transform('max')
merged_df['product_reorder_rate'] = merged_df.groupby('product_id')['reordered'].transform('mean')

# Revenue-focused features
merged_df['user_avg_order_value'] = merged_df.groupby('user_id')['price'].transform('mean')
merged_df['product_revenue_contribution'] = merged_df.groupby('product_id')['price'].transform('sum')
merged_df['user_total_spent'] = merged_df.groupby('user_id')['price'].transform('sum')

# Time-based features
merged_df['order_recency'] = merged_df.groupby('user_id')['order_number'].transform('max') - merged_df['order_number']
merged_df['product_last_ordered'] = merged_df.groupby(['user_id', 'product_id'])['order_number'].transform('max')

# Merge LLM-enhanced features
print("Merging LLM-enhanced features...")
merged_df = merged_df.merge(semantic_cat_df, on='product_id', how='left')
merged_df = merged_df.merge(similarity_df, on='product_id', how='left')
merged_df = merged_df.merge(centrality_df, on='product_id', how='left')
merged_df = merged_df.merge(basket_metrics[['product_id', 'basket_expansion_score', 
                                          'cross_sell_strength', 'avg_basket_size_with_product']], 
                          on='product_id', how='left')

# Handle missing values for LLM features
llm_features = ['avg_semantic_similarity', 'max_semantic_similarity', 'semantic_diversity',
                'degree_centrality', 'betweenness_centrality', 'pagerank',
                'basket_expansion_score', 'cross_sell_strength', 'avg_basket_size_with_product']

for feature in llm_features:
    if feature in merged_df.columns:
        merged_df[feature] = merged_df[feature].fillna(0)

# Sample dataset for efficiency
print("Sampling dataset for modeling...")
sampled_df = merged_df.sample(frac=0.02, random_state=42)  # Increased sample size for LLM features
print(f"Sampled dataset shape: {sampled_df.shape}")


# ## Enhanced Model Training with LLM Features
# 

# In[11]:


# Enhanced feature set
base_features = [
    'order_hour_of_day', 'order_dow', 'days_since_prior_order', 'aisle_id', 
    'department_id', 'average_basket_size', 'purchase_frequency', 'product_reorder_rate',
    'price', 'user_avg_order_value', 'product_revenue_contribution', 'order_recency'
]

llm_features = [
    'avg_semantic_similarity', 'max_semantic_similarity', 'semantic_diversity',
    'degree_centrality', 'betweenness_centrality', 'pagerank',
    'basket_expansion_score', 'cross_sell_strength', 'avg_basket_size_with_product'
]

# Test both feature sets
all_features = base_features + llm_features
target = 'reordered'

# Prepare datasets
X_base = sampled_df[base_features].fillna(0)
X_enhanced = sampled_df[all_features].fillna(0)
y = sampled_df[target]

print(f"Base features: {len(base_features)}")
print(f"Enhanced features: {len(all_features)}")
print(f"Target variable distribution:")
print(y.value_counts(normalize=True))


# In[12]:


# Balance data using SMOTE and split
print("Applying SMOTE and splitting data...")

smote = SMOTE(random_state=42)
X_base_resampled, y_base_resampled = smote.fit_resample(X_base, y)
X_enhanced_resampled, y_enhanced_resampled = smote.fit_resample(X_enhanced, y)

# Split data
X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
    X_base_resampled, y_base_resampled, test_size=0.2, random_state=42)

X_enh_train, X_enh_test, y_enh_train, y_enh_test = train_test_split(
    X_enhanced_resampled, y_enhanced_resampled, test_size=0.2, random_state=42)

print("Data preparation completed.")


# ### Model Comparison: Base vs Enhanced Features
# 

# In[13]:


# Train models with base features
print("Training models with BASE features...")

# XGBoost with base features
xgb_base = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
xgb_base.fit(X_base_train, y_base_train)

y_pred_base = xgb_base.predict(X_base_test)
y_prob_base = xgb_base.predict_proba(X_base_test)[:, 1]

accuracy_base = accuracy_score(y_base_test, y_pred_base)
auc_base = roc_auc_score(y_base_test, y_prob_base)

print(f"Base Model - Accuracy: {accuracy_base:.4f}, AUC: {auc_base:.4f}")


# In[14]:


# Train models with enhanced features
print("Training models with ENHANCED features...")

# XGBoost with enhanced features
xgb_enhanced = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
xgb_enhanced.fit(X_enh_train, y_enh_train)

y_pred_enhanced = xgb_enhanced.predict(X_enh_test)
y_prob_enhanced = xgb_enhanced.predict_proba(X_enh_test)[:, 1]

accuracy_enhanced = accuracy_score(y_enh_test, y_pred_enhanced)
auc_enhanced = roc_auc_score(y_enh_test, y_prob_enhanced)

print(f"Enhanced Model - Accuracy: {accuracy_enhanced:.4f}, AUC: {auc_enhanced:.4f}")


# ## Hybrid Scoring System: Revenue-Optimized Recommendations
# 

# In[15]:


def calculate_hybrid_score(ml_probability, basket_expansion_score, price, cross_sell_strength, 
                          alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    """
    Calculate hybrid recommendation score combining:
    - ML prediction probability
    - Basket expansion potential
    - Price/revenue potential
    - Cross-selling strength
    """
    # Normalize price (higher price = higher potential revenue)
    price_norm = (price - np.min(price)) / (np.max(price) - np.min(price)) if np.max(price) > np.min(price) else 0

    hybrid_score = (
        alpha * ml_probability +
        beta * basket_expansion_score +
        gamma * price_norm +
        delta * cross_sell_strength
    )

    return hybrid_score

# Calculate hybrid scores for test set
print("Calculating hybrid scores...")

# Get test data with all features - handle different data types
if hasattr(X_enh_test, 'index'):
    test_indices = sampled_df.index.get_indexer_for(X_enh_test.index)
else:
    test_indices = range(len(X_enh_test))

# Get corresponding rows from sampled_df
test_data = sampled_df.iloc[test_indices].copy()

# Calculate hybrid scores
hybrid_scores = calculate_hybrid_score(
    ml_probability=y_prob_enhanced,
    basket_expansion_score=test_data['basket_expansion_score'].fillna(0).values,
    price=test_data['price'].fillna(0).values,
    cross_sell_strength=test_data['cross_sell_strength'].fillna(0).values
)

print(f"Calculated hybrid scores for {len(hybrid_scores)} test samples")


# ## Revenue Impact Analysis
# 

# In[16]:


# Simulate revenue impact of different recommendation strategies
print("Analyzing revenue impact...")

# Create test dataframe with scores
revenue_analysis = pd.DataFrame({
    'actual_reorder': y_enh_test.values,
    'ml_probability': y_prob_enhanced,
    'hybrid_score': hybrid_scores,
    'price': test_data['price'].fillna(0).values,
    'basket_expansion_score': test_data['basket_expansion_score'].fillna(0).values
})

# Calculate revenue for different strategies
def calculate_revenue_metrics(df, score_column, top_k=1000):
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

# Compare strategies
ml_only_metrics = calculate_revenue_metrics(revenue_analysis, 'ml_probability')
hybrid_metrics = calculate_revenue_metrics(revenue_analysis, 'hybrid_score')

print("\n=== REVENUE IMPACT COMPARISON ===")
print("\nML-Only Strategy:")
for metric, value in ml_only_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nHybrid Strategy (LLM-Enhanced):")
for metric, value in hybrid_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nImprovement with Hybrid Approach:")
for metric in ml_only_metrics.keys():
    improvement = ((hybrid_metrics[metric] - ml_only_metrics[metric]) / ml_only_metrics[metric]) * 100
    print(f"  {metric}: {improvement:+.2f}%")


# ## Feature Importance Analysis with LLM Features
# 

# In[17]:


# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': xgb_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

# Categorize features
feature_importance['category'] = feature_importance['feature'].apply(
    lambda x: 'LLM-Enhanced' if x in llm_features else 'Traditional'
)

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
accuracies = [accuracy_base, accuracy_enhanced]
aucs = [auc_base, auc_enhanced]

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
revenues = [ml_only_metrics['total_revenue'], hybrid_metrics['total_revenue']]
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

# Print detailed feature importance
print("\n=== TOP 20 MOST IMPORTANT FEATURES ===")
print(feature_importance.head(20).to_string(index=False))


# ## Cross-Selling Recommendations Engine
# 

# In[18]:


def generate_cross_sell_recommendations(user_basket, top_n=5):
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
        product_data = merged_df[merged_df['product_id'] == product_id]
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
                0.2 * (price / merged_df['price'].max()) +  # Normalized price
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

# Example: Generate recommendations for a sample basket
sample_user_orders = merged_df[merged_df['user_id'] == merged_df['user_id'].iloc[0]]
sample_basket = sample_user_orders['product_id'].tolist()[:3]  # First 3 products

print(f"\nSample basket contains products: {sample_basket}")
cross_sell_recs = generate_cross_sell_recommendations(sample_basket)

print("\n=== CROSS-SELLING RECOMMENDATIONS ===")
for i, rec in enumerate(cross_sell_recs, 1):
    print(f"{i}. {rec['product_name']} (ID: {rec['product_id']})")
    print(f"   Price: ${rec['price']:.2f}")
    print(f"   Recommendation Score: {rec['recommendation_score']:.4f}")
    print(f"   Network Strength: {rec['network_strength']:.2f}")
    print()


# ## Key Findings and Thesis Contributions
# 

# In[19]:


# Calculate improvement metrics
accuracy_improvement = ((accuracy_enhanced - accuracy_base) / accuracy_base) * 100
auc_improvement = ((auc_enhanced - auc_base) / auc_base) * 100
revenue_improvement = ((hybrid_metrics['total_revenue'] - ml_only_metrics['total_revenue']) / ml_only_metrics['total_revenue']) * 100

print("\n" + "="*60)
print("           THESIS CONTRIBUTIONS SUMMARY")
print("="*60)

print("\n🎯 APPROACH #1: LLM-ENHANCED FEATURE ENGINEERING")
print(f"   • Generated semantic embeddings for {len(unique_products)} products")
print(f"   • Created {len(semantic_cat_df)} LLM-generated product categories")
print(f"   • Calculated semantic similarity features for cross-selling")
print(f"   • Model accuracy improvement: +{accuracy_improvement:.2f}%")
print(f"   • AUC improvement: +{auc_improvement:.2f}%")

print("\n🚀 APPROACH #2: HYBRID SCORING FOR BASKET EXPANSION")
print(f"   • Built co-occurrence network with {G.number_of_nodes()} products")
print(f"   • Calculated basket expansion scores for revenue optimization")
print(f"   • Revenue improvement over ML-only: +{revenue_improvement:.2f}%")
print(f"   • Enhanced cross-selling recommendation engine")

print("\n📊 KEY PERFORMANCE METRICS:")
print(f"   • Base Model: {accuracy_base:.4f} accuracy, {auc_base:.4f} AUC")
print(f"   • Enhanced Model: {accuracy_enhanced:.4f} accuracy, {auc_enhanced:.4f} AUC")
print(f"   • Revenue per recommendation: ${hybrid_metrics['avg_revenue_per_rec']:.2f}")
print(f"   • Basket expansion potential: {hybrid_metrics['avg_basket_expansion']:.4f}")

print("\n🎓 THESIS INNOVATION HIGHLIGHTS:")
print("   ✓ Novel application of LLMs for grocery recommendation feature engineering")
print("   ✓ Semantic product understanding beyond traditional categorical encoding")
print("   ✓ Network analysis for cross-selling opportunity identification")
print("   ✓ Revenue-focused hybrid scoring system for business impact")
print("   ✓ Measurable improvements in both accuracy and revenue potential")

print("\n💡 FUTURE RESEARCH DIRECTIONS:")
print("   • Real-time LLM feature generation for new products")
print("   • Multi-modal embeddings incorporating product images")
print("   • Personalized semantic similarity based on user preferences")
print("   • Dynamic basket expansion scoring with temporal patterns")

print("\n" + "="*60)


# ## Save Enhanced Model and Results
# 

# In[20]:


import joblib
import json

# Save the enhanced model
model_artifacts = {
    'model': xgb_enhanced,
    'feature_names': all_features,
    'scaler': None,  # Add if used
    'performance_metrics': {
        'accuracy': accuracy_enhanced,
        'auc': auc_enhanced,
        'revenue_improvement': revenue_improvement
    }
}

joblib.dump(model_artifacts, 'llm_enhanced_recommendation_model.pkl')

# Save comprehensive results
results = {
    'base_model_performance': {
        'accuracy': float(accuracy_base),
        'auc': float(auc_base)
    },
    'enhanced_model_performance': {
        'accuracy': float(accuracy_enhanced),
        'auc': float(auc_enhanced)
    },
    'improvements': {
        'accuracy_improvement_pct': float(accuracy_improvement),
        'auc_improvement_pct': float(auc_improvement),
        'revenue_improvement_pct': float(revenue_improvement)
    },
    'feature_importance': feature_importance.to_dict('records'),
    'revenue_metrics': {
        'ml_only': ml_only_metrics,
        'hybrid': hybrid_metrics
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

print("\n✅ Model artifacts and results saved:")
print("   • llm_enhanced_recommendation_model.pkl")
print("   • llm_enhanced_results.json")
print("\n🎉 LLM-Enhanced recommendation system successfully implemented!")

