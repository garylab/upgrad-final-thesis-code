import os
import pandas as pd
import openai
from dotenv import load_dotenv
output_path = "dataset/products-with-llm-prices.csv"
df = pd.read_csv("dataset/products.csv")

batch_size = 500

load_dotenv()
client = openai.OpenAI()


def estimate_prices_batch(product_names):
    prompt = (
        "You are an e-commerce expert. Estimate the price in USD for each product below. "
        "Return only a numbered list of prices, one per line, no explanations. "
        "Products:\n" +
        "\n".join([f"{i+1}. {name}" for i, name in enumerate(product_names)])
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an e-commerce expert estimating reasonable product prices in USD."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    # Parse the response into a list of floats
    prices = []
    for line in response.choices[0].message.content.strip().split('\n'):
        price_str = line.split('.', 1)[-1].strip().replace('$', '').replace(',', '')
        try:
            prices.append(float(price_str))
        except ValueError:
            prices.append(0.0)
            
    return prices


def estimate_all_prices():
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        start_index = len(done_df)
        print(f"Resuming from index {start_index}")
    else:
        start_index = 0

    for i in range(start_index, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batch["price"] = estimate_prices_batch(batch["product_name"].tolist())
        
        batch.to_csv(
            output_path,
            mode='a',
            header=not os.path.exists(output_path) and i == 0,  # Only write header if file doesn't exist
            index=False
        )
        
        print(f"Appended batch {i} to {i + batch_size}")


if __name__ == "__main__":
    estimate_all_prices()