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