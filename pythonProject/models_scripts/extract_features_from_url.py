import numpy as np
import pandas as pd
import re
import tldextract
import numpy as np
from urllib.parse import urlparse

# Create a DataFrame
df = pd.read_csv('../parsed_datasets/balanced_urls_dataset.csv')

# Regex for detecting IP addresses
ip_regex = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

# Suspicious keywords to check
suspicious_keywords = ['login', 'secure', 'account', 'verify', 'password', 'update', 'bank',
                       'server', 'client', 'confirm', 'banking', 'click', 'lucky', 'bonus',
                       'suspend', 'dropbox', 'alibaba', 'admin', 'signout', 'review', 'billing',
                       'redirectme', 'redirect', 'recovery', 'giveaway', 'submit', 'resolution',
                       'restore', 'verification', 'webspace', 'webnode', 'required', 'webhostapp',
                       'wp', 'content', 'site', 'images', 'js', 'css', 'view']


# Function to calculate Shannon entropy
def calculate_entropy(url):
    probabilities = [float(url.count(c)) / len(url) for c in set(url)]
    entropy = -sum([p * np.log2(p) for p in probabilities])
    return entropy


# Function to extract features
def extract_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)

    # Structural Features
    features['url_length'] = len(url)
    features['num_special_chars'] = len(re.findall(r'[-@_=?.]', url))
    features['path_length'] = len(parsed_url.path)
    features['has_ip_address'] = 1 if re.search(ip_regex, parsed_url.netloc) else 0
    features['num_subdomains'] = domain_info.subdomain.count('.') + 1 if domain_info.subdomain else 0
    features['contains_suspicious_keywords'] = 1 if any(keyword in url for keyword in suspicious_keywords) else 0
    features['contains_port_number'] = 1 if ':' in parsed_url.netloc else 0

    # Lexical Features
    features['entropy'] = calculate_entropy(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_uppercase'] = sum(c.isupper() for c in url)
    features['domain_length'] = len(domain_info.domain)
    features['contains_hex_chars'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0
    features['non_alphanumeric_ratio'] = len(re.findall(r'[^a-zA-Z0-9]', url)) / len(url)
    features['vowel_consonant_ratio'] = 0 if len(domain_info.domain) == 0 else sum(c in 'aeiouAEIOU' for c in domain_info.domain) / len(domain_info.domain)

    # Domain-Based Features
    features['tld'] = domain_info.suffix
    features['tld_popularity'] = 1 if domain_info.suffix in ['com', 'org', 'net'] else 0

    # Statistical Features
    tokens = re.split(r'[./]', url)
    tokens = [t for t in tokens if t]
    features['num_tokens'] = len(tokens)
    features['avg_token_length'] = np.mean([len(token) for token in tokens]) if tokens else 0
    features['longest_token_length'] = max([len(token) for token in tokens]) if tokens else 0
    features['url_depth'] = len(parsed_url.path.split('/')) - 1

    # HTTPS Usage
    features['uses_https'] = 1 if parsed_url.scheme == 'https' else 0

    return features


# Function to extract and aggregate features from links
def extract_features_for_multiple_links(links):
    if not links:
        return {}

    features = [extract_url_features(link) for link in links]

    # Ensure all values are numeric and handle missing features
    numeric_features = [
        {key: (float(value) if isinstance(value, (int, float)) else 0.0) for key, value in f.items()}
        for f in features
    ]

    aggregated = {}

    # Aggregate features (mean, max, sum, etc.)
    for key in numeric_features[0]:
        aggregated[f"{key}_mean"] = np.mean([f[key] for f in numeric_features])
        aggregated[f"{key}_max"] = np.max([f[key] for f in numeric_features])
        aggregated[f"{key}_sum"] = np.sum([f[key] for f in numeric_features])

    return aggregated


# Apply feature extraction and aggregation
df_features = df['extracted_links'].apply(extract_features_for_multiple_links).apply(pd.Series)

# Combine aggregated features with the original dataset
df_combined = pd.concat([df, df_features], axis=1)
df_combined.to_csv('../parsed_datasets/urls_features_dataset.csv', index=False)