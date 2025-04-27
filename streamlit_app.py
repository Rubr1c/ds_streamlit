import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import re
from urllib.parse import urlparse, urlsplit
import tldextract  # You might need to install this: pip install tldextract
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide", page_title="Cybersecurity Analytics Dashboard")
st.title("KH5004CMD: Cybersecurity Analytics Dashboard")

ATTACKS_DATA_PATH = 'datasets/cybersecurity_attacks.csv'
PHISHING_DATA_PATH = 'datasets/PhiUSIIL_Phishing_URL_Dataset.csv'
MODEL_PATH = 'phishing_hybrid_model.joblib'
SCALER_PATH = 'scaler_hybrid.joblib'

FINAL_MODEL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'NoOfSubDomain', 'TLDLength',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
    'HasFavicon', 'HasCopyrightInfo', 'HasSubmitButton', 'HasPasswordField'
]

URL_DERIVED_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'NoOfSubDomain', 'TLDLength',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
]

MANUAL_INPUT_FEATURES = [f for f in FINAL_MODEL_FEATURES if f not in URL_DERIVED_FEATURES]

@st.cache_data
def load_data(filepath):
    """Loads data from CSV, handling potential errors."""
    if not os.path.exists(filepath):
        st.error(f"Error: Data file not found at {filepath}")
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {e}")
        return None

@st.cache_resource
def load_model_or_scaler(filepath):
    """Loads a pickled model or scaler, handling potential errors."""
    if not os.path.exists(filepath):
        st.error(f"Error: File not found at {filepath}")
        return None
    try:
        return load(filepath)
    except Exception as e:
        st.error(f"Error loading file from {filepath}: {e}")
        return None

def get_domain_features(url):
    """Extracts domain-related features from URL."""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''

        extracted = tldextract.extract(url)
        domain = extracted.domain + '.' + extracted.suffix if extracted.domain and extracted.suffix else ''
        subdomains = extracted.subdomain.split('.') if extracted.subdomain else []
        no_of_subdomain = len(subdomains) if subdomains != [''] else 0

        domain_length = len(domain)
        is_domain_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname) else 0
        tld_length = len(extracted.suffix)

        return domain, domain_length, no_of_subdomain, is_domain_ip, tld_length
    except Exception:
        return '', 0, 0, 0, 0

def count_chars(text):
    """Counts different character types in a string."""
    no_of_letters = sum(c.isalpha() for c in text)
    no_of_digits = sum(c.isdigit() for c in text)
    no_of_equals = text.count('=')
    no_of_qmark = text.count('?')
    no_of_ampersand = text.count('&')
    special_chars_pattern = r'[~!@#$%^&*()_+`\-={}\[\]|\\:\";\'<>,./]' # Example set
    no_of_other_special = len(re.findall(special_chars_pattern, text)) - no_of_equals - no_of_qmark - no_of_ampersand
    return no_of_letters, no_of_digits, no_of_equals, no_of_qmark, no_of_ampersand, no_of_other_special

def calculate_url_features(url):
    """Calculates all URL-derived features required by the model."""
    features = {}
    url = url.strip()
    url_length = len(url)
    features['URLLength'] = url_length

    domain, domain_length, no_of_subdomain, is_domain_ip, tld_length = get_domain_features(url)
    features['DomainLength'] = domain_length
    features['NoOfSubDomain'] = no_of_subdomain
    features['IsDomainIP'] = is_domain_ip
    features['TLDLength'] = tld_length

    no_of_letters, no_of_digits, no_of_equals, no_of_qmark, no_of_ampersand, no_of_other_special = count_chars(url)
    features['NoOfLettersInURL'] = no_of_letters
    features['NoOfDegitsInURL'] = no_of_digits
    features['NoOfEqualsInURL'] = no_of_equals
    features['NoOfQMarkInURL'] = no_of_qmark
    features['NoOfAmpersandInURL'] = no_of_ampersand
    features['NoOfOtherSpecialCharsInURL'] = no_of_other_special

    features['LetterRatioInURL'] = no_of_letters / url_length if url_length > 0 else 0
    features['DegitRatioInURL'] = no_of_digits / url_length if url_length > 0 else 0
    total_special_chars = no_of_equals + no_of_qmark + no_of_ampersand + no_of_other_special
    features['SpacialCharRatioInURL'] = total_special_chars / url_length if url_length > 0 else 0

    features['IsHTTPS'] = 1 if url.startswith('https://') else 0

    return features

df_attacks = load_data(ATTACKS_DATA_PATH)
df_phishing = load_data(PHISHING_DATA_PATH)
model = load_model_or_scaler(MODEL_PATH)
scaler = load_model_or_scaler(SCALER_PATH)

if model is None or scaler is None:
    st.error("Critical component (model or scaler) failed to load. Prediction functionality disabled.")

tab1, tab2, tab3 = st.tabs([
    "Cybersecurity Attacks EDA",
    "Phishing URLs EDA",
    "Phishing Prediction"
])

with tab1:
    st.header("Exploratory Data Analysis: Cybersecurity Attacks")
    if df_attacks is not None:
        st.subheader("Dataset Sample")
        st.dataframe(df_attacks.head())

        st.subheader("Key Visualizations")

        if 'Geo-location Data' in df_attacks.columns and 'Attack Type' in df_attacks.columns:
            df_attacks['Geo_Cleaned'] = df_attacks['Geo-location Data'].fillna('Unknown')
            df_attacks['AttackType_Cleaned'] = df_attacks['Attack Type'].fillna('Unknown')

            top_n_locations = 15
            top_locations = df_attacks['Geo_Cleaned'].value_counts().head(top_n_locations).index
            df_attacks_top_geo = df_attacks[df_attacks['Geo_Cleaned'].isin(top_locations)]
            st.write(f"Focusing analysis on the top {top_n_locations} most frequent locations.")

            if not df_attacks_top_geo.empty:
                attack_geo_ct = pd.crosstab(df_attacks_top_geo['Geo_Cleaned'], df_attacks_top_geo['AttackType_Cleaned'])
                st.write(f"--- Attack Type vs Geo-location Crosstab (Top {top_n_locations} Locations) ---")
                st.dataframe(attack_geo_ct)

                # Distribution of Attack Types 
                fig_attack_dist, ax_attack_dist = plt.subplots(figsize=(10, 6))
                attack_counts = df_attacks['Attack Type'].fillna('Missing').value_counts()
                sns.barplot(x=attack_counts.values, y=attack_counts.index, palette='viridis', ax=ax_attack_dist)
                ax_attack_dist.set_title('Distribution of Attack Types')
                ax_attack_dist.set_xlabel('Count')
                ax_attack_dist.set_ylabel('Attack Type')
                plt.tight_layout()
                st.pyplot(fig_attack_dist)

                # General distribution: Top 15 geo-locations by overall attack count
                fig_bar_general, ax_bar_general = plt.subplots(figsize=(10, 6))
                top_locations_general = df_attacks['Geo_Cleaned'].value_counts().head(15)
                sns.barplot(x=top_locations_general.values, y=top_locations_general.index, palette='magma', ax=ax_bar_general)
                ax_bar_general.set_title('Top 15 Geo-locations by Overall Attack Count')
                ax_bar_general.set_xlabel('Count')
                ax_bar_general.set_ylabel('Geo-location')
                plt.tight_layout()
                st.pyplot(fig_bar_general)

            else:
                st.warning(f"No data available for the top {top_n_locations} locations.")
        else:
            st.warning("Required columns ('Geo-location Data', 'Attack Type') not available for RQ2 analysis.")

    else:
        st.warning("Attacks dataset could not be loaded.")

with tab2:
    st.header("Exploratory Data Analysis: Phishing URLs")
    if df_phishing is not None:
        st.subheader("Dataset Sample")
        st.dataframe(df_phishing.head())

        st.subheader("Key Visualizations (RQ3 Focus)")

        if 'label' in df_phishing.columns:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            label_counts = df_phishing['label'].value_counts()
            sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax3, palette="coolwarm")
            ax3.set_title('Distribution of Labels (0: Legitimate, 1: Phishing)')
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(['Legitimate', 'Phishing'])
            ax3.set_ylabel('Frequency')
            st.pyplot(fig3)
        else:
             st.warning("Column 'label' not found.")

        # Replace URLLength distribution with LetterRatioInURL
        if 'LetterRatioInURL' in df_phishing.columns and 'label' in df_phishing.columns:
            fig_ratio, ax_ratio = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df_phishing, x='LetterRatioInURL', hue='label', kde=True, palette='viridis', bins=40, ax=ax_ratio)
            ax_ratio.set_title('Letter Ratio in URL Distribution by Label (RQ3)')
            ax_ratio.set_xlabel('Letter Ratio in URL')
            ax_ratio.set_ylabel('Frequency')
            ax_ratio.legend(title='Label', labels=['Phishing (1)', 'Legitimate (0)'])
            plt.tight_layout()
            st.pyplot(fig_ratio)
        else:
            st.warning("Columns 'LetterRatioInURL' or 'label' not found.")


    else:
        st.warning("Phishing dataset could not be loaded.")



with tab3:
    st.header("Predict Phishing URL")

    if model is None or scaler is None:
        st.error("Prediction unavailable because the model or scaler could not be loaded.")
    else:
        st.markdown("Enter the URL and other required features to predict if it's phishing.")

        url_input = st.text_input("Enter the full URL (including http:// or https://):", placeholder="e.g., https://www.google.com")

        manual_inputs = {}
        st.subheader("Enter Non-URL Based Features:")
        cols = st.columns(len(MANUAL_INPUT_FEATURES))
        col_idx = 0
        for feature in MANUAL_INPUT_FEATURES:
            with cols[col_idx % len(MANUAL_INPUT_FEATURES)]:
                 manual_inputs[feature] = st.selectbox(f"{feature} (0=No, 1=Yes):", options=[0, 1], key=feature)
            col_idx += 1


        predict_button = st.button("Predict", type="primary")

        if predict_button:
            if not url_input or not url_input.startswith(('http://', 'https://')):
                st.warning("Please enter a valid URL including http:// or https://")
            else:
                with st.spinner("Analyzing URL and making prediction..."):
                    try:
                        url_features = calculate_url_features(url_input)

                        input_data = {}
                        input_data.update(url_features)
                        input_data.update(manual_inputs)

                        input_df = pd.DataFrame([input_data])
                        for col in FINAL_MODEL_FEATURES:
                             if col not in input_df.columns:
                                  if col in ['HasFavicon', 'HasCopyrightInfo', 'HasSubmitButton', 'HasPasswordField']:
                                      input_df[col] = 0
                                  else:
                                      input_df[col] = 0
                                      st.warning(f"Warning: Feature '{col}' was unexpectedly missing and defaulted to 0.")
                        input_df = input_df[FINAL_MODEL_FEATURES] # Crucial: Ensure column order matches training

                        input_scaled = scaler.transform(input_df)

                        prediction = model.predict(input_scaled)
                        probability = model.predict_proba(input_scaled)

                        st.subheader("Prediction Result:")
                        is_phishing = prediction[0] == 1
                        if is_phishing:
                            st.error(f"**Prediction: Phishing** (Probability: {probability[0][1]:.2%})")
                        else:
                            st.success(f"**Prediction: Legitimate** (Probability: {probability[0][0]:.2%})")

                        with st.expander("Show Features Used for Prediction"):
                            st.dataframe(input_df)

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.error("Please check the input URL and feature values.")

st.sidebar.info(
    "Cybersecurity Analytics Dashboard\n\n"
    "Module: KH5004CMD Data Science\n\n"
    "Displays EDA and Phishing Prediction based on trained models."
)