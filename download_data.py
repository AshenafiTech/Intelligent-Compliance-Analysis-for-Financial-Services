#!/usr/bin/env python3
"""Download real CFPB complaint data"""
import pandas as pd
import os

def download_cfpb_data():
    """Download and filter CFPB complaint data"""
    print("Downloading CFPB complaint data...")
    
    # CFPB API endpoint for complaints
    url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
    
    try:
        # Read directly from URL (first 10000 rows for demo)
        df = pd.read_csv(url, nrows=10000)
        print(f"Downloaded {len(df)} complaints")
        
        # Filter for relevant products
        products = [
            'Credit card or prepaid card',
            'Debt collection', 
            'Mortgage',
            'Student loan',
            'Bank account or service',
            'Money transfer, virtual currency, or money service'
        ]
        
        df_filtered = df[df['Product'].isin(products)].copy()
        df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
        
        # Clean and prepare data
        df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].str.lower()
        df_filtered['cleaned_narrative'] = df_filtered['cleaned_narrative'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df_filtered.to_csv('data/processed/filtered_complaints.csv', index=False)
        
        print(f"Saved {len(df_filtered)} filtered complaints")
        print("Product distribution:")
        print(df_filtered['Product'].value_counts())
        
        return df_filtered
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Creating sample data instead...")
        
        # Fallback sample data
        sample_data = {
            'Consumer complaint narrative': [
                "I applied for a credit card online but was denied without any clear explanation. The process was confusing and when I called customer service, they were unhelpful and couldn't provide specific reasons for the denial.",
                "My Buy Now Pay Later payment was processed twice in the same day, causing overdraft fees in my bank account. I contacted customer support multiple times but received no resolution or refund for the duplicate charge.",
                "The personal loan interest rate was not clearly disclosed during the application process. Hidden fees appeared after approval that were never mentioned upfront, making the loan much more expensive than advertised.",
                "My money transfer to my family overseas failed but the funds were still deducted from my account. It took over three weeks to get a refund and customer service was unresponsive during this time.",
                "My savings account was suddenly closed without any prior notice or explanation. I lost access to my funds for several days and had to visit the branch multiple times to resolve the issue.",
                "The BNPL service charged me unexpected late fees even though I made my payment on the due date. Their payment system seems to have processing delays that result in false late charges.",
                "Credit card customer service is absolutely terrible. It takes forever to reach a representative and when you do, they can't resolve basic issues or provide clear information about account problems."
            ],
            'Product': [
                'Credit card or prepaid card',
                'Credit card or prepaid card', 
                'Student loan',
                'Money transfer, virtual currency, or money service',
                'Bank account or service',
                'Credit card or prepaid card',
                'Credit card or prepaid card'
            ],
            'Issue': [
                'Application processing delay',
                'Billing disputes',
                'Getting a loan',
                'Money was not available when promised',
                'Account opening, closing, or management',
                'Billing disputes',
                'Customer service / Customer relations'
            ],
            'cleaned_narrative': [
                "i applied for a credit card online but was denied without any clear explanation the process was confusing and when i called customer service they were unhelpful and couldnt provide specific reasons for the denial",
                "my buy now pay later payment was processed twice in the same day causing overdraft fees in my bank account i contacted customer support multiple times but received no resolution or refund for the duplicate charge",
                "the personal loan interest rate was not clearly disclosed during the application process hidden fees appeared after approval that were never mentioned upfront making the loan much more expensive than advertised",
                "my money transfer to my family overseas failed but the funds were still deducted from my account it took over three weeks to get a refund and customer service was unresponsive during this time",
                "my savings account was suddenly closed without any prior notice or explanation i lost access to my funds for several days and had to visit the branch multiple times to resolve the issue",
                "the bnpl service charged me unexpected late fees even though i made my payment on the due date their payment system seems to have processing delays that result in false late charges",
                "credit card customer service is absolutely terrible it takes forever to reach a representative and when you do they cant resolve basic issues or provide clear information about account problems"
            ]
        }
        
        df_sample = pd.DataFrame(sample_data)
        os.makedirs('data/processed', exist_ok=True)
        df_sample.to_csv('data/processed/filtered_complaints.csv', index=False)
        
        print(f"Created sample dataset with {len(df_sample)} complaints")
        return df_sample

if __name__ == "__main__":
    download_cfpb_data()