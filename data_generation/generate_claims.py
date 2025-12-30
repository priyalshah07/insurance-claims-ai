

import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta
import random

faker = Faker()
np.random.seed(42)

N_CLAIMS = 60000

cpt_df = pd.read_csv("/Users/priyalshah/Documents/insurance-claims-ai/data_generation/codebooks/cpt_codes.csv", dtype={"cpt_code": str})
cpt_df["cpt_code"] = cpt_df["cpt_code"].str.strip()

icd_df = pd.read_csv('/Users/priyalshah/Documents/insurance-claims-ai/data_generation/codebooks/icd_codes.csv')

providers = {                                    # key = provider specialty, value = CPT codes that the specialty can perform
    "Cardiology": ["93000", "99214"],
    "Primary Care": ["99213", "99214"],
    "Orthopedics": ["27447"],
    "Radiology": ["70551"],
    "Gastroenterology": ["45378"]
}

denial_reasons = [                               # list of common reasons claims get denied
    "Missing modifier",
    "Invalid CPT-ICD combination",
    "Prior authorization required",
    "Duplicate claim",
    "Upcoding suspected"
]

claims = []

for i in range(N_CLAIMS):
    specialty = random.choice(list(providers.keys()))
    cpt_code = random.choice(providers[specialty])   

    cpt_match = cpt_df[cpt_df["cpt_code"] == cpt_code]
    if cpt_match.empty:
        raise ValueError(f"CPT code {cpt_code} not found in codebook")

    cpt_row = cpt_match.iloc[0]                                  # finds the CPT row in the codebook to get details


    icd_code = icd_df.sample(1).iloc[0]["icd_code"]

    charge = np.random.normal(cpt_row["avg_charge"], cpt_row["avg_charge"] * 0.15) # base charge around average with some variance
    charge = max(50, round(charge, 2))                # ensure minimum charge of $50, rounded to 2 decimal places

    prior_auth = random.choice([True, False])
    modifier = random.choice([None, "26", "TC"])      

    deny = False                                   # Assume claim is approved unless a rule fails
    denial_reason = None

    # Denial logic (intentional)
    if cpt_row["requires_auth"] and not prior_auth:         # if prior auth is required but not provided
        deny = True                                         
        denial_reason = "Prior authorization required"
    elif modifier is None and random.random() < 0.15:     # if modifier is missing (15% of the time)
        deny = True
        denial_reason = "Missing modifier"
    elif random.random() < 0.1:                            # if there is a diagnosis mismatch (10% of the time)
        deny = True
        denial_reason = "Invalid CPT-ICD combination"
    elif charge > cpt_row["avg_charge"] * 1.6:         # If charge is unusually high compared to peers (average)
        deny = True
        denial_reason = "Upcoding suspected"

    claims.append({                                          # store the claim
        "claim_id": f"CLM-{100000+i}",
        "submission_date": faker.date_between(start_date="-180d", end_date="today"),
        "patient_age": random.randint(18, 85),
        "patient_gender": random.choice(["M", "F"]),
        "provider_id": f"PRV-{random.randint(1000, 2000)}",
        "provider_specialty": specialty,
        "cpt_code": cpt_code,
        "icd_code": icd_code,
        "modifier": modifier,
        "charge_amount": charge,
        "place_of_service": random.choice(["11", "21", "22"]),
        "prior_authorization": prior_auth,
        "claim_status": "Denied" if deny else "Approved",
        "denial_reason": denial_reason
    })

df = pd.DataFrame(claims)
df.to_csv("/Users/priyalshah/Documents/insurance-claims-ai/data/raw_claims.csv", index=False)

print("Claims generated and saved to raw_claims.csv: ", len(df))
print(df["claim_status"].value_counts(normalize=True))

print(df.head())
print("\nDenial reasons:\n", df["denial_reason"].value_counts())



