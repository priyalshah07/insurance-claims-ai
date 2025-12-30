import pandas as pd
from rules import (
    check_prior_auth,
    check_modifier,
    check_charge_outlier,
    check_provider_scope
)

claims = pd.read_csv("data/raw_claims.csv")
claims["cpt_code"] = claims["cpt_code"].astype(str).str.strip()

# Load CPT reference data (CPT lookup)
cpt_df = pd.read_csv(
    "data_generation/codebooks/cpt_codes.csv",
    dtype={"cpt_code": str}
)
cpt_df["cpt_code"] = cpt_df["cpt_code"].str.strip()

cpt_lookup = {
    row["cpt_code"]: {
        "avg_charge": row["avg_charge"],
        "requires_auth": row["requires_auth"]
    }
    for _, row in cpt_df.iterrows()
}

# Provider scope map
provider_map = {
    "Cardiology": ["93000", "99214"],
    "Primary Care": ["99213", "99214"],
    "Orthopedics": ["27447"],
    "Radiology": ["70551"],
    "Gastroenterology": ["45378"]
}

rule_results = []

# Apply rules claim-by-claim
for _, claim in claims.iterrows():
    failures = []

    reason = check_prior_auth(claim, cpt_lookup)
    if reason:
        failures.append(reason)

    reason = check_modifier(claim)
    if reason:
        failures.append(reason)

    reason = check_charge_outlier(claim, cpt_lookup)
    if reason:
        failures.append(reason)

    reason = check_provider_scope(claim, provider_map)
    if reason:
        failures.append(reason)

    rule_results.append({
        "rule_failed": len(failures) > 0,
        "rule_failure_reasons": failures
    })

# Combine results
rules_df = pd.DataFrame(rule_results)
final_df = pd.concat([claims.reset_index(drop=True), rules_df], axis=1)

# Save output
final_df.to_csv("data/validated_claims.csv", index=False)

# Diagnostics
print("Validation complete")
print(final_df["rule_failed"].value_counts(normalize=True))
print(
    final_df["rule_failure_reasons"]
    .explode()
    .value_counts()
)
