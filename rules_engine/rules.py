
def check_prior_auth(claim, cpt_lookup):
    cpt = claim["cpt_code"]
    if cpt_lookup[cpt]["requires_auth"] and not claim["prior_authorization"]:
        return "Prior authorization required"
    return None


CPTS_REQUIRING_MODIFIER = {"93000", "70551"}
def check_modifier(claim):
    if claim["cpt_code"] in CPTS_REQUIRING_MODIFIER and claim["modifier"] is None:
        return "Missing modifier"
    return None



def check_charge_outlier(claim, cpt_lookup):
    avg = cpt_lookup[claim["cpt_code"]]["avg_charge"]
    if claim["charge_amount"] > avg * 1.6:
        return "Charge exceeds peer norm"
    return None


def check_provider_scope(claim, provider_map):
    specialty = claim["provider_specialty"]
    if specialty not in provider_map:
        return "Unknown provider specialty"

    if claim["cpt_code"] not in provider_map[specialty]:
        return "CPT outside provider scope"

    return None

