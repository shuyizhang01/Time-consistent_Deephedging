import torch

def score(a1, a2, y, alpha, C=5):
    """
    Scoring function (4.6) for bounded costs |Y| < C.
    G1(x) = C, G2(x) = log(x + C)
    """
    I_tail = (y > a1).float()

    a2_plus_C = torch.clamp(a2 + C, min=1e-6)
    y_plus_C = torch.clamp(y + C, min=1e-6)

    term1 = torch.log(a2_plus_C / y_plus_C)
    term2 = -a2 / a2_plus_C
    term3 = (
        a1 * ((1.0 - alpha) - I_tail)
        + y * I_tail
    ) / (a2_plus_C * (1.0 - alpha))

    return term1 + term2 + term3

def score_power(a1, a2, y, alpha, C=5, p=2, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = -(x + C)^p
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps)
    y_C  = torch.clamp(y  + C, min=eps)

    term1 = a2_C**p - y_C**p
    adjustment = (a1 * (I_tail - (1 - alpha)) - y * I_tail) / (1 - alpha)
    prefactor  = -p * a2_C**(p - 1)

    return term1 + prefactor * a2 + prefactor * adjustment


def score_exp(a1, a2, y, alpha, C=5, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = exp(-(x+C)) - 1
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps)
    y_C  = torch.clamp(y  + C, min=eps)

    exp_a2 = torch.exp(-a2_C)
    exp_y  = torch.exp(-y_C)

    term1 = exp_y - exp_a2
    adjustment = (a1 * (I_tail - (1 - alpha)) - y * I_tail) / (1 - alpha)
    prefactor  = -exp_a2

    return term1 + prefactor * a2 + prefactor * adjustment


def score_rational(a1, a2, y, alpha, C=5, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = -(x+C)/(1+(x+C))
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps)
    y_C  = torch.clamp(y  + C, min=eps)

    denom_a2 = 1.0 + a2_C
    denom_y  = 1.0 + y_C

    term1 = a2_C / denom_a2 - y_C / denom_y
    adjustment = (a1 * (I_tail - (1.0 - alpha)) - y * I_tail) / (1.0 - alpha)
    prefactor  = -1.0 / (denom_a2 ** 2)

    return term1 + prefactor * a2 + prefactor * adjustment


def score_arctan(a1, a2, y, alpha, C=5, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = -arctan(x+C)
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps)
    y_C  = torch.clamp(y  + C, min=eps)

    term1 = torch.arctan(a2_C) - torch.arctan(y_C)
    adjustment = (a1 * (I_tail - (1.0 - alpha)) - y * I_tail) / (1.0 - alpha)
    prefactor  = -1.0 / (1.0 + a2_C ** 2)

    return term1 + prefactor * a2 + prefactor * adjustment


def score_arcsin(a1, a2, y, alpha, C=5, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = arcsin(1/(x+1+C))
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps) + 1.0
    y_C  = torch.clamp(y  + C, min=eps) + 1.0

    term1 = torch.arcsin(1.0 / y_C) - torch.arcsin(1.0 / a2_C)

    sqrt_term = torch.sqrt(torch.clamp(a2_C**2 - 1.0, min=eps))
    prefactor = -1.0 / (a2_C * sqrt_term)
    adjustment = (a1 * (I_tail - (1.0 - alpha)) - y * I_tail) / (1.0 - alpha)

    return term1 + prefactor * a2 + prefactor * adjustment


def score_arcsinh(a1, a2, y, alpha, C=5, eps=1e-8):
    """
    Scoring function (4.4) for G1(x) = C, G2(x) = -arcsinh(x+C)
    """
    I_tail = (y > a1).float()

    a2_C = torch.clamp(a2 + C, min=eps)
    y_C  = torch.clamp(y  + C, min=eps)

    term1 = torch.arcsinh(a2_C) - torch.arcsinh(y_C)
    prefactor = -1.0 / torch.sqrt(a2_C**2 + 1.0)
    adjustment = (a1 * (I_tail - (1.0 - alpha)) - y * I_tail) / (1.0 - alpha)

    return term1 + prefactor * a2 + prefactor * adjustment


# Registry for easy lookup in training
SCORE_FUNCTIONS = {
    'log':        score,
    'power03':    lambda a1, a2, y, alpha, C: score_power(a1, a2, y, alpha, C, p=0.3),
    'power05':    lambda a1, a2, y, alpha, C: score_power(a1, a2, y, alpha, C, p=0.5),
    'power07':    lambda a1, a2, y, alpha, C: score_power(a1, a2, y, alpha, C, p=0.7),
    'exponential': score_exp,
    'rational':   score_rational,
    'arctan':     score_arctan,
    'arcsin':     score_arcsin,
    'arcsinh':    score_arcsinh,
}


def critic_loss(a1_pred, a2_excess, costs, V_next, alpha, scoring_fn='log', C=300):
    """
    Critic loss: expected scoring rule over (VaR, ES) pair.

    Args:
        a1_pred:    predicted VaR,    shape [T, B]
        a2_excess:  predicted excess, shape [T, B]
        costs:      one-step costs,   shape [T, B]
        V_next:     next-state value, shape [T, B]
        alpha:      CVaR level (e.g. 0.95)
        scoring_fn: key into SCORE_FUNCTIONS
        C:          shift constant for boundedness
    """
    y = costs + V_next
    V_pred = a1_pred + a2_excess

    score_fn = SCORE_FUNCTIONS.get(scoring_fn)
    if score_fn is None:
        raise ValueError(f"Unknown scoring function '{scoring_fn}'. "
                         f"Available: {list(SCORE_FUNCTIONS.keys())}")

    scores = score_fn(a1_pred, V_pred, y, alpha, C)
    return scores.mean()


def actor_loss(log_probs, a1_precomputed, costs, V_next, alpha):
    """
    Actor loss: policy gradient weighted by tail excess.

    Args:
        log_probs:        log π(a|s),  shape [T, B]
        a1_precomputed:   VaR estimate (detached), shape [T, B]
        costs:            one-step costs,           shape [T, B]
        V_next:           next-state value,         shape [T, B]
        alpha:            CVaR level
    """
    target = costs + V_next
    excess = (target - a1_precomputed).detach()
    excess_pos = torch.clamp(excess, min=0.0)
    loss = (1.0 / (1.0 - alpha)) * (excess_pos * log_probs).mean()
    return loss
