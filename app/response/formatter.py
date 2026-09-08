"""
Response Layer
---------------
Takes the raw output of the model inference layer and turns it into the
final API response: a phishing SCORE, a risk LEVEL, and an OPTIONAL
human-readable EXPLANATION built from the model's feature importances.

Keeping this as its own layer means the risk-level thresholds, the wording
of explanations, and the response schema (i.e. the API "contract" for
client apps) can all evolve independently of the model itself.
"""

def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _label(score: float) -> str:
    return "phishing" if score >= 0.5 else "legitimate"


_EXPLANATIONS = {
    "has_https": "the connection does not use HTTPS",
    "tld_suspicious": "the domain uses a top-level domain often abused for phishing",
    "digit_ratio": "the URL contains an unusually high proportion of digits",
    "num_suspicious_words": "the URL contains words commonly used in phishing (e.g. verify, login, secure)",
    "has_suspicious_words": "the URL contains words commonly used in phishing (e.g. verify, login, secure)",
    "num_hyphens": "the hostname contains an unusually high number of hyphens",
    "hostname_length": "the hostname is unusually long",
    "num_digits": "the URL contains an unusually high number of digits",
    "has_ip_address": "the URL uses a raw IP address instead of a domain name",
    "num_at_symbols": "the URL contains an '@' symbol, often used to disguise the real destination",
    "is_shortened": "the URL was created with a link-shortening service",
    "brand_name_in_subdomain": "a well-known brand name appears in the subdomain rather than the real domain",
    "num_subdomains": "the URL has an unusually high number of subdomains",
    "shannon_entropy": "the URL text has unusually high randomness (entropy), often seen in auto-generated phishing links",
    "url_length": "the overall URL is unusually long",
    "longest_word_length": "the URL contains an unusually long unbroken token",
    "num_percent": "the URL contains encoded ('%') characters",
    "has_port": "the URL specifies a non-standard port",
}


def _explain(features: dict, top_contributing_features: list, predicted_label: int) -> list:
    explanation = []
    for item in top_contributing_features:
        name = item["feature"]
        value = item["value"]
        text = _EXPLANATIONS.get(name)
        if not text:
            continue
        # Only surface an explanation line if the feature's value actually
        # looks "activated" / suspicious for boolean-ish or count features.
        if name in ("has_https",):
            if predicted_label == 1 and value == 0:
                explanation.append(text)
            continue
        if isinstance(value, (int, float)) and value == 0:
            continue
        if text not in explanation:
            explanation.append(text)
    return explanation


def build_response(url: str, inference_result: dict, include_explanation: bool = True) -> dict:
    score = inference_result["phishing_probability"]
    predicted_label = inference_result["predicted_label"]

    response = {
        "url": url,
        "score": score,
        "label": _label(score),
        "risk_level": _risk_level(score),
    }

    if include_explanation:
        response["explanation"] = _explain(
            inference_result["features"],
            inference_result["top_contributing_features"],
            predicted_label,
        )
        response["top_contributing_features"] = inference_result["top_contributing_features"]

    return response
