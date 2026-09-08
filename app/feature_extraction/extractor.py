"""
Feature Extraction Layer
-------------------------
Converts a raw URL string into a fixed-length numeric feature vector.

Design notes for the project report:
- All features are LEXICAL / STRUCTURAL, derived purely from the URL string
  itself (length, characters, tokens, domain structure). No live network
  calls (DNS / WHOIS / page fetch) are made here.
- This keeps the layer fast, stateless, and cheap to run on every request,
  which matters because it sits in the hot path of a cloud API and may be
  invoked by browsers, mobile apps and other services at high volume.
- The feature set is inspired by the well-known UCI "Phishing Websites"
  feature families (URL-based features), restricted to the subset that can
  be computed instantly and deterministically from the URL text alone.
"""

import math
import re
from urllib.parse import urlparse

# Ordered list of feature names -> this order MUST match the order produced
# by extract_features(). Both training and inference import this constant so
# the two layers can never drift out of sync.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_ampersands",
    "num_digits",
    "num_percent",
    "digit_ratio",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_port",
    "is_shortened",
    "has_suspicious_words",
    "num_suspicious_words",
    "brand_name_in_subdomain",
    "shannon_entropy",
    "longest_word_length",
    "tld_suspicious",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rb.gy", "tiny.cc", "rebrand.ly",
}

SUSPICIOUS_WORDS = [
    "secure", "account", "update", "verify", "login", "signin", "bank",
    "confirm", "password", "webscr", "ebayisapi", "suspend", "billing",
    "urgent", "alert", "limited", "click", "recover", "unlock", "wallet",
    "security", "authenticate", "validate",
]

# A small illustrative set of well-known brand names commonly impersonated
# in phishing subdomains, e.g. "paypal.com.verify-login.xyz"
KNOWN_BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "chase", "instagram",
    "whatsapp", "outlook", "office365", "icloud", "linkedin",
]

SUSPICIOUS_TLDS = {
    "zip", "xyz", "top", "gq", "ml", "cf", "tk", "work", "click", "link",
    "loan", "review", "country", "kim", "cricket", "science", "party",
}

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _longest_word_length(url: str) -> int:
    tokens = re.split(r"[/\-_.?=&%:]+", url)
    tokens = [t for t in tokens if t]
    return max((len(t) for t in tokens), default=0)


def extract_features(url: str) -> dict:
    """Convert a single URL string into a dict of numeric features.

    Returns a dict keyed by FEATURE_NAMES so it can be fed directly into a
    pandas DataFrame column-aligned with the trained model.
    """
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        # No scheme provided -> assume http(s) so urlparse behaves correctly
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    lower_url = url.lower()

    num_dots = hostname.count(".")
    labels = [l for l in hostname.split(".") if l]
    # subdomain count = number of labels beyond registrable domain+tld (~2 labels)
    num_subdomains = max(len(labels) - 2, 0)

    has_ip = bool(IP_PATTERN.match(hostname))

    has_https = 1 if parsed.scheme == "https" else 0

    has_port = 1 if (":" in (parsed.netloc or "") and not has_ip and parsed.port) else (
        1 if parsed.port else 0
    )

    registrable_guess = ".".join(labels[-2:]) if len(labels) >= 2 else hostname
    is_shortened = 1 if registrable_guess in SHORTENER_DOMAINS else 0

    found_words = [w for w in SUSPICIOUS_WORDS if w in lower_url]
    has_susp_words = 1 if found_words else 0

    # Brand impersonation heuristic: a known brand name appears somewhere in
    # the hostname, but NOT as the actual registrable domain (i.e. it's
    # stuffed into a subdomain or hyphenated to trick the eye).
    brand_in_sub = 0
    for brand in KNOWN_BRANDS:
        if brand in hostname and brand != registrable_guess.split(".")[0]:
            brand_in_sub = 1
            break

    tld = labels[-1] if labels else ""
    tld_suspicious = 1 if tld in SUSPICIOUS_TLDS else 0

    digits = sum(c.isdigit() for c in url)

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": num_dots,
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_digits": digits,
        "num_percent": url.count("%"),
        "digit_ratio": digits / len(url) if url else 0.0,
        "num_subdomains": num_subdomains,
        "has_ip_address": int(has_ip),
        "has_https": has_https,
        "has_port": has_port,
        "is_shortened": is_shortened,
        "has_suspicious_words": has_susp_words,
        "num_suspicious_words": len(found_words),
        "brand_name_in_subdomain": brand_in_sub,
        "shannon_entropy": round(_shannon_entropy(url), 4),
        "longest_word_length": _longest_word_length(url),
        "tld_suspicious": tld_suspicious,
    }
    return features


def extract_feature_vector(url: str):
    """Return features as an ordered list matching FEATURE_NAMES."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]
