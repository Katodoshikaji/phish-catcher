"""
Synthetic labeled dataset generator.

Why synthetic instead of downloading a public dataset?
This project environment has no reliable outbound access to dataset
mirrors (UCI / PhishTank / Kaggle), so we generate a labeled dataset
programmatically using the same lexical patterns that real-world phishing
and legitimate URLs are known to exhibit (see project report for the
justification and references to the UCI "Phishing Websites" feature set).

The generator produces:
  - LEGITIMATE examples: URLs built from a list of well-known, real
    registrable domains, with normal paths/query strings, https, no
    suspicious tricks.
  - PHISHING examples: URLs built using well-documented phishing
    obfuscation techniques: raw IP hosts, "@" redirection tricks, brand
    names stuffed into subdomains/hyphens, suspicious keywords
    (login/verify/secure/update...), long random subdomains, suspicious
    TLDs, URL shorteners, excessive hyphens/digits, http instead of https.

Output: app/data/dataset.csv with columns = FEATURE_NAMES + ["label"]
        (label: 1 = phishing, 0 = legitimate)
"""

import csv
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.feature_extraction.extractor import extract_features, FEATURE_NAMES  # noqa: E402

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "twitter.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "netflix.com", "yahoo.com", "reddit.com", "ebay.com", "bing.com",
    "office.com", "adobe.com", "github.com", "stackoverflow.com", "dropbox.com",
    "paypal.com", "spotify.com", "zoom.us", "salesforce.com", "wordpress.com",
    "cnn.com", "bbc.com", "nytimes.com", "forbes.com", "medium.com",
    "quora.com", "pinterest.com", "whatsapp.com", "telegram.org", "slack.com",
    "shopify.com", "airbnb.com", "booking.com", "expedia.com", "walmart.com",
    "target.com", "bestbuy.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "citibank.com", "hsbc.com", "irs.gov", "usa.gov", "nasa.gov",
    "harvard.edu", "mit.edu", "stanford.edu", "coursera.org", "udemy.com",
    "khanacademy.org", "nature.com", "sciencedirect.com", "ieee.org", "who.int",
    "un.org", "python.org", "npmjs.com", "docker.com", "kubernetes.io",
    "mozilla.org", "w3.org", "cloudflare.com", "digitalocean.com", "heroku.com",
    "atlassian.com", "trello.com", "notion.so", "figma.com", "canva.com",
    "twitch.tv", "discord.com", "steam.com", "epicgames.com", "ea.com",
    "samsung.com", "sony.com", "intel.com", "nvidia.com", "amd.com",
    "ibm.com", "oracle.com", "sap.com", "cisco.com", "vmware.com",
    "hp.com", "dell.com", "lenovo.com", "asus.com", "logitech.com",
    "uber.com", "lyft.com", "doordash.com", "grubhub.com", "instacart.com",
]

PATH_WORDS = [
    "home", "about", "products", "services", "contact", "blog", "news",
    "help", "support", "account", "profile", "settings", "search", "cart",
    "checkout", "orders", "docs", "api", "login", "signup", "dashboard",
]

QUERY_KEYS = ["id", "ref", "page", "q", "utm_source", "lang", "sort", "category"]

BRAND_TARGETS = [
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "chase", "instagram",
    "whatsapp", "outlook", "office365", "icloud", "linkedin",
]

SUSPICIOUS_KEYWORDS = [
    "secure", "account", "update", "verify", "login", "signin", "confirm",
    "password", "suspend", "billing", "urgent", "alert", "recover",
    "unlock", "wallet", "security", "authenticate", "validate",
]

SUSPICIOUS_TLDS = ["xyz", "top", "gq", "ml", "cf", "tk", "work", "click",
                    "link", "loan", "review", "kim", "cricket", "party", "zip"]

SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly"]


def _rand_alnum(n):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_legitimate_url():
    domain = random.choice(LEGIT_DOMAINS)
    scheme = "https"
    use_www = random.random() < 0.5
    host = ("www." if use_www else "") + domain
    depth = random.randint(0, 3)
    path = "/".join(random.choices(PATH_WORDS, k=depth))
    query = ""
    if random.random() < 0.4:
        k = random.choice(QUERY_KEYS)
        v = _rand_alnum(random.randint(2, 8))
        query = f"?{k}={v}"
        if random.random() < 0.3:
            k2 = random.choice(QUERY_KEYS)
            v2 = _rand_alnum(random.randint(2, 6))
            query += f"&{k2}={v2}"
    url = f"{scheme}://{host}/{path}{query}"
    return url.rstrip("/")


def make_phishing_url():
    technique = random.choice([
        "ip_host", "at_symbol", "brand_subdomain", "hyphen_brand",
        "suspicious_tld", "shortener", "long_random_subdomain", "keyword_stuffing",
    ])

    if technique == "ip_host":
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        path = "/" + random.choice(SUSPICIOUS_KEYWORDS) + "/" + _rand_alnum(6)
        scheme = random.choice(["http", "https"])
        return f"{scheme}://{ip}{path}"

    if technique == "at_symbol":
        brand = random.choice(BRAND_TARGETS)
        fake_host = f"{brand}.com-{_rand_alnum(5)}.{random.choice(SUSPICIOUS_TLDS)}"
        return f"http://{brand}.com@{fake_host}/{random.choice(SUSPICIOUS_KEYWORDS)}"

    if technique == "brand_subdomain":
        brand = random.choice(BRAND_TARGETS)
        sub = f"{brand}-{random.choice(SUSPICIOUS_KEYWORDS)}"
        tld = random.choice(SUSPICIOUS_TLDS)
        host = f"{sub}.{_rand_alnum(6)}.{tld}"
        path = "/" + random.choice(SUSPICIOUS_KEYWORDS)
        return f"http://{host}{path}"

    if technique == "hyphen_brand":
        brand = random.choice(BRAND_TARGETS)
        word = random.choice(SUSPICIOUS_KEYWORDS)
        host = f"{brand}-{word}-{_rand_alnum(4)}.{random.choice(SUSPICIOUS_TLDS)}"
        return f"http://{host}/{random.choice(PATH_WORDS)}"

    if technique == "suspicious_tld":
        word = random.choice(SUSPICIOUS_KEYWORDS)
        host = f"{word}-{_rand_alnum(5)}.{random.choice(SUSPICIOUS_TLDS)}"
        return f"http://{host}/{_rand_alnum(8)}"

    if technique == "shortener":
        host = random.choice(SHORTENERS)
        return f"http://{host}/{_rand_alnum(7)}"

    if technique == "long_random_subdomain":
        brand = random.choice(BRAND_TARGETS)
        sub = _rand_alnum(random.randint(15, 25))
        host = f"{sub}.{brand}.{_rand_alnum(6)}.{random.choice(SUSPICIOUS_TLDS)}"
        return f"http://{host}/{random.choice(SUSPICIOUS_KEYWORDS)}"

    # keyword_stuffing
    words = random.sample(SUSPICIOUS_KEYWORDS, k=min(3, len(SUSPICIOUS_KEYWORDS)))
    host = "-".join(words) + f"-{_rand_alnum(4)}.{random.choice(SUSPICIOUS_TLDS)}"
    query = f"?redirect={_rand_alnum(10)}&token={_rand_alnum(12)}"
    return f"http://{host}/{random.choice(PATH_WORDS)}{query}"


def generate(n_per_class=1500):
    rows = []
    urls_seen = set()

    legit_count = 0
    while legit_count < n_per_class:
        u = make_legitimate_url()
        if u in urls_seen:
            continue
        urls_seen.add(u)
        feats = extract_features(u)
        row = [feats[name] for name in FEATURE_NAMES]
        row.append(0)
        rows.append((u, row))
        legit_count += 1

    phish_count = 0
    while phish_count < n_per_class:
        u = make_phishing_url()
        if u in urls_seen:
            continue
        urls_seen.add(u)
        feats = extract_features(u)
        row = [feats[name] for name in FEATURE_NAMES]
        row.append(1)
        rows.append((u, row))
        phish_count += 1

    random.shuffle(rows)
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    rows = generate(n_per_class=1500)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"] + FEATURE_NAMES + ["label"])
        for url, row in rows:
            writer.writerow([url] + row)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
