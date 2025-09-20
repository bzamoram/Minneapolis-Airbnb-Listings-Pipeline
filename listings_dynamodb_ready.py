import pandas as pd
import json
import ast
import re
import html
import math
from typing import Any, List, Dict

# --- Utility functions ---

def is_missing(val: Any) -> bool:
#Returns True if val is None, empty string, 'nan', or NaN.
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        return s == "" or s == "nan" or s == "none"
    return False

def try_cast_number(val: Any):
#If val looks like a number string or float, convert to int or float, else return cleaned string.
    if is_missing(val):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return val
    s = str(val).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    return s

def strip_html(text: Any) -> Any:
#Remove HTML tags and unescape HTML entities for clean text.
    if is_missing(text):
        return None
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()

def parse_price(value: Any):
#Clean price string, remove currency symbols, return int or float or None.
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return try_cast_number(value)
    s = str(value)
    s_clean = re.sub(r"[^\d\.-]", "", s)  # remove non-numeric chars except dot/dash
    if s_clean in ("", "-", "."):
        return None
    try:
        f = float(s_clean)
    except ValueError:
        return None
    if f.is_integer():
        return int(f)
    return f

def parse_list_field(raw: Any) -> List[str]:
#Parse a string that should represent a list into a Python list of strings.
    if is_missing(raw):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if not is_missing(x)]

    s = str(raw).strip()

#Try JSON parse if bracketed list string
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [strip_html(x) for x in parsed if not is_missing(x)]
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [strip_html(x) for x in parsed if not is_missing(x)]
            except Exception:
                pass

#Try splitting by '", "' pattern commonly used
    if '", "' in s or '",\n"' in s or '",\r\n"' in s:
        content = s.strip("[]")
        parts = re.split(r'"\s*,\s*"', content)
        parts = [p.strip(' "[]') for p in parts if p.strip(' "[]')]
        return [strip_html(p) for p in parts]

# Fallback: split on commas, strip quotes and spaces
    parts = [p.strip(' "\'') for p in s.strip("[]").split(",") if p.strip(' "\'')]
    return [strip_html(p) for p in parts]

def clean_review_dict(raw_review: Dict[str, Any]) -> Dict[str, Any]:
# Clean a review dict by stripping html, casting numbers, removing listing_id.
    cleaned = {}
    for k, v in raw_review.items():
        if str(k).lower() == "listing_id":
            continue  # Avoid repeating foreign key in review
        key = k
        if key in ("id", "reviewer_id"):
            if is_missing(v):
                cleaned[key] = None
            else:
                casted = try_cast_number(v)
                cleaned[key] = str(casted) if isinstance(casted, str) and casted.isdigit() else casted
        elif key == "date":
            cleaned[key] = strip_html(v)
        elif key == "comments":
            cleaned[key] = strip_html(v)
        else:
            cleaned[key] = try_cast_number(v) if isinstance(v, (int, float, str)) else v
    return cleaned

def parse_amenities(raw):
# Convert amenities from string to list of strings.
    if is_missing(raw):
        return []
    try:
        # Try to parse as JSON list
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Strip braces/quotes if needed, split on commas
    raw = raw.replace("{", "").replace("}", "")
    return [x.strip(' "').strip() for x in raw.split(",") if x.strip(' "').strip()]

def prune_none(d: Any) -> Any:
#Remove keys with None values at top level and nested dicts.
    if isinstance(d, dict):
        return {k: prune_none(v) for k, v in d.items() if v is not None and (not isinstance(v, dict) or any(x is not None for x in v.values()))}
    return d

# --- Main function to build listing object ---

def build_listing_object(row: pd.Series) -> Dict[str, Any]:
# Transform one listing row + its reviews into a clean JSON object for DynamoDB.
    r = row.to_dict()
    listing_id_raw = r.get("id") or r.get("listing_id") or r.get("listingid")
    listing_id = None if is_missing(listing_id_raw) else str(listing_id_raw).strip()

    listing_obj = {
        "PK": listing_id,
        "SK": "LISTING",
        "type": "listing",
        "id": listing_id,
        "url": strip_html(r.get("listing_url")),
        "name": strip_html(r.get("name")),
        "description": strip_html(r.get("description")),
        "neighborhood": {
            "overview": strip_html(r.get("neighborhood_overview")),
            "cleansed": strip_html(r.get("neighbourhood_cleansed") or r.get("neighbourhood_cleansed")),
            "group_cleansed": strip_html(r.get("neighbourhood_group_cleansed")),
            "neighbourhood": strip_html(r.get("neighbourhood"))
        },
        "host": {
            "id": str(r.get("host_id")) if not is_missing(r.get("host_id")) else None,
            "url": strip_html(r.get("host_url")),
            "name": strip_html(r.get("host_name")),
            "since": strip_html(r.get("host_since")),
            "location": strip_html(r.get("host_location")),
            "about": strip_html(r.get("host_about")),
            "is_superhost": str(r.get("host_is_superhost")).strip().lower() in ("t", "true", "yes", "1"),
            "listings_count": try_cast_number(r.get("host_listings_count")),
            "verifications": parse_list_field(r.get("host_verifications")),
            "has_profile_pic": str(r.get("host_has_profile_pic")).strip().lower() in ("t", "true", "yes", "1"),
            "identity_verified": str(r.get("host_identity_verified")).strip().lower() in ("t", "true", "yes", "1"),
            "thumbnail_url": strip_html(r.get("host_thumbnail_url")),
            "picture_url": strip_html(r.get("host_picture_url")),
            "response_time": strip_html(r.get("host_response_time")),
            "response_rate": try_cast_number(r.get("host_response_rate")),
            "acceptance_rate": try_cast_number(r.get("host_acceptance_rate"))
        },
        "property": {
            "type": strip_html(r.get("property_type") or r.get("room_type")),
            "room_type": strip_html(r.get("room_type")),
            "accommodates": try_cast_number(r.get("accommodates")),
            "bathrooms": try_cast_number(r.get("bathrooms")),
            "bedrooms": try_cast_number(r.get("bedrooms")),
            "beds": try_cast_number(r.get("beds")),
            "amenities": parse_amenities(r.get("amenities"))
        },
        "pricing": {
            "price": parse_price(r.get("price")),
            "estimated_revenue_l365d": try_cast_number(r.get("estimated_revenue_l365d") or r.get("calculated_host_listings_count"))
        },
        "reviews": {
            "count": try_cast_number(r.get("number_of_reviews")),
            "review_scores": {
                "rating": try_cast_number(r.get("review_scores_rating")),
                "accuracy": try_cast_number(r.get("review_scores_accuracy")),
                "cleanliness": try_cast_number(r.get("review_scores_cleanliness")),
                "checkin": try_cast_number(r.get("review_scores_checkin")),
                "communication": try_cast_number(r.get("review_scores_communication")),
                "location": try_cast_number(r.get("review_scores_location")),
                "value": try_cast_number(r.get("review_scores_value"))
            },
            "first_review": strip_html(r.get("first_review")),
            "last_review": strip_html(r.get("last_review")),
            "reviews_per_month": try_cast_number(r.get("reviews_per_month")),
        },
        "coordinates": {
            "latitude": try_cast_number(r.get("latitude")),
            "longitude": try_cast_number(r.get("longitude"))
        },
        "license": strip_html(r.get("license")),
        "instant_bookable": str(r.get("instant_bookable")).strip().lower() in ("t", "true", "yes", "1"),
        "minimum_nights": try_cast_number(r.get("minimum_nights")),
        "maximum_nights": try_cast_number(r.get("maximum_nights"))
    }
    # Remove keys with None values for clean JSON
    return prune_none(listing_obj)

def build_review_object(rev: Dict[str, Any]) -> Dict[str, Any]:
    listing_id = rev.get("listing_id") 
    if is_missing(listing_id): 
        return None
    listing_id = str(listing_id).strip()

    review_id = str(rev.get("id") or rev.get("review_id") or "")
    review_id = review_id.strip()
    if not review_id:
        return None
    
    raw_reviewerid = rev.get("reviewerid")
    reviewerid = str(raw_reviewerid).strip() if raw_reviewerid is not None else None
    if reviewerid == "":
        reviewerid = None

    review_obj = {
        "PK": listing_id,
        "SK": f"REVIEW#{review_id}",
        "type": "review",
        "id": review_id,
        "date": strip_html(rev.get("date", "")),
        "reviewerid": str(rev.get("reviewerid") or ""),
        "reviewername": strip_html(rev.get("reviewername") or ""),
        "comments": strip_html(rev.get("comments") or "")
    }
    review_obj.update(clean_review_dict(rev))
    return prune_none(review_obj)

# --- Main ETL pipeline script ---

def main(listings_path="listings.csv", 
         reviews_path="reviews.csv", 
         output_path="airbnb_listings.json", 
         sample_n=50, 
         random_sample=True
):
    """
    Reads listings and reviews CSVs, cleans and transforms data,
    builds DynamoDB-ready JSON with PK/SK for listings and reviews,
    and writes the output JSON file.
    """

    # 1. Read CSV files into pandas DataFrames (all columns as strings to avoid type issues)
    listings = pd.read_csv(listings_path, dtype=str, low_memory=False)
    reviews = pd.read_csv(reviews_path, dtype=str, low_memory=False)

    # 2. Pre-process all reviews into a dictionary keyed by listing_id
    reviews_map = {}
    for _, rev_row in reviews.iterrows():
        rev = rev_row.to_dict()
        listing_id = rev.get("listing_id") 
        if is_missing(listing_id):
            continue
        listing_id = str(listing_id).strip()
        reviews_map.setdefault(listing_id, []).append(rev)

    # 3. Select sample of listings (random or first n)
    if random_sample:
        if len(listings) <= sample_n:
            sample_df = listings.copy()
        else:
            sample_df = listings.sample(n=sample_n, random_state=42)
    else:
        sample_df = listings.head(sample_n)

    # 4. Build JSON documents per listing with embedded reviews
    json_docs = []
    for _, row in sample_df.iterrows():
        # Build and append listing object
        listing_obj = build_listing_object(row)
        json_docs.append(listing_obj)

        listing_id = str(row.get("id")).strip()

        for review in reviews_map.get(listing_id, []):
            review_obj = build_review_object(review)
            if review_obj is not None:
                print(review_obj)
                json_docs.append(review_obj)
    
    # 5. Write output JSON with indentation, UTF-8 for readability and character support
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(json_docs)} listings with reviews to {output_path}")


if __name__ == "__main__":
    main()
