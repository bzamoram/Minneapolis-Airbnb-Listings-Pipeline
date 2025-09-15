#!/usr/bin/env python3 
# First line makes that when running the entire script using ./generate_json_clean.py, 
# then it the back python3 generate_json_clean.py runs

import pandas as pd #library for working with tabular data
import json # translator between python objects and JSON text
import ast # safely evaluates strings that look like python literals
import re # super-powered "find & replace" (strips $ signs and clean up HTML tags)
import html # built-in module for working with HTML entities (&amp -> &, &quot -> ")
import math # basic calculator and numeric tools (detects empty cells in csvs: NaN (not-a-number))
import random # python's random number library
from typing import Any, List, Dict # helps describing data shapes and readability 

# ---------------------------------------------------------------------
# Data Cleaning for creating denormalized nosql data in JSON format
# ---------------------------------------------------------------------

def is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        return s == "" or s == "nan" or s == "none"
    return False
# Returns True if the value is missing / NaN / None / empty-string / 'nan' string.
# We check multiple cases because pandas may give np.nan or the literal 'nan' string.
# pandas NaN is float('nan'), so it returns True but if the values is not empty or does 
# not have nan or none, then returns False

def try_cast_number(val: Any):
    if is_missing(val):
        return None
    # if already a number (int/float), normalize ints
    if isinstance(val, (int,)):
        return val
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return val
# Converts a string numeric to int/float. If it's already numeric, returns it.
# If it's not numeric, return the original input (cleaned string).


    s = str(val).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    return s
# If a value looks like an integer (e.g. "42") or a float (e.g. "3.14"),
# convert it to the proper number type; otherwise return it as a string.

def strip_html(text: Any) -> Any:
    if is_missing(text):
        return None
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()
# Remove common HTML tags like <br>, <p>, etc., and unescape HTML entities.
# If input is missing, return None.

def parse_price(value: Any):
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return try_cast_number(value)
    s = str(value)
    s_clean = re.sub(r"[^\d\.-]", "", s)
    if s_clean == "" or s_clean == "-" or s_clean == ".":
        return None
    try:
        f = float(s_clean)
    except ValueError:
        return None
    if f.is_integer():
        return int(f)
    return f
# remove currency symbols and commas
# Parse price strings like "$63.00" -> 63.0 (or 63 if integer)
# Return None if missing.

def parse_list_field(raw: Any) -> List[str]:
    if is_missing(raw):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if not is_missing(x)]

    s = str(raw).strip()
# Parse a field that *should* be a list (amenities, verifications, etc.)
# Handles these cases:
# - actual Python-style list string: "['a', 'b']"
# - JSON list string: '["a", "b"]'
# - comma-separated string: "a, b, c"
# Returns a cleaned list of strings (empty list on missing).

# valid JSON list? if goes as []
    if s.startswith("[") and s.endswith("]"):
# try json.loads first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [strip_html(x) if not is_missing(x) else None for x in parsed]
# if JSON parsing fails, and it is stored as python literal string, then we try ast.literal_eval as fallback
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [strip_html(x) if not is_missing(x) else None for x in parsed]
# if both json.loads and ast.literal_eval fail, then we fall through to heuristic splitting
            except Exception:
                pass

# Heuristic splitting:
# If the string contains patterns like '", "' we try to split there first
    if '", "' in s or '",\n"' in s or '",\r\n"' in s:
# we remove starting/ending brackets/quotes if present, then split on '", "'
        content = s.strip("[]")
        parts = re.split(r'"\s*,\s*"', content)
        parts = [p.strip(' "[]') for p in parts if p.strip(' "[]')]
        return [strip_html(p) for p in parts]
# Fallback: splitting on comma may split items that themselves contain comma
    parts = [p.strip(' "\'') for p in s.strip("[]").split(",") if p.strip(' "\'')]
    return [strip_html(p) for p in parts]

def clean_review_dict(raw_review: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in raw_review.items():
# Skips listing_id (it's redundant because each review is already nested under its 
# parent listing, so no need to repeat the foreign key)
        if str(k).lower() in ("listing_id",):
            continue
        key = k
        if key in ("id", "reviewer_id"):
# For review IDs and reviewer IDs:
# - we keep them safe as strings if needed (to avoid losing precision on very large numbers when converting to int)
# - otherwise, we cast to int if it's clearly numeric
            if is_missing(v):
                cleaned[key] = None
            else:
                casted = try_cast_number(v)
                cleaned[key] = str(casted) if isinstance(casted, str) and casted.isdigit() else casted
        elif key == "date":
            cleaned[key] = strip_html(v)
# Reviews date: strip out any HTML tags/entities and keep it as a clean string            
        elif key == "comments":
            cleaned[key] = strip_html(v)
# Reviews comments: also sanitize HTML (remove <br>, escape codes, etc.)            
        else:
# All other fields (reviewer_name, scores, etc.):
# - If numeric-looking, cast to int/float
# - Otherwise, just keep as cleaned string
            cleaned[key] = try_cast_number(v) if isinstance(v, (int, float, str)) else v
    return cleaned

# ---------------------------------------------------------------------
# Main conversion logic for the JSON format of the nosql data
# ---------------------------------------------------------------------

def build_listing_object(row: pd.Series, reviews_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
# Transforms a pandas Series (in this case one listing) into the cleaned, denormalized JSON object.
# We keep IDs as strings (recommended for keys) but convert numeric fields.

    r = row.to_dict()
# Converts row to dict for easier handling

    listing_id_raw = r.get("id") or r.get("listing_id") or r.get("listingid")
    listing_id = None if is_missing(listing_id_raw) else str(listing_id_raw).strip()

    detailed_reviews = reviews_map.get(listing_id, [])
# Here we gather reviews by using the string form of id as key in the map

# Now we dive into building the object's fields carefully
    listing_obj = {
        "id": listing_id,  # we are keeping it as string (safe for keys)
        "listing_url": strip_html(r.get("listing_url")),
        "name": strip_html(r.get("name")),
        "description": strip_html(r.get("description")),
        "neighbourhood": {
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
            "is_superhost": True if str(r.get("host_is_superhost")).strip().lower() in ("t", "true", "yes", "1") else False,
            "listings_count": try_cast_number(r.get("host_listings_count")),
            "verifications": parse_list_field(r.get("host_verifications")),
            "has_profile_pic": True if str(r.get("host_has_profile_pic")).strip().lower() in ("t", "true", "yes", "1") else False,
            "identity_verified": True if str(r.get("host_identity_verified")).strip().lower() in ("t", "true", "yes", "1") else False,
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
            "amenities": parse_list_field(r.get("amenities"))
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
            "detailed_reviews": detailed_reviews # keeps the raw review text and metadata right inside the listing.
        },
        "coordinates": {
            "latitude": try_cast_number(r.get("latitude")),
            "longitude": try_cast_number(r.get("longitude"))
        },
        "license": strip_html(r.get("license")),
        "instant_bookable": True if str(r.get("instant_bookable")).strip().lower() in ("t", "true", "yes", "1") else False,
        "minimum_nights": try_cast_number(r.get("minimum_nights")),
        "maximum_nights": try_cast_number(r.get("maximum_nights"))
    }

    def prune_none(d):
        if isinstance(d, dict):
            return {k: prune_none(v) for k, v in d.items() if v is not None and (not isinstance(v, dict) or any(x is not None for x in v.values()))}
        return d
    return prune_none(listing_obj)
# remove keys where all values are None, so only fields with values per object are kept 

# ------------------------------------
# Entrypoint for final JSON file
# ------------------------------------

def main(
    listings_path: str = "listings.csv",
    reviews_path: str = "reviews.csv",
    output_path: str = "listings.json",
    sample_n: int = 50,
    random_sample: bool = True
):
# This main function will perform the final operation to obtain the JSON file    

# 1) We start by loading csvs, we read everything as strings to avoid pandas float-int conversions on big ids
    listings = pd.read_csv(listings_path, low_memory=False, dtype=str)
    reviews = pd.read_csv(reviews_path, low_memory=False, dtype=str)

# 2) We build reviews map keyed by listing_id (string)
    reviews_map: Dict[str, List[Dict[str, Any]]] = {}
    for _, rev_row in reviews.iterrows():
        rev = rev_row.to_dict()
        lid_raw = rev.get("listing_id") or rev.get("listingId") or rev.get("id_listing")
        if is_missing(lid_raw):
            continue
        lid = str(lid_raw).strip()
        cleaned_rev = clean_review_dict(rev)
        reviews_map.setdefault(lid, []).append(cleaned_rev)
# For each review row:
#   - Get its listing_id 
#   - Skip if missing
#   - Clean the review with clean_review_dict()
#   - Append it to reviews_map[lid]
# Result: reviews_map = { "listing_id": [review1, review2, ...], ... }

# 3) We choose sample of listings (random or first-N)
    if random_sample:
        if len(listings) <= sample_n:
            sample_df = listings.copy()
        else:
            sample_df = listings.sample(n=sample_n, random_state=42)
    else:
        sample_df = listings.head(sample_n)
# Selects which listings to include in the sample
# - If random_sample=True:
#     - If dataset is smaller than sample_n → take all listings
#     - Else → take a reproducible random sample of size sample_n
# - If random_sample=False:
#     - Simply take the first sample_n listings

# 4) We build the JSON-ready list
    json_docs = []
    for _, row in sample_df.iterrows():
        doc = build_listing_object(row, reviews_map)
        json_docs.append(doc)      

# 5) Now, we write the JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_docs, f, indent=2, ensure_ascii=False)
# - open output_path in write mode (UTF-8 for special characters)
# - dump json_docs into the file
# - indent=2 → pretty-print for readability
# - ensure_ascii=False → keep non-ASCII chars (e.g., accents) human-readable  

    print(f"✅ Wrote {len(json_docs)} listings to {output_path}")

if __name__ == "__main__": # is the file being run directly and not imported?
    main() # calls the main function
