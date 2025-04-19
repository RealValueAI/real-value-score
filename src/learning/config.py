from typing import List, Sequence

NUMERIC_FEATURES_DEFAULT: List[str] = [
    "price_per_sqm",
    "area",
    "rooms",
    "floor",
    "house_floors",
    "height",
    "area_rooms",
    "discount_value",
    "num_subways",
    "subway_min_dist",
    "subway_mean_dist",
    "center_dist_km",
]

CATEGORICAL_FEATURES_DEFAULT: List[str] = [
    "platform_id",
    "seller_type",
    "property_type",
    "category",
    "deal_type",
    "discount_status",
    "placement_paid",
    "flat_type",
    "balcony_type",
    "window_view",
    "renovation_offer",
    "building_state",
    "primary_subway",
    "center_bucket",
]

TARGETS_DEFAULT: Sequence[str] = (
    "flat_quality",
    "building_quality",
    "location_quality",
)
