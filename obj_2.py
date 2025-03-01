from datasets import load_dataset


# shout-out to chatgpt for translating from screenshot
legend_pptx = [
    "Low Density Residential Zone",
    "Medium Density Residential Zone",
    "High Density Residential Zone",
    "Commercial Zone",
    "Industrial Zone",
    "Rural and Agricultural Conservative Zone",
    "Rural and Agricultural Zone",
    "Historical Protection Zone",
    "Military Zone",
    "Government Institutes, Public Utilities and Amenities Zone"
]

land_cover_map = load_dataset("pints-sig/Land_Cover_Map_2023_10m_classified_pixels_GB")
print(land_cover_map)