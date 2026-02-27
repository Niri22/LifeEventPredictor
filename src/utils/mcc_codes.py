"""
MCC (Merchant Category Code) lookup table for synthetic data generation.
Grouped by spending category with realistic codes from ISO 18245.
"""

MCC_CATEGORIES = {
    "groceries": {
        "codes": [5411, 5422, 5441, 5451, 5462],
        "merchants": ["Loblaws", "Metro", "Sobeys", "No Frills", "Farm Boy", "Costco"],
        "amount_range": (15.0, 250.0),
    },
    "dining": {
        "codes": [5812, 5813, 5814],
        "merchants": ["Tim Hortons", "Starbucks", "McDonalds", "The Keg", "Earls", "Skip The Dishes"],
        "amount_range": (5.0, 150.0),
    },
    "transport": {
        "codes": [4111, 4121, 4131, 5541, 5542],
        "merchants": ["Uber", "Lyft", "Presto", "Shell", "Petro-Canada", "Esso"],
        "amount_range": (3.0, 120.0),
    },
    "travel": {
        "codes": [3000, 3001, 4511, 7011, 7012],
        "merchants": ["Air Canada", "WestJet", "Marriott", "Airbnb", "Expedia", "Porter Airlines"],
        "amount_range": (100.0, 3000.0),
    },
    "utilities": {
        "codes": [4814, 4899, 4900],
        "merchants": ["Rogers", "Bell", "Telus", "Hydro One", "Enbridge"],
        "amount_range": (50.0, 300.0),
    },
    "rent_mortgage": {
        "codes": [6513],
        "merchants": ["Landlord Payment", "RBC Mortgage", "TD Mortgage", "CMHC"],
        "amount_range": (1200.0, 3500.0),
    },
    "subscriptions": {
        "codes": [4816, 5815, 5816, 5817, 5818],
        "merchants": ["Netflix", "Spotify", "Amazon Prime", "Apple", "Disney+", "YouTube Premium"],
        "amount_range": (5.0, 25.0),
    },
    "shopping": {
        "codes": [5311, 5411, 5691, 5699, 5732],
        "merchants": ["Amazon", "Walmart", "Canadian Tire", "Hudson Bay", "Best Buy"],
        "amount_range": (20.0, 500.0),
    },
    "health": {
        "codes": [5912, 8011, 8021, 8031, 8042, 8099],
        "merchants": ["Shoppers Drug Mart", "Rexall", "LifeLabs", "Dental Office", "Optometrist"],
        "amount_range": (15.0, 400.0),
    },
    "fitness": {
        "codes": [7941, 7911],
        "merchants": ["GoodLife Fitness", "Equinox", "Yoga Studio", "CrossFit"],
        "amount_range": (30.0, 120.0),
    },
    "luxury": {
        "codes": [5094, 5944, 5945, 7298],
        "merchants": ["Holt Renfrew", "Tiffany", "Nordstrom", "Saks", "Spa & Wellness"],
        "amount_range": (100.0, 2000.0),
    },
    "financial_services": {
        "codes": [6010, 6011, 6012, 6051],
        "merchants": ["Wealthsimple", "Questrade", "TD Direct Investing", "RBC InvestEase"],
        "amount_range": (100.0, 10000.0),
    },
    "insurance": {
        "codes": [6300],
        "merchants": ["Manulife", "Sun Life", "Intact Insurance", "Aviva"],
        "amount_range": (100.0, 500.0),
    },
    "education": {
        "codes": [8220, 8241, 8244, 8249],
        "merchants": ["University of Toronto", "Coursera", "Udemy", "CFA Institute"],
        "amount_range": (50.0, 5000.0),
    },
    "professional_services": {
        "codes": [7392, 8111, 8931, 8999],
        "merchants": ["Law Office", "Accounting Firm", "Consulting Co", "Advisory Services"],
        "amount_range": (200.0, 5000.0),
    },
    "conferences_saas": {
        "codes": [7399, 5734],
        "merchants": ["Collision Conf", "AWS", "Google Cloud", "Slack", "Notion", "Figma"],
        "amount_range": (15.0, 2500.0),
    },
    "baby_family": {
        "codes": [5641, 5945, 8351],
        "merchants": ["BuyBuyBaby", "Toys R Us", "Indigo Kids", "Daycare Centre"],
        "amount_range": (20.0, 800.0),
    },
    "home_improvement": {
        "codes": [5200, 5211, 5231, 5251],
        "merchants": ["Home Depot", "Lowes", "RONA", "Home Hardware"],
        "amount_range": (30.0, 2000.0),
    },
    "real_estate": {
        "codes": [6513, 6540],
        "merchants": ["RE/MAX", "Royal LePage", "Sothebys Realty", "Property Management"],
        "amount_range": (500.0, 10000.0),
    },
}

# Flat lookup: MCC code -> category name
MCC_TO_CATEGORY: dict[int, str] = {}
for category, info in MCC_CATEGORIES.items():
    for code in info["codes"]:
        if code not in MCC_TO_CATEGORY:
            MCC_TO_CATEGORY[code] = category


def get_category(mcc: int) -> str:
    return MCC_TO_CATEGORY.get(mcc, "other")


# Spending profiles by income bracket determine how likely each MCC category is
SPEND_WEIGHTS_BY_BRACKET = {
    "low": {
        "groceries": 0.25, "dining": 0.10, "transport": 0.15, "utilities": 0.10,
        "rent_mortgage": 0.15, "subscriptions": 0.05, "shopping": 0.08,
        "health": 0.05, "fitness": 0.02, "luxury": 0.00, "travel": 0.02,
        "financial_services": 0.01, "insurance": 0.02,
    },
    "mid": {
        "groceries": 0.20, "dining": 0.12, "transport": 0.10, "utilities": 0.08,
        "rent_mortgage": 0.15, "subscriptions": 0.05, "shopping": 0.10,
        "health": 0.04, "fitness": 0.03, "luxury": 0.02, "travel": 0.05,
        "financial_services": 0.02, "insurance": 0.04,
    },
    "high": {
        "groceries": 0.12, "dining": 0.15, "transport": 0.08, "utilities": 0.05,
        "rent_mortgage": 0.12, "subscriptions": 0.04, "shopping": 0.10,
        "health": 0.04, "fitness": 0.04, "luxury": 0.06, "travel": 0.10,
        "financial_services": 0.04, "insurance": 0.04, "professional_services": 0.02,
    },
    "ultra": {
        "groceries": 0.08, "dining": 0.15, "transport": 0.05, "utilities": 0.03,
        "rent_mortgage": 0.10, "subscriptions": 0.03, "shopping": 0.08,
        "health": 0.04, "fitness": 0.04, "luxury": 0.10, "travel": 0.15,
        "financial_services": 0.06, "insurance": 0.04, "professional_services": 0.03,
        "conferences_saas": 0.02,
    },
}
