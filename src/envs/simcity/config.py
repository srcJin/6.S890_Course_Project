# config.py

# Define building types
BUILDING_TYPES = ["Park", "House", "Shop"]

# Define building costs
BUILDING_COSTS = {
    "Park": {"money": 1, "reputation": 3},
    "House": {"money": 2, "reputation": 2},
    "Shop": {"money": 3, "reputation": 1},
}

# Define building utilities
BUILDING_UTILITIES = {
    "Park": {"money": -1, "reputation": 3},
    "House": {"money": 2, "reputation": 0},
    "Shop": {"money": 3, "reputation": -1},
}

# Define building effects
BUILDING_EFFECTS = {
    "Park": {"G": 30, "V": -30, "D": 0, "neighbors": {"G": 10, "V": -10, "D": 0}},
    "House": {"G": -30, "V": 0, "D": 30, "neighbors": {"G": -10, "V": 10, "D": 10}},
    "Shop": {"G": -30, "V": 30, "D": -30, "neighbors": {"G": -10, "V": 10, "D": -10}},
}
