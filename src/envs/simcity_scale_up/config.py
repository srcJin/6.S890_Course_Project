# src/envs/simcity_scale_up/config.py
# Urban Resilience-Focused SimCity Environment

# Define building types (5 types) focused on urban resilience
BUILDING_TYPES = [
    "GreenPark",
    "ResilientHouse",
    "CommunityHub",
    "SolarGrid",
    "FloodBarrier",
]

# Grid parameters represent urban resilience metrics:
# S = Sustainability (environmental impact, renewable energy)
# W = Well-being (community health, social cohesion)
# R = Resilience (disaster preparedness, adaptability)
# C = Climate (carbon footprint, climate adaptation)

# Define building costs (money and reputation required)
BUILDING_COSTS = {
    "GreenPark": {"money": 8, "reputation": 12},  # Urban green infrastructure
    "ResilientHouse": {"money": 15, "reputation": 8},  # Climate-adapted housing
    "CommunityHub": {"money": 20, "reputation": 15},  # Social resilience center
    "SolarGrid": {"money": 25, "reputation": 5},  # Renewable energy infrastructure
    "FloodBarrier": {
        "money": 30,
        "reputation": 10,
    },  # Climate protection infrastructure
}

# Define building utilities (ongoing income/costs)
BUILDING_UTILITIES = {
    "GreenPark": {
        "money": -2,
        "reputation": 5,
    },  # Maintenance cost but high social value
    "ResilientHouse": {
        "money": 3,
        "reputation": 2,
    },  # Efficient housing generates value
    "CommunityHub": {
        "money": 1,
        "reputation": 4,
    },  # Community services, moderate income
    "SolarGrid": {"money": 6, "reputation": 1},  # Energy generation income
    "FloodBarrier": {
        "money": -1,
        "reputation": 3,
    },  # Maintenance cost but protection value
}

# Define building effects on grid parameters (S, W, R, C)
BUILDING_EFFECTS = {
    "GreenPark": {
        "S": 40,
        "W": 35,
        "R": 20,
        "C": 25,
        "neighbors": {"S": 15, "W": 12, "R": 8, "C": 10},
    },
    "ResilientHouse": {
        "S": 15,
        "W": 25,
        "R": 30,
        "C": 20,
        "neighbors": {"S": 8, "W": 10, "R": 12, "C": 8},
    },
    "CommunityHub": {
        "S": 10,
        "W": 45,
        "R": 35,
        "C": 5,
        "neighbors": {"S": 5, "W": 18, "R": 15, "C": 3},
    },
    "SolarGrid": {
        "S": 50,
        "W": 5,
        "R": 25,
        "C": 45,
        "neighbors": {"S": 20, "W": 2, "R": 10, "C": 18},
    },
    "FloodBarrier": {
        "S": 5,
        "W": 10,
        "R": 50,
        "C": 30,
        "neighbors": {"S": 2, "W": 5, "R": 20, "C": 12},
    },
}

# Define non-buildable terrain types for realistic urban constraints
TERRAIN_TYPES = {
    "River": {"symbol": "~", "description": "Natural water body - non-buildable"},
    "Mountain": {
        "symbol": "^",
        "description": "High elevation terrain - non-buildable",
    },
    "Lake": {"symbol": "o", "description": "Water body - non-buildable"},
    "Highway": {
        "symbol": "=",
        "description": "Major transportation infrastructure - non-buildable",
    },
    "Railway": {"symbol": "||", "description": "Rail transportation - non-buildable"},
}

# Define pre-built infrastructure that starts on the map
PREBUILT_INFRASTRUCTURE = {
    "Hospital": {
        "symbol": "H",
        "effects": {"S": 10, "W": 40, "R": 30, "C": 5},
        "neighbor_effects": {"S": 5, "W": 15, "R": 10, "C": 2},
        "description": "Essential healthcare facility",
    },
    "School": {
        "symbol": "E",
        "effects": {"S": 15, "W": 35, "R": 25, "C": 10},
        "neighbor_effects": {"S": 8, "W": 12, "R": 8, "C": 3},
        "description": "Educational institution",
    },
    "FireStation": {
        "symbol": "F",
        "effects": {"S": 5, "W": 20, "R": 45, "C": 5},
        "neighbor_effects": {"S": 2, "W": 8, "R": 20, "C": 2},
        "description": "Emergency response facility",
    },
    "PowerPlant": {
        "symbol": "P",
        "effects": {"S": -10, "W": -5, "R": 30, "C": -15},
        "neighbor_effects": {"S": -5, "W": -2, "R": 10, "C": -8},
        "description": "Traditional power generation facility",
    },
}

# Default 8x8 grid layout with terrain and pre-built infrastructure
# 0 = buildable, 1 = non-buildable terrain, 2 = pre-built infrastructure
DEFAULT_GRID_LAYOUT = [
    [0, 0, 1, 1, 1, 0, 0, 0],  # Row 0: River running through
    [0, 2, 0, 1, 1, 0, 2, 0],  # Row 1: Hospital and School
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 2:
    [1, 1, 1, 1, 1, 1, 1, 1],  # Row 3: Highway/Railway corridor
    [0, 0, 0, 0, 0, 0, 0, 0],  # Row 4: Open development area
    [0, 2, 0, 0, 0, 0, 2, 0],  # Row 5: Fire Station and Power Plant
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 6: Mountain/Lake area
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 7: Mountain/Lake area
]

# Specific terrain and infrastructure assignments for the default layout
DEFAULT_TERRAIN_ASSIGNMENT = {
    # Non-buildable terrain (type 1)
    (0, 2): "River",
    (0, 3): "River",
    (0, 4): "River",
    (1, 3): "River",
    (1, 4): "River",
    (2, 3): "River",
    (2, 4): "River",
    (3, 0): "Highway",
    (3, 1): "Highway",
    (3, 2): "Highway",
    (3, 3): "Highway",
    (3, 4): "Highway",
    (3, 5): "Highway",
    (3, 6): "Highway",
    (3, 7): "Highway",
    (6, 3): "Lake",
    (6, 4): "Lake",
    (7, 3): "Mountain",
    (7, 4): "Mountain",
}

# Pre-built infrastructure assignments (type 2)
DEFAULT_INFRASTRUCTURE_ASSIGNMENT = {
    (1, 1): "Hospital",
    (1, 6): "School",
    (5, 1): "FireStation",
    (5, 6): "PowerPlant",
}
