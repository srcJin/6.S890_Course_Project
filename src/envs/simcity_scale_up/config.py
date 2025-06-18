# src/envs/simcity_scale_up/config.py
# Urban Resilience-Focused SimCity Environment

# Define player building types (6 types) - mix of basic development and resilience projects
BUILDING_TYPES = [
    # Basic urban development
    "House",
    "Shop", 
    # Resilience-focused projects
    "GreenPark",
    "CommunityHub",
    "SolarGrid",
    "FloodBarrier",
]

# Grid parameters represent urban resilience metrics:
# S = Sustainability (environmental impact, renewable energy)
# W = Well-being (community health, social cohesion)
# R = Resilience (disaster preparedness, adaptability)
# C = Climate (carbon footprint, climate adaptation)

# Define player building costs (money and reputation required)
BUILDING_COSTS = {
    # Basic development - lower cost, traditional approach
    "House": {"money": 10, "reputation": 5},  # Standard residential housing
    "Shop": {"money": 12, "reputation": 3},  # Commercial retail space
    # Resilience projects - higher cost, better long-term benefits
    "GreenPark": {"money": 15, "reputation": 12},  # Urban green infrastructure
    "CommunityHub": {"money": 20, "reputation": 15},  # Social resilience center
    "SolarGrid": {"money": 25, "reputation": 8},  # Renewable energy infrastructure
    "FloodBarrier": {"money": 30, "reputation": 10},  # Climate protection infrastructure
}

# Define player building utilities (ongoing income/costs)
BUILDING_UTILITIES = {
    # Basic development - immediate economic returns
    "House": {"money": 4, "reputation": 1},  # Rental income, basic community value
    "Shop": {"money": 6, "reputation": 0},  # Commercial profits, neutral reputation
    # Resilience projects - balanced or long-term benefits
    "GreenPark": {"money": -1, "reputation": 4},  # Maintenance cost but high social value
    "CommunityHub": {"money": 1, "reputation": 5},  # Community services, high social impact
    "SolarGrid": {"money": 5, "reputation": 2},  # Energy generation income, clean reputation
    "FloodBarrier": {"money": -2, "reputation": 3},  # Maintenance cost but protection value
}

# Define player building effects on grid parameters (S, W, R, C)
BUILDING_EFFECTS = {
    # Basic development - mixed or negative sustainability impacts
    "House": {
        "S": -10, "W": 20, "R": 15, "C": -5,
        "neighbors": {"S": -5, "W": 8, "R": 5, "C": -2},
    },
    "Shop": {
        "S": -15, "W": 10, "R": 5, "C": -10,
        "neighbors": {"S": -8, "W": 5, "R": 2, "C": -5},
    },
    # Resilience projects - strong positive impacts on urban resilience
    "GreenPark": {
        "S": 40, "W": 35, "R": 20, "C": 30,
        "neighbors": {"S": 15, "W": 12, "R": 8, "C": 12},
    },
    "CommunityHub": {
        "S": 10, "W": 45, "R": 35, "C": 5,
        "neighbors": {"S": 5, "W": 18, "R": 15, "C": 3},
    },
    "SolarGrid": {
        "S": 50, "W": 5, "R": 25, "C": 45,
        "neighbors": {"S": 20, "W": 2, "R": 10, "C": 18},
    },
    "FloodBarrier": {
        "S": 5, "W": 10, "R": 50, "C": 35,
        "neighbors": {"S": 2, "W": 5, "R": 20, "C": 15},
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

# Define pre-built city projects that exist at game start and influence urban development
# These are city-owned projects that affect the urban resilience metrics
PREBUILT_PROJECTS = {
    "Hospital": {
        "symbol": "H",
        "effects": {"S": 10, "W": 40, "R": 30, "C": 5},
        "neighbor_effects": {"S": 5, "W": 15, "R": 10, "C": 2},
        "description": "City hospital providing essential healthcare services",
        "category": "healthcare",
    },
    "School": {
        "symbol": "E",
        "effects": {"S": 15, "W": 35, "R": 25, "C": 10},
        "neighbor_effects": {"S": 8, "W": 12, "R": 8, "C": 3},
        "description": "Public school enhancing community education",
        "category": "education",
    },
    "FireStation": {
        "symbol": "F",
        "effects": {"S": 5, "W": 20, "R": 45, "C": 5},
        "neighbor_effects": {"S": 2, "W": 8, "R": 20, "C": 2},
        "description": "Emergency services boosting city resilience",
        "category": "emergency",
    },
    "PowerPlant": {
        "symbol": "P",
        "effects": {"S": -10, "W": -5, "R": 30, "C": -15},
        "neighbor_effects": {"S": -5, "W": -2, "R": 10, "C": -8},
        "description": "Legacy power plant (players can build cleaner alternatives)",
        "category": "energy",
    },
}

# Default 8x8 grid layout with terrain and pre-built projects
# 0 = buildable space for player projects, 1 = non-buildable terrain, 2 = pre-built city projects
DEFAULT_GRID_LAYOUT = [
    [0, 0, 1, 1, 1, 0, 0, 0],  # Row 0: River running through
    [0, 2, 0, 1, 1, 0, 2, 0],  # Row 1: Hospital and School
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 2:
    [1, 1, 1, 1, 1, 1, 1, 1],  # Row 3: Highway corridor
    [0, 0, 0, 0, 0, 0, 0, 0],  # Row 4: Open development area for player projects
    [0, 2, 0, 0, 0, 0, 2, 0],  # Row 5: Fire Station and Power Plant
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 6: Lake area
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 7: Mountain area
]

# Specific terrain assignments for the default layout
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

# Pre-built city project assignments (type 2)
DEFAULT_PREBUILT_ASSIGNMENT = {
    (1, 1): "Hospital",
    (1, 6): "School",
    (5, 1): "FireStation",
    (5, 6): "PowerPlant",
}
