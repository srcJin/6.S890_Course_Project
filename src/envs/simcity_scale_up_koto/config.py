# src/envs/simcity_scale_up/config.py
# Urban Resilience-Focused SimCity Environment

# Grid parameters represent urban resilience metrics (6-parameter system):
# G = Greenery (urban green spaces, biodiversity, environmental quality)
# V = Vitality (economic activity, social vibrancy, community life)
# D = Density (population density, urban development intensity)
# A = Adaptability (climate adaptation capacity, infrastructure flexibility)
# S = Sustainability (environmental footprint, resource efficiency)
# F = Flood_Resistance (disaster preparedness, protective infrastructure)

# Universal income lifecycle for all buildings
INCOME_LIFECYCLE = {
    "duration": 25,  # Buildings generate income for 25 turns (valuable period)
    "decay_rate": 0.02,  # 2% reduction per turn from start
    "start_delay": 1,  # Income starts from turn 2 after construction
    "replacement_threshold": 0.3,  # Can be replaced when income drops below 30%
}


# Define player building utilities (ongoing income/costs)
BUILDING_UTILITIES = {
    # Basic development - immediate economic returns
    "House": {"money": 7, "reputation": 2},  # Scaled from 4/1 to match shorter duration
    "Shop": {"money": 10, "reputation": 0},  # Scaled from 6/0
    # Resilience projects - balanced or long-term benefits (scaled)
    "GreenPark": {
        "money": -2,
        "reputation": 7,
    },  # Scaled from -1/4
    "CommunityHub": {
        "money": 2,
        "reputation": 8,
    },  # Scaled from 1/5
    "SolarGrid": {
        "money": 8,
        "reputation": 3,
    },  # Scaled from 5/2
    "FloodBarrier": {
        "money": -3,
        "reputation": 5,
    },  # Scaled from -2/3
}

# Define player building effects on grid parameters (G, V, D, A, S, F)
TERRAIN_AND_PROJECTS = {
    # Basic development - mixed or negative sustainability impacts
    "Empty": {
        "id": 0,
        "type": "terrain",
        "is_buildable": True,
        "effect": {"G": 0, "V": 0, "D": 0, "A": 0, "S": 0, "F": 0},
        "cost": {"money": 0, "reputation": 0},
        "utility": {"money": 0, "reputation": 0},
        "symbol": ".",
        "description": "Buildable land",
    },
    "Water": {
        "id": 1,
        "type": "terrain",
        "is_buildable": False,
        "effect": {"G": 5, "V": -5, "D": -10, "A": 0, "S": 15, "F": 25},
        "cost": {"money": 0, "reputation": 0},
        "utility": {"money": 0, "reputation": 0},
        "symbol": "~",
        "description": "Natural water body - non-buildable",
    },
    "Road": {
        "id": 2,
        "type": "terrain",
        "is_buildable": False,
        "effect": {"G": -15, "V": 20, "D": 5, "A": 0, "S": -20, "F": 0},
        "cost": {"money": 0, "reputation": 0},
        "utility": {"money": 0, "reputation": 0},
        "symbol": "=",
        "description": "Major transportation infrastructure - non-buildable",
    },
    # Basic urban development
    "House": {
        "id": 100,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -8, "V": 10, "D": 20, "A": 5, "S": -5, "F": 0},
        "utility": {"money": 4, "reputation": 1},
        "cost": {"money": 10, "reputation": 5},
        "symbol": "H",
        "neighbors": {"G": -3, "V": 5, "D": 6, "A": 2, "S": -2, "F": 0},
        "description": "Standard residential housing",
    },
    "Shop": {
        "id": 101,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -10, "V": 25, "D": 15, "A": 5, "S": -8, "F": 0},
        "utility": {"money": 6, "reputation": 0},
        "cost": {"money": 12, "reputation": 3},
        "symbol": "S",
        "neighbors": {"G": -4, "V": 8, "D": 4, "A": 2, "S": -3, "F": 0},
        "description": "Commercial retail space",
    },
    "Office": {
        "id": 102,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -12, "V": 20, "D": 18, "A": 8, "S": -10, "F": 0},
        "cost": {"money": 20, "reputation": 7},
        "utility": {"money": 8, "reputation": 0},
        "symbol": "O",
        "neighbors": {"G": -5, "V": 6, "D": 6, "A": 3, "S": -4, "F": 0},
        "description": "Commercial office building",
    },
    "Factory": {
        "id": 103,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -20, "V": 15, "D": 25, "A": 5, "S": -18, "F": -3},
        "cost": {"money": 25, "reputation": 10},
        "utility": {"money": 10, "reputation": 2},
        "symbol": "F",
        "neighbors": {"G": -8, "V": 4, "D": 8, "A": 2, "S": -6, "F": -1},
        "description": "Industrial manufacturing facility",
    },
    # Resilience projects - strong positive impacts on urban resilience
    "Park": {
        "id": 201,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 30, "V": 15, "D": -5, "A": 12, "S": 25, "F": 8},
        "cost": {"money": 8, "reputation": 6},
        "utility": {"money": -1, "reputation": 3},
        "symbol": "P",
        "neighbors": {"G": 10, "V": 6, "D": -2, "A": 4, "S": 8, "F": 3},
        "description": "Green park for community recreation",
    },
    "Shelter": {
        "id": 202,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 15, "V": 35, "D": 10, "A": 40, "S": 20, "F": 30},
        "cost": {"money": 12, "reputation": 8},
        "utility": {"money": 1, "reputation": 4},
        "symbol": "C",
        "neighbors": {"G": 5, "V": 12, "D": 3, "A": 15, "S": 8, "F": 10},
        "description": "Community resilience center",
    },
    "Watergate": {
        "id": 203,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 10, "V": 25, "D": 5, "A": 35, "S": 45, "F": 20},
        "cost": {"money": 15, "reputation": 5},
        "utility": {"money": 3, "reputation": 2},
        "symbol": "G",
        "neighbors": {"G": 3, "V": 8, "D": 2, "A": 12, "S": 18, "F": 8},
        "description": "Renewable energy infrastructure",
    },
    "FloodBarrier": {
        "id": 204,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 5, "V": 10, "D": 0, "A": 25, "S": 15, "F": 50},
        "cost": {"money": 18, "reputation": 7},
        "utility": {"money": -2, "reputation": 3},
        "symbol": "B",
        "neighbors": {"G": 2, "V": 3, "D": 0, "A": 8, "S": 5, "F": 20},
        "description": "Climate protection infrastructure",
    },
}

# A 12x12 grid layout with terrain and pre-built projects
INITIAL_GRID = [
    [1, 103, 103, 103, 2, 103, 103, 103, 103, 1, 1, 1],
    [1, 100, 100, 100, 2, 0, 0, 0, 0, 1, 1, 1],
    [2, 100, 100, 100, 2, 0, 0, 0, 0, 1, 1, 1],
    [1, 100, 100, 100, 2, 0, 0, 0, 0, 1, 1, 1],
    [1, 103, 100, 100, 2, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 100, 0, 2, 0, 100, 100, 0, 0, 0, 1],
    [1, 1, 0, 0, 2, 0, 100, 100, 0, 0, 0, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]
