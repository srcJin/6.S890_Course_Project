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
        "effect": {"G": 8, "V": -3, "D": -10, "A": -5, "S": 10, "F": -10},
        "cost": {"money": 0, "reputation": 0},
        "utility": {"money": 0, "reputation": 0},
        "symbol": "~",
        "description": "Natural water body - flood risk area",
    },
    "Road": {
        "id": 2,
        "type": "terrain",
        "is_buildable": False,
        "effect": {"G": -18, "V": 15, "D": 8, "A": -3, "S": -25, "F": -2},
        "cost": {"money": 0, "reputation": 0},
        "utility": {"money": 0, "reputation": 0},
        "symbol": "=",
        "description": "Major transportation infrastructure - pollution and noise",
    },
    # Basic urban development
    "House": {
        "id": 100,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -8, "V": 10, "D": 20, "A": 5, "S": -5, "F": 0},
        "utility": {"money": 5, "reputation": 2},
        "cost": {"money": 12, "reputation": 4},
        "symbol": "H",
        "neighbors": {
            "G": -5,
            "V": 8,
            "D": 8,
            "A": -4,
            "S": -6,
            "F": -8,
        },  # 受水边洪水和道路污染影响，但影响减轻
        "description": "Standard residential housing",
    },
    "Shop": {
        "id": 101,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -10, "V": 25, "D": 15, "A": 5, "S": -8, "F": 0},
        "utility": {"money": 6, "reputation": 3},
        "cost": {"money": 14, "reputation": 3},
        "symbol": "S",
        "neighbors": {
            "G": -6,
            "V": 11,
            "D": 6,
            "A": -8,
            "S": -9,
            "F": -10,
        },  # 受洪水和污染影响，但交通便利
        "description": "Commercial retail space",
    },
    "Office": {
        "id": 102,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -12, "V": 20, "D": 18, "A": 8, "S": -10, "F": 0},
        "cost": {"money": 16, "reputation": 5},
        "utility": {"money": 7, "reputation": 2},
        "symbol": "O",
        "neighbors": {
            "G": -8,
            "V": 10,
            "D": 9,
            "A": -10,
            "S": -12,
            "F": -12,
        },  # 高风险但交通便利
        "description": "Commercial office building",
    },
    "Factory": {
        "id": 103,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": -20, "V": 15, "D": 25, "A": 5, "S": -18, "F": -3},
        "cost": {"money": 14, "reputation": 4},
        "utility": {"money": 8, "reputation": 1},
        "symbol": "F",
        "neighbors": {
            "G": -12,
            "V": 6,
            "D": 9,
            "A": -14,
            "S": -16,
            "F": -15,
        },  # 最高环境风险，但仍可承受
        "description": "Industrial manufacturing facility",
    },
    # Resilience projects - strong positive impacts on urban resilience
    "Park": {
        "id": 201,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 35, "V": 15, "D": -5, "A": 12, "S": 30, "F": 8},
        "cost": {"money": 10, "reputation": 4},
        "utility": {"money": 4, "reputation": 8},
        "symbol": "P",
        "neighbors": {
            "G": 3,
            "V": 6,
            "D": -2,
            "A": -6,
            "S": 2,
            "F": -3,
        },  # 公园受环境影响但有正面加成
        "description": "Green park for community recreation - Environmental resilience focus",
    },
    "Shelter": {
        "id": 202,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 15, "V": 40, "D": 10, "A": 45, "S": 20, "F": 35},
        "cost": {"money": 12, "reputation": 6},
        "utility": {"money": 5, "reputation": 10},
        "symbol": "C",
        "neighbors": {
            "G": 4,
            "V": 17,
            "D": 3,
            "A": 20,
            "S": 5,
            "F": 10,
        },  # 避难所强抗环境影响并提供正面效应
        "description": "Community resilience center - Social resilience focus",
    },
    "Watergate": {
        "id": 203,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 10, "V": 25, "D": 5, "A": 30, "S": 50, "F": 20},
        "cost": {"money": 14, "reputation": 5},
        "utility": {"money": 6, "reputation": 8},
        "symbol": "G",
        "neighbors": {
            "G": 2,
            "V": 8,
            "D": 2,
            "A": 10,
            "S": 20,
            "F": 8,
        },  # 技术设施提供显著环境改善
        "description": "Renewable energy infrastructure - Technology resilience focus",
    },
    "FloodBarrier": {
        "id": 204,
        "type": "project",
        "is_buildable": True,
        "effect": {"G": 5, "V": 10, "D": 0, "A": 30, "S": 15, "F": 55},
        "cost": {"money": 15, "reputation": 6},
        "utility": {"money": 3, "reputation": 12},
        "symbol": "B",
        "neighbors": {
            "G": 2,
            "V": 3,
            "D": 0,
            "A": 18,
            "S": 8,
            "F": 40,
        },  # 防洪设施在水边更有效，显著提升抗灾能力
        "description": "Climate protection infrastructure - Infrastructure resilience focus",
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


REALWORLD_GRID = [
    [1, 103, 103, 103, 2, 103, 103, 103, 103, 1, 1, 1],
    [1, 100, 100, 100, 2, 201, 201, 201, 201, 1, 1, 1],
    [2, 100, 100, 100, 2, 201, 201, 201, 201, 1, 1, 1],
    [1, 100, 100, 100, 2, 201, 201, 201, 201, 1, 1, 1],
    [1, 103, 100, 100, 2, 201, 201, 201, 201, 1, 1, 1],
    [1, 1, 100, 201, 2, 101, 100, 100, 201, 101, 101, 1],
    [1, 1, 100, 100, 2, 101, 100, 100, 101, 101, 101, 1],
    [203, 203, 201, 201, 2, 201, 201, 201, 201, 201, 201, 203],
    [1, 1, 201, 201, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 2, 2, 2, 2, 2, 2, 2, 103, 103, 103, 1],
    [2, 2, 2, 2, 103, 103, 103, 103, 103, 103, 103, 1],
    [2, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 1],
]
