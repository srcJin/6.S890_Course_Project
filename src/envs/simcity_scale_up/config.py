# src/envs/simcity_scale_up/config.py
# Urban Resilience-Focused SimCity Environment

# ============================================================================
# COMPETITION MECHANISM DESIGN
# ============================================================================
# This environment creates agent competition through THREE core mechanisms:
#
# 1. SPACE COMPETITION (Zero-Sum)
#    - 41 buildable cells for 4 agents = ~10 buildings per agent
#    - First-mover advantage: occupying prime locations blocks others
#    - Strategic positioning: neighbor effects create spatial dependencies
#
# 2. RESOURCE COMPETITION (Economic Trade-offs)
#    - Initial: 50 money + 50 reputation per agent
#    - Economic buildings (House/Shop/Factory): High profit, low reputation cost
#    - Resilience buildings (GreenPark/SolarGrid/FloodBarrier): High reputation cost
#    - Key tension: Early economic growth vs late-game reputation requirements
#
# 3. SCORING COMPETITION (Mixed Objectives)
#    - InterestDrivenPlayer: 80% self_score + 20% env_score
#    - AltruisticPlayer: 20% self_score + 80% env_score
#    - BalancedPlayer: 50% self_score + 50% env_score
#    - Conflict: Individual profit vs collective environmental score
#
# STRATEGIC DEPTH:
# - Phase 1 (Cells 0-15): Economic expansion race
# - Phase 2 (Cells 16-30): Resource accumulation & positioning
# - Phase 3 (Cells 31-41): Reputation conversion & environmental restoration
# ============================================================================

# Define player building types (6 types) - mix of basic development and resilience projects
BUILDING_TYPES = [
    # Basic urban development (economic focus, environmental cost)
    "House",
    "Shop",
    "Factory",
    # Resilience-focused projects (environmental benefits, economic trade-offs)
    "GreenPark",
    "SolarGrid",
    "FloodBarrier",
]

# Grid parameters represent urban resilience metrics:
# S = Sustainability (environmental impact, renewable energy)
# W = Well-being (community health, social cohesion)
# R = Resilience (disaster preparedness, adaptability)
# C = Climate (carbon footprint, climate adaptation)

# Define player building costs (money and reputation required)
# COMPETITION KEY: Reputation becomes bottleneck - force agents to compete for reputation sources
BUILDING_COSTS = {
    # Economic buildings: Low money cost, minimal reputation (accessible early game)
    "House": {"money": 8, "reputation": 3},       # Cheap entry point
    "Shop": {"money": 12, "reputation": 2},       # Low barrier economic building
    "Factory": {"money": 16, "reputation": 0},    # Zero reputation allows pure economic play
    # Resilience buildings: Higher cost, HIGH reputation barrier (requires preparation)
    "GreenPark": {"money": 12, "reputation": 8},   # Moderate barrier
    "SolarGrid": {"money": 18, "reputation": 12},  # High reputation requirement
    "FloodBarrier": {"money": 16, "reputation": 15}, # Highest reputation barrier
}

# Define player building utilities (ongoing income/costs)
# COMPETITION KEY: Utilities create strategic dilemma between money and reputation
BUILDING_UTILITIES = {
    # Economic buildings: Positive money, negative/low reputation (encourage early building)
    "House": {"money": 3, "reputation": 0},        # Pure economic, neutral reputation
    "Shop": {"money": 5, "reputation": -1},        # High profit but reputation decay
    "Factory": {"money": 7, "reputation": -2},     # Highest profit but severe reputation loss
    # Resilience buildings: Lower money, high reputation (encourage late-game transition)
    "GreenPark": {"money": 1, "reputation": 4},    # Small income + strong reputation gain
    "SolarGrid": {"money": 2, "reputation": 3},    # Balanced income + reputation
    "FloodBarrier": {"money": 0, "reputation": 5},  # Pure reputation focus
}

# Define player building effects on grid parameters (S, W, R, C)
# Balanced for 41 buildable cells: early buildings damage environment, resilience buildings restore it
BUILDING_EFFECTS = {
    # Basic development - strong negative environmental impacts
    "House": {
        "S": -8, "W": 12, "R": 8, "C": -6,
        "neighbors": {"S": -3, "W": 4, "R": 2, "C": -2},
    },
    "Shop": {
        "S": -12, "W": 6, "R": 4, "C": -10,
        "neighbors": {"S": -4, "W": 2, "R": 1, "C": -3},
    },
    "Factory": {
        "S": -20, "W": -5, "R": 10, "C": -18,
        "neighbors": {"S": -8, "W": -2, "R": 2, "C": -6},
    },
    # Resilience projects - moderate positive impacts (not overpowered)
    "GreenPark": {
        "S": 18, "W": 20, "R": 10, "C": 15,
        "neighbors": {"S": 6, "W": 6, "R": 3, "C": 5},
    },
    "SolarGrid": {
        "S": 25, "W": 5, "R": 12, "C": 22,
        "neighbors": {"S": 8, "W": 2, "R": 4, "C": 7},
    },
    "FloodBarrier": {
        "S": 8, "W": 10, "R": 25, "C": 20,
        "neighbors": {"S": 3, "W": 3, "R": 8, "C": 6},
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
# COMPETITION KEY: PreBuilt projects create ASYMMETRIC starting conditions
# - North region (Hospital/School): High W+R values → favors resilience buildings
# - South region (FireStation/PowerPlant): Mixed values → economic vs environmental tension
PREBUILT_PROJECTS = {
    "Hospital": {
        "symbol": "H",
        "effects": {"S": 8, "W": 35, "R": 25, "C": 5},
        "neighbor_effects": {"S": 3, "W": 12, "R": 8, "C": 2},
        "description": "Healthcare hub - boosts Well-being and Resilience in north region",
        "category": "healthcare",
    },
    "School": {
        "symbol": "E",
        "effects": {"S": 12, "W": 30, "R": 20, "C": 8},
        "neighbor_effects": {"S": 5, "W": 10, "R": 6, "C": 3},
        "description": "Education center - creates balanced growth in northeast",
        "category": "education",
    },
    "FireStation": {
        "symbol": "F",
        "effects": {"S": 5, "W": 15, "R": 35, "C": 5},
        "neighbor_effects": {"S": 2, "W": 5, "R": 12, "C": 2},
        "description": "Emergency services - creates high-Resilience zone in southwest",
        "category": "emergency",
    },
    "PowerPlant": {
        "symbol": "P",
        "effects": {"S": -15, "W": -8, "R": 25, "C": -20},
        "neighbor_effects": {"S": -6, "W": -3, "R": 8, "C": -8},
        "description": "Legacy fossil fuel plant - POLLUTES southeast (S/C penalties)",
        "category": "energy",
    },
}

# Default 8x8 grid layout with terrain and pre-built projects
# 0 = buildable space, 1 = non-buildable terrain, 2 = pre-built city projects
#
# REGIONAL COMPETITION ZONES (after PreBuilt effects applied):
#
# NORTH (Rows 0-2): "Premium Development Zone"
#   - Hospital(1,1) + School(1,6) effects → High W+R baseline
#   - 14 buildable cells, fought over for resilience building synergy
#   - Best for: GreenPark, FloodBarrier (amplify existing high W/R)
#
# SOUTH (Rows 4-7): "Industrial Reclamation Zone"
#   - PowerPlant(5,6) pollution → Low S+C baseline
#   - FireStation(5,1) → High R but needs S+C repair
#   - 27 buildable cells, larger but environmentally challenged
#   - Early: Cheap Factory/Shop (already polluted, less marginal damage)
#   - Late: SolarGrid opportunity (restore S+C near PowerPlant)
#
# Highway(Row 3): Divides city, creates strategic barrier
DEFAULT_GRID_LAYOUT = [
    [0, 0, 1, 1, 1, 0, 0, 0],  # Row 0: River running through
    [0, 2, 0, 1, 1, 0, 2, 0],  # Row 1: Hospital(1,1) and School(1,6)
    [0, 0, 0, 1, 1, 0, 0, 0],  # Row 2: North zone buildable
    [1, 1, 1, 1, 1, 1, 1, 1],  # Row 3: Highway - strategic divider
    [0, 0, 0, 0, 0, 0, 0, 0],  # Row 4: South zone open area
    [0, 2, 0, 0, 0, 0, 2, 0],  # Row 5: FireStation(5,1) and PowerPlant(5,6)
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
# COMPETITION KEY: Strategic locations near prebuilt projects have higher value
# - Hospital (1,1) & School (1,6): High W+R bonuses → prime locations for resilience buildings
# - PowerPlant (5,6): Negative S+C → opportunity to build SolarGrid nearby for contrast
# - FireStation (5,1): High R bonus → synergy with FloodBarrier
DEFAULT_PREBUILT_ASSIGNMENT = {
    (1, 1): "Hospital",
    (1, 6): "School",
    (5, 1): "FireStation",
    (5, 6): "PowerPlant",
}

# ============================================================================
# INITIAL MAP STATE (After PreBuilt Effects Applied)
# ============================================================================
#
# Baseline everywhere: S=20, W=25, R=15, C=10
#
# NORTH REGION (Rows 0-2):
#   Cell (0,0): S=20, W=25, R=15, C=10 (untouched)
#   Cell (0,1): S=23, W=37, R=23, C=12 (Hospital neighbor)
#   Cell (1,0): S=23, W=37, R=23, C=12 (Hospital neighbor)
#   Cell (1,2): S=23, W=37, R=23, C=12 (Hospital neighbor)
#   Cell (1,1): S=28, W=60, R=40, C=15 (Hospital direct) [NON-BUILDABLE]
#   Cell (1,6): S=32, W=55, R=35, C=18 (School direct) [NON-BUILDABLE]
#   Cell (1,5): S=25, W=35, R=21, C=13 (School neighbor)
#   Cell (2,6): S=25, W=35, R=21, C=13 (School neighbor)
#
# SOUTH REGION (Rows 4-7):
#   Cell (5,1): S=25, W=40, R=50, C=15 (FireStation direct) [NON-BUILDABLE]
#   Cell (4,1): S=22, W=30, R=27, C=12 (FireStation neighbor)
#   Cell (5,0): S=22, W=30, R=27, C=12 (FireStation neighbor)
#   Cell (5,6): S=5, W=17, R=40, C=-10 (PowerPlant direct) [NON-BUILDABLE]
#   Cell (4,6): S=14, W=22, R=23, C=2 (PowerPlant neighbor) <- POLLUTED!
#   Cell (5,5): S=14, W=22, R=23, C=2 (PowerPlant neighbor) <- POLLUTED!
#   Cell (5,7): S=14, W=22, R=23, C=2 (PowerPlant neighbor) <- POLLUTED!
#
# KEY COMPETITIVE IMPLICATIONS:
# 1. North cells start with W=35-37 (vs baseline 25) -> GreenPark synergy
# 2. PowerPlant neighbors start with C=2 (vs baseline 10) -> SolarGrid opportunity
# 3. Asymmetric value: North 14 cells (premium) vs South 27 cells (cheaper but polluted)
#
# ============================================================================
# EXPECTED COMPETITIVE DYNAMICS
# ============================================================================
#
# EARLY GAME (Moves 1-16, ~40% filled):
# - InterestDriven: Rush South for Factory/Shop (already polluted, less marginal harm)
# - Altruistic: Compete for North premium spots (amplify existing high W/R)
# - Balanced: Split between regions
# - Strategic tension: North spots scarce but valuable, South abundant but damaged
#
# MID GAME (Moves 17-32, ~80% filled):
# - InterestDriven: Cash-rich but reputation-poor (Factory -2/turn penalty)
# - Altruistic: Can afford FloodBarrier in North (boost already-high R to 75+)
# - Regional competition: Last North spots command premium
#
# LATE GAME (Moves 33-41, 100% filled):
# - InterestDriven: Locked out of resilience buildings (need 12-15 reputation)
# - Altruistic: Build SolarGrid near PowerPlant (restore C from 2 to 24+)
# - Outcome: Altruistic boosts env_score via regional restoration strategy
# - Winner: Depends on alpha/beta weights and early positional advantage
# ============================================================================
