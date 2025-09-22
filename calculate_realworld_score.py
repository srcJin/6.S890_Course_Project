#!/usr/bin/env python3
"""
Calculate the total score for REALWORLD_GRID by counting building types
and summing their effects on urban resilience parameters (G, V, D, A, S, F).
"""

# REALWORLD_GRID from config.py
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

# Building effects from TERRAIN_AND_PROJECTS
BUILDING_EFFECTS = {
    # Terrain
    0: {"G": 0, "V": 0, "D": 0, "A": 0, "S": 0, "F": 0, "name": "Empty"},
    1: {"G": 8, "V": -3, "D": -10, "A": -5, "S": 10, "F": -10, "name": "Water"},
    2: {"G": -18, "V": 15, "D": 8, "A": -3, "S": -25, "F": -2, "name": "Road"},

    # Basic Development
    100: {"G": -8, "V": 10, "D": 20, "A": 5, "S": -5, "F": 0, "name": "House"},
    101: {"G": -10, "V": 25, "D": 15, "A": 5, "S": -8, "F": 0, "name": "Shop"},
    102: {"G": -12, "V": 20, "D": 18, "A": 8, "S": -10, "F": 0, "name": "Office"},
    103: {"G": -20, "V": 15, "D": 25, "A": 5, "S": -18, "F": -3, "name": "Factory"},

    # Resilience Projects
    201: {"G": 35, "V": 15, "D": -5, "A": 12, "S": 30, "F": 8, "name": "Park"},
    202: {"G": 15, "V": 40, "D": 10, "A": 45, "S": 20, "F": 35, "name": "Shelter"},
    203: {"G": 10, "V": 25, "D": 5, "A": 30, "S": 50, "F": 20, "name": "Watergate"},
    204: {"G": 5, "V": 10, "D": 0, "A": 30, "S": 15, "F": 55, "name": "FloodBarrier"},
}

def calculate_realworld_score():
    """Calculate total score for REALWORLD_GRID"""

    # Count each building type
    building_counts = {}
    for row in REALWORLD_GRID:
        for cell in row:
            building_counts[cell] = building_counts.get(cell, 0) + 1

    # Calculate total effects
    total_effects = {"G": 0, "V": 0, "D": 0, "A": 0, "S": 0, "F": 0}

    print("REALWORLD_GRID Building Analysis")
    print("=" * 50)
    print()

    print("Building Counts:")
    print("-" * 30)
    for building_id, count in sorted(building_counts.items()):
        name = BUILDING_EFFECTS[building_id]["name"]
        print(f"{name} (ID {building_id}): {count}")
    print()

    print("Parameter Effects by Building Type:")
    print("-" * 50)
    for building_id, count in sorted(building_counts.items()):
        effects = BUILDING_EFFECTS[building_id]
        name = effects["name"]

        print(f"{name} (×{count}):")
        for param in ["G", "V", "D", "A", "S", "F"]:
            effect_per_building = effects[param]
            total_effect = effect_per_building * count
            total_effects[param] += total_effect
            print(f"  {param}: {effect_per_building:+3d} × {count:2d} = {total_effect:+4d}")
        print()

    print("TOTAL SCORES:")
    print("=" * 30)
    parameter_names = {
        "G": "Greenery",
        "V": "Vitality",
        "D": "Density",
        "A": "Adaptability",
        "S": "Sustainability",
        "F": "Flood Resistance"
    }

    overall_total = 0
    for param in ["G", "V", "D", "A", "S", "F"]:
        score = total_effects[param]
        overall_total += score
        print(f"{param} ({parameter_names[param]:13s}): {score:+5d}")

    print("-" * 30)
    print(f"OVERALL TOTAL:                    {overall_total:+5d}")
    print()

    # Category breakdown
    terrain_score = sum(BUILDING_EFFECTS[bid]["G"] + BUILDING_EFFECTS[bid]["V"] +
                       BUILDING_EFFECTS[bid]["D"] + BUILDING_EFFECTS[bid]["A"] +
                       BUILDING_EFFECTS[bid]["S"] + BUILDING_EFFECTS[bid]["F"]
                       for bid, count in building_counts.items()
                       if bid in [0, 1, 2] for _ in range(count))

    basic_dev_score = sum(BUILDING_EFFECTS[bid]["G"] + BUILDING_EFFECTS[bid]["V"] +
                         BUILDING_EFFECTS[bid]["D"] + BUILDING_EFFECTS[bid]["A"] +
                         BUILDING_EFFECTS[bid]["S"] + BUILDING_EFFECTS[bid]["F"]
                         for bid, count in building_counts.items()
                         if bid in [100, 101, 102, 103] for _ in range(count))

    resilience_score = sum(BUILDING_EFFECTS[bid]["G"] + BUILDING_EFFECTS[bid]["V"] +
                          BUILDING_EFFECTS[bid]["D"] + BUILDING_EFFECTS[bid]["A"] +
                          BUILDING_EFFECTS[bid]["S"] + BUILDING_EFFECTS[bid]["F"]
                          for bid, count in building_counts.items()
                          if bid in [201, 202, 203, 204] for _ in range(count))

    print("Category Breakdown:")
    print("-" * 30)
    print(f"Terrain/Infrastructure: {terrain_score:+5d}")
    print(f"Basic Development:      {basic_dev_score:+5d}")
    print(f"Resilience Projects:    {resilience_score:+5d}")
    print(f"Total:                  {terrain_score + basic_dev_score + resilience_score:+5d}")

    return overall_total, total_effects

def calculate_frontend_style_score():
    """
    Calculate score the same way the frontend does - by simulating grid parameters
    for each cell and averaging across the entire grid (like parseKotoParameters)
    """
    print("\nFRONTEND-STYLE CALCULATION")
    print("=" * 50)
    print("Simulating 12x12 grid with 6 parameters per cell (G,V,D,A,S,F)")
    print()

    # Create a 12x12x6 grid to simulate what the frontend would see
    grid_parameters = []

    for i in range(12):
        for j in range(12):
            cell_id = REALWORLD_GRID[i][j]
            effects = BUILDING_EFFECTS[cell_id]

            # For each cell, store the 6 parameter values (G,V,D,A,S,F)
            cell_params = [effects["G"], effects["V"], effects["D"],
                          effects["A"], effects["S"], effects["F"]]
            grid_parameters.extend(cell_params)

    # Now calculate averages the same way as parseKotoParameters
    parameters = {"G": 0, "V": 0, "D": 0, "A": 0, "S": 0, "F": 0}
    param_names = ['G', 'V', 'D', 'A', 'S', 'F']

    # Sum all values for each parameter
    for i in range(12):
        for j in range(12):
            for param in range(6):
                index = (i * 12 + j) * 6 + param
                parameters[param_names[param]] += grid_parameters[index]

    # Average across all grid cells (144 total)
    num_cells = 144
    for key in parameters:
        parameters[key] = parameters[key] / num_cells

    print("Frontend-style Parameter Averages:")
    print("-" * 40)
    parameter_descriptions = {
        "G": "Greenery",
        "V": "Vitality",
        "D": "Density",
        "A": "Adaptability",
        "S": "Sustainability",
        "F": "Flood Resistance"
    }

    total = 0
    for param in param_names:
        avg_value = parameters[param]
        total += avg_value
        print(f"{param} ({parameter_descriptions[param]:13s}): {avg_value:+7.2f}")

    print("-" * 40)
    print(f"FRONTEND TOTAL (sum of averages):      {total:+7.2f}")
    print()

    # Also show what frontend progress bars would display
    print("Frontend Progress Bar Values (0-100 scale):")
    print("-" * 45)
    for param in param_names:
        avg_value = parameters[param]
        # Frontend normalization: (value + 50) * 0.4, clamped to 0-100
        normalized = max(0, min(100, (avg_value + 50) * 0.4))
        print(f"{param} ({parameter_descriptions[param]:13s}): {normalized:5.1f}%")

    return parameters

if __name__ == "__main__":
    # Original building-count calculation
    calculate_realworld_score()

    # Frontend-style calculation
    frontend_params = calculate_frontend_style_score()