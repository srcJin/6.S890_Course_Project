# Urban Resilience SimCity Scale-Up Environment

## Overview

This is a scaled-up version of the SimCity environment focused on **urban resilience**. The environment simulates urban planning decisions with an emphasis on sustainability, community well-being, disaster preparedness, and climate adaptation. This implementation represents a significant expansion from the original 4x4 grid to an 8x8 grid with realistic urban constraints.

## Key Features

### Scaled-Up Specifications
- **Grid Size**: 8x8 (64 cells total, 41 buildable)
- **Players**: 4 agents (P1, P2, P3, P4)
- **Building Types**: 5 resilience-focused buildings
- **Grid Parameters**: 4 urban resilience metrics
- **Terrain Constraints**: 19 non-buildable locations
- **Pre-built Infrastructure**: 4 essential city facilities

### Grid Parameters (Urban Resilience Metrics)
- **S (Sustainability)**: Environmental impact, renewable energy adoption
- **W (Well-being)**: Community health, social cohesion
- **R (Resilience)**: Disaster preparedness, infrastructure adaptability
- **C (Climate)**: Carbon footprint, climate adaptation measures

### Building Types
1. **GreenPark**: Urban green infrastructure
   - Cost: $8 money, 12 reputation
   - Effects: High sustainability (40) and well-being (35)
   - Utility: -$2 money, +5 reputation (maintenance cost, social value)
   
2. **ResilientHouse**: Climate-adapted housing
   - Cost: $15 money, 8 reputation
   - Effects: Balanced resilience (30) and well-being (25)
   - Utility: +$3 money, +2 reputation (efficient housing income)
   
3. **CommunityHub**: Social resilience centers
   - Cost: $20 money, 15 reputation
   - Effects: Very high well-being (45) and resilience (35)
   - Utility: +$1 money, +4 reputation (community services)
   
4. **SolarGrid**: Renewable energy infrastructure
   - Cost: $25 money, 5 reputation
   - Effects: Very high sustainability (50) and climate adaptation (45)
   - Utility: +$6 money, +1 reputation (energy generation income)
   
5. **FloodBarrier**: Climate protection infrastructure
   - Cost: $30 money, 10 reputation
   - Effects: Very high resilience (50) and climate adaptation (30)
   - Utility: -$1 money, +3 reputation (maintenance cost, protection value)

### Terrain Constraints (Non-buildable)
- **River (~)**: Natural water bodies running through the city
- **Mountain (^)**: High elevation terrain in southern area
- **Lake (o)**: Water bodies in central area
- **Highway (=)**: Major transportation corridor (Row 3)
- **Railway (||)**: Rail transportation infrastructure

### Pre-built Infrastructure
- **Hospital (H)**: Essential healthcare facility at (1,1)
  - Effects: +40 well-being, +30 resilience, +10 sustainability
- **School (E)**: Educational institution at (1,6)
  - Effects: +35 well-being, +25 resilience, +15 sustainability
- **Fire Station (F)**: Emergency response facility at (5,1)
  - Effects: +45 resilience, +20 well-being, +5 sustainability
- **Power Plant (P)**: Traditional power generation at (5,6)
  - Effects: +30 resilience, -10 sustainability, -5 well-being
   - Maximum well-being and resilience impact
   
4. **SolarGrid**: Renewable energy infrastructure
   - Cost: $25 money, 5 reputation
   - Highest sustainability and climate benefits
   
5. **FloodBarrier**: Climate protection infrastructure
   - Cost: $30 money, 10 reputation
   - Maximum resilience and climate adaptation

### Realistic Urban Constraints

#### Non-Buildable Terrain
- **River**: Natural water body running through the city
- **Highway**: Major transportation corridor (Row 3)
- **Lake**: Water body in the southern area
- **Mountain**: High elevation terrain

#### Pre-Built Infrastructure
- **Hospital** (1,1): Essential healthcare facility
- **School** (1,6): Educational institution  
- **Fire Station** (5,1): Emergency response facility
- **Power Plant** (5,6): Traditional power generation

### Grid Layout Visualization
```
   0 1 2 3 4 5 6 7
0  . . ~ ~ ~ . . . 
1  . H . ~ ~ . E . 
2  . . . ~ ~ . . . 
3  = = = = = = = = 
4  . . . . . . . . 
5  . F . . . . P . 
6  . . . o o . . . 
7  . . . ^ ^ . . . 
```

**Legend:**
- Infrastructure: H=Hospital, E=School, F=FireStation, P=PowerPlant
- Terrain: ~=River, ==Highway, o=Lake, ^=Mountain
- .=Buildable space (41 total)

## Player Types

The environment supports 4 different player archetypes with varying priorities:

1. **Altruistic Player**: Community-focused (α=0.3, β=0.7)
2. **Balanced Player**: Balanced approach (α=0.5, β=0.5)
3. **Interest-Driven Player**: Economic efficiency (α=0.7, β=0.3)
4. **Environmental Player**: Sustainability-focused (α=0.5, β=0.5)

## Player Archetypes

The 4 players represent different urban planning perspectives:

1. **P1 - Community-Focused Player (Altruistic)**: Prioritizes well-being and social resilience (α=0.2, β=0.8)
2. **P2 - Balanced Urban Planner**: Takes a holistic approach to all metrics (α=0.5, β=0.5)  
3. **P3 - Economic Efficiency Player (Interest-Driven)**: Focuses on cost-effective development (α=0.8, β=0.2)
4. **P4 - Environmental Specialist (Balanced)**: Emphasizes sustainability and climate adaptation (α=0.5, β=0.5)

## Game Mechanics

### Resource System
- **Money**: Required for building construction and maintenance
- **Reputation**: Social capital needed for community projects
- Starting resources: 80 money, 80 reputation (scaled up)

### Scoring System
- **Self Score**: Individual player achievements
- **Integrated Score**: α × Self Score + β × Environment Score
- **Environment Score**: Weighted combination of grid resilience metrics
  - Formula: 0.3×S + 0.25×W + 0.3×R + 0.15×C

### Win Conditions
Game ends when:
- 80% of buildable spaces are occupied, OR
- 200 turns have elapsed

## Usage Examples

### Basic Environment Setup
```python
from src.envs.simcity_scale_up.environment import SimCityScaleUpEnv

# Initialize environment
env = SimCityScaleUpEnv()
obs, info = env.reset()

# Check environment properties  
print(f"Grid shape: {env.grid.shape}")     # (8, 8, 4)
print(f"Players: {env.agents}")            # ['P1', 'P2', 'P3', 'P4']
print(f"Buildings: {env.BUILDING_TYPES}")  # 5 building types
print(f"Buildable cells: {sum(1 for x in range(8) for y in range(8) if env._is_buildable(x, y))}")  # 41
```

### Action Encoding
Actions are encoded as:
- **0**: No-op (skip turn)
- **1 + building_type_index * 64 + cell_id**: Build specific building at cell

```python
# Example: Build GreenPark (type 0) at position (0,0)
# cell_id = 0*8 + 0 = 0
# action = 1 + 0*64 + 0 = 1
action_greenpark_00 = 1

# Example: Build SolarGrid (type 3) at position (4,5) 
# cell_id = 4*8 + 5 = 37
# action = 1 + 3*64 + 37 = 230
action_solargrid_45 = 1 + 3*64 + 4*8 + 5
```

### Sample Game Loop
```python
import random

# Run a complete game
env = SimCityScaleUpEnv()
obs, info = env.reset()

for step in range(100):
    current_agent = env.agent_selection
    
    # Get valid buildable locations
    buildable = [(x, y) for x in range(8) for y in range(8) 
                 if env._is_buildable(x, y)]
    
    if buildable and random.random() > 0.1:  # 90% chance to build
        # Choose random buildable location and building type
        x, y = random.choice(buildable)
        building_idx = random.randint(0, 4)
        action = 1 + building_idx * 64 + x * 8 + y
        print(f"{current_agent} building {env.BUILDING_TYPES[building_idx]} at ({x},{y})")
    else:
        action = 0  # Skip turn
        print(f"{current_agent} skipping turn")
    
    env.step(action)
    
    # Check if game is over
    if all(env.terminations.values()):
        break

# Display final results
print("\nFinal city state:")
env.render()

# Display final scores
env_score = env.calculate_environment_score()
print(f"\nEnvironment Score: {env_score['env_score']:.2f}")
for agent in env.agents:
    player = env.players[agent]
    print(f"{agent}: Score={player.integrated_score:.2f}, Money={player.resources['money']}, Reputation={player.resources['reputation']}")
```

## Grid Layout Example

```
[ ][ ][~][~][~][ ][ ][ ]  # Row 0: River
[ ][H][ ][~][~][ ][E][ ]  # Row 1: Hospital & School  
[ ][ ][ ][~][~][ ][ ][ ]  # Row 2: Development area
[=][=][=][=][=][=][=][=]  # Row 3: Highway corridor
[ ][ ][ ][ ][ ][ ][ ][ ]  # Row 4: Open development
[ ][F][ ][ ][ ][ ][P][ ]  # Row 5: Fire Station & Power Plant
[ ][ ][ ][o][o][ ][ ][ ]  # Row 6: Lake area
[ ][ ][ ][^][^][ ][ ][ ]  # Row 7: Mountain area

Legend:
[ ] = Buildable      ~ = River       o = Lake
H = Hospital         E = School      ^ = Mountain  
F = Fire Station     P = Power Plant = = Highway
G = GreenPark        R = ResilientHouse  C = CommunityHub
S = SolarGrid        F = FloodBarrier
```

## Integration with Frontend

The environment integrates with the scaled-up frontend (6s890-frontend scale_up branch):
- **8x8 grid visualization** with appropriate cell sizing
- **4 player interface** with urban resilience role descriptions
- **Building selection UI** for all 5 building types
- **Terrain and infrastructure visualization** with color coding
- **Real-time metrics display** for S, W, R, C parameters
- **Resource tracking** for all 4 players

## Research Applications

This scaled-up environment is ideal for studying:
- **Multi-agent cooperation** in urban planning contexts
- **Sustainability vs. development trade-offs**
- **Climate adaptation strategies** in city planning
- **Social equity** in resource allocation and infrastructure access
- **Resilience building** through strategic infrastructure planning
- **Constraint handling** in realistic urban development scenarios

## Testing

The environment has been thoroughly tested with:
- ✅ Environment initialization and reset
- ✅ Action encoding and decoding  
- ✅ Building placement and constraint validation
- ✅ Terrain and infrastructure effects
- ✅ Multi-agent step execution
- ✅ Score calculation and game termination
- ✅ Visualization and rendering
- ✅ Frontend integration

## Files

- `environment.py`: Main environment implementation
- `config.py`: Building types, costs, effects, and terrain definitions  
- `players.py`: Player archetype implementations
- `prebuilt_scenarios.json`: Predefined city scenarios
- `log.py`: Logging and visualization utilities
- `README_SCALE_UP.md`: This documentation
