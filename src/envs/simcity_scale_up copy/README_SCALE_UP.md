# Urban Resilience SimCity Scale-Up Environment

## Overview

This is a scaled-up version of the SimCity environment focused on urban resilience. The environment simulates urban planning decisions with an emphasis on sustainability, community well-being, disaster preparedness, and climate adaptation.

## Key Features

### Scaled-Up Specifications
- **Grid Size**: 8x8 (64 cells total, 41 buildable)
- **Players**: 4 agents (P1, P2, P3, P4)
- **Building Types**: 5 resilience-focused buildings
- **Grid Parameters**: 4 urban resilience metrics

### Grid Parameters (Urban Resilience Metrics)
- **S (Sustainability)**: Environmental impact, renewable energy adoption
- **W (Well-being)**: Community health, social cohesion
- **R (Resilience)**: Disaster preparedness, infrastructure adaptability
- **C (Climate)**: Carbon footprint, climate adaptation measures

### Building Types
1. **GreenPark**: Urban green infrastructure
   - Cost: $8 money, 12 reputation
   - High sustainability and well-being impact
   
2. **ResilientHouse**: Climate-adapted housing
   - Cost: $15 money, 8 reputation
   - Balanced resilience and well-being benefits
   
3. **CommunityHub**: Social resilience centers
   - Cost: $20 money, 15 reputation
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

## Key Differences from Original

1. **Scale**: 8x8 grid vs 4x4 (4x larger)
2. **Players**: 4 agents vs 3
3. **Buildings**: 5 types vs 3, all resilience-focused
4. **Parameters**: 4 resilience metrics vs 3 generic (G,V,D)
5. **Realism**: Non-buildable terrain and pre-built infrastructure
6. **Complexity**: More strategic depth with constrained buildable areas

This environment emphasizes sustainable urban development, community well-being, disaster preparedness, and climate adaptation.
