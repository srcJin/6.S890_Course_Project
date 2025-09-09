## **MVP Game Design Document**

### **Goals**
1. Players (agents) need to take actions in a resource-constrained and dynamic environment by constructing different types of buildings to optimize their rewards.
2. Each player pursues personal interests (such as money and reputation) while considering the global environment score to avoid the negative impact of environmental collapse.
3. The game is designed to teach agents that cooperation can improve total scores in the long term while preserving enough competition to maintain complexity and strategic diversity.

---

### **Board**
- **Map Size:** 4x4 grid.
- **Initial State:** Empty, where each grid cell can accommodate different types of buildings.

Each grid cell is associated with the following three core indices, all represented as relative values in the range [0, 100], with an initial value of 30. These indices are dynamically updated each round based on interactions between cells:

1. **Greenery Index ($G$)**

2. **Vitality Index ($V$)**

3. **Density Index ($D$)**

---

### **Imperfect Information**
1. **Hidden Resources:**
   - Each player's money and reputation are not visible to other players.

---

### **Resource Constraints**
- Each player starts with 50 money and 50 reputation.
- When a player's resources are depleted, they must skip their turn.

---

### **Building Types**

Building are designed to affect both current cells and their surrounding cells.x

### **Park**
- **Cost**
  - Consumes 1 money
  - Consumes 3 reputation
- **Utility**
  - Reduces the builder's money by 1 per turn
  - Increases the builder's reputation by 3 per turn
- **Benefits**
  - Increases greenery index: $G +30$ for itself, $G +10$ for adjacent cells.
- **Drawbacks**
  - Suppresses vitality: $V -30$ for itself, $V -10$ for adjacent cells.

### **House**
- **Cost**
  - Consumes 2 money
  - Consumes 2 reputation
- **Utility**
  - Increases the builder's money by 2 per turn
- **Benefits**
  - Increases density index: $D +30$ for itself, $D +10$ for adjacent cells.
- **Drawbacks**
  - Decreases greenery index: $G -30$ for itself, $G -10$ for adjacent cells.

### **Shop**
- **Cost**
  - Consumes 3 money
  - Consumes 1 reputation
- **Utility**
  - Increases the builder's money by 3 per turn
  - Reduces the builder's reputation by 1 per turn
- **Benefits**
  - Increases vitality index: $V +30$ for itself, $V +10$ for adjacent cells.
- **Drawbacks**
  - Reduces residential comfort: $D -30$ for itself, $D -10$ for adjacent cells.

---

### **Players**
- **Number of Players:** 3 (can be extended to more players).
- **Player Characteristics:**
  - **Resources:** Each player has independent money and reputation.
  - **Reward Function:** Each player earns rewards based on their actions, consisting of **personal rewards** and **environment score contributions**:
    $$
    u_i = \alpha \cdot \text{self\_score}_i + \beta \cdot \text{environment\_score}
    $$
    where $\alpha$ and $\beta$ are preference weights assigned to each player at the start of the game, introducing strategic differentiation.
  - Each player tracks their **spend** and **earned** for subsequent scoring.

---

### **Player Actions**
Players take turns, and during their turn, they can perform the following actions:
0. **No-op:** No action performed.
1. **Construct:** Spend money and reputation to place a building on an empty grid cell.

---

## **Player Strategies**

### **Player A: Altrustic Player**
- **Traits:** Focuses on reputation by building houses and collaborating with parks to create high-reputation neighborhoods.
- **Reward Parameters:** alpha = 0.2, beta = 0.8

### **Player B: Balanced Player**
- **Traits:** Balances between monetary and reputation rewards, aiming to optimize both economic and environmental outcomes.
- **Reward Parameters:** alpha = 0.5, beta = 0.5

### **Player C: Interest-driven Player**
- **Traits:** Focuses on monetary rewards, pursuing short-term benefits through shop construction and high-vitality areas.
- **Reward Parameters:** alpha = 0.8, beta = 0.2

## **Scoring Mechanism**

Each player has their own self score. Now we assign all player have the same self-score calculation machanism.

$$
\text{self\_score}_i = 0.5 \cdot \text{current\_money}_i + 0.5 \cdot \text{current\_reputation}_i
$$

The environment score ($\text{Environment Score}$) is calculated based on the weighted averages of all grid cells:
$$
\text{Environment Score} = w_G \cdot \bar{G} + w_V \cdot \bar{V} + w_D \cdot \bar{D}
$$

- $\bar{G}, \bar{V}, \bar{D}$: Average indices of greenery, vitality, and density across the board.
- $w_G, w_V, w_D$: Weights, initially set to $w_G = 1/3, w_V = 1/3, w_D = 1/3$.

The final score for each player is:
$$
\text{Intergrated Score}_i = \alpha \cdot \text{self\_score}_i + \beta \cdot \text{environment\_score}
$$
Where $\alpha$ and $\beta$ is based on the player's type 

---

### **Game End Condition**
The game ends when the board is filled or a maximum turn reached.

---

### **Evaluation**
After the game ends, each player’s contribution to their objectives and the overall goals is evaluated. There is no defined "winner" in this game.

---

## **Multi-Agent Learning Objectives**

1. **Cooperation:**
   - Agents need to balance environmental scores and personal scores to learn cooperation in specific scenarios.
2. **Competition or Blocking:**
   - Agents can choose competitive strategies to reduce other players’ rewards and gain advantages.

---

## **Game Dynamics and Experimental Directions**
We aim to calculate each player’s optimal strategy and find the Course Correlated Equilibrium.
