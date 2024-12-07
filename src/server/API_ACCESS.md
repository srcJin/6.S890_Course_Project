# Observation

```bash
curl http://localhost:5000/obs
```

# Reset

```bash
curl -X POST http://localhost:5000/reset -H "Content-Type: application/json" -d '{}'
```

---

# Step
 (example for `action=0` for agent `P1`):

```bash
curl -X POST http://localhost:5000/step -H "Content-Type: application/json" -d '{
  "actions": {
    "P1": 0,
    "P2": 1,
    "P3": 2
  }
}'
```
