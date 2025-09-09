#!/usr/bin/env python3
"""
Test script to verify the koto training setup works correctly
Tests the environment registration and basic training initialization
"""

import sys
import os
sys.path.insert(0, 'src')

def test_koto_training_setup():
    """Test that the koto environment is properly registered and can be used for training"""
    print("🧪 Testing Koto Training Setup...")
    
    try:
        # Test environment registration
        print("📝 Testing environment registration...")
        from envs import REGISTRY
        
        if "simcity_scale_up_koto" in REGISTRY:
            print("✅ Koto environment registered successfully")
        else:
            print("❌ Koto environment not found in registry")
            print(f"Available environments: {list(REGISTRY.keys())}")
            return False
        
        # Test environment creation
        print("🏗️ Testing environment creation...")
        env_fn = REGISTRY["simcity_scale_up_koto"]
        env = env_fn(
            grid_x=12,
            grid_y=12,
            common_reward=False,
            reward_scalarisation="sum"
        )
        
        print(f"✅ Environment created successfully")
        print(f"   - Agents: {env.n_agents}")
        print(f"   - Observation size: {env.get_obs_size()}")
        print(f"   - State size: {env.get_state_size()}")
        print(f"   - Action size: {env.get_total_actions()}")
        print(f"   - Episode limit: {env.episode_limit}")
        
        # Test a few environment steps
        print("👣 Testing environment execution...")
        env.reset()
        
        # Test 3 random steps
        import numpy as np
        for step in range(3):
            actions = np.random.randint(0, env.get_total_actions(), size=env.n_agents)
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(f"   Step {step + 1}: obs_shape={obs.shape}, rewards={rewards.shape if hasattr(rewards, 'shape') else type(rewards)}")
            
            if terminated or truncated:
                print(f"   Episode ended early: terminated={terminated}, truncated={truncated}")
                break
        
        env.close()
        
        # Test configuration loading
        print("⚙️ Testing configuration loading...")
        config_path = "src/config/envs/simcity_scale_up_koto.yaml"
        if os.path.exists(config_path):
            print("✅ Koto environment config file exists")
            with open(config_path, 'r') as f:
                config_content = f.read()
                if "simcity_scale_up_koto" in config_content:
                    print("✅ Config file contains correct environment name")
                else:
                    print("❌ Config file missing environment name")
                    return False
        else:
            print("❌ Koto environment config file missing")
            return False
        
        print("\n✅ All training setup tests passed! Ready for MAPPO training.")
        print("🚀 You can now run: ./train_scale_up_koto_mappo.sh")
        return True
        
    except Exception as e:
        print(f"\n❌ Training setup test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_koto_training_setup()
    sys.exit(0 if success else 1)