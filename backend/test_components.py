#!/usr/bin/env python3
"""Test script to verify all RL-GYM backend components."""
import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from rl_core import BasePolicy, MLP, set_seed, ReplayBuffer
        print("  ✓ rl_core base modules")
    except Exception as e:
        print(f"  ✗ rl_core base modules: {e}")
        return False
    
    try:
        from rl_core.algorithms import DQN, PPO, SAC, A2C
        print("  ✓ rl_core algorithms")
    except Exception as e:
        print(f"  ✗ rl_core algorithms: {e}")
        return False
    
    try:
        from environments import GridWorldEnv, NavigationEnv, PlatformerEnv
        print("  ✓ environments")
    except Exception as e:
        print(f"  ✗ environments: {e}")
        return False
    
    try:
        from nlp import GoalParser, RewardGenerator
        print("  ✓ nlp modules")
    except Exception as e:
        print(f"  ✗ nlp modules: {e}")
        return False
    
    try:
        from datasets import DatasetLoader, DatasetValidator, DatasetPreprocessor
        print("  ✓ datasets modules")
    except Exception as e:
        print(f"  ✗ datasets modules: {e}")
        return False
    
    try:
        from training import TrainingSession, TrainingManager, TrainingCallback
        print("  ✓ training modules")
    except Exception as e:
        print(f"  ✗ training modules: {e}")
        return False
    
    try:
        from api import app
        print("  ✓ api module")
    except Exception as e:
        print(f"  ✗ api module: {e}")
        return False
    
    return True


def test_environment_creation():
    """Test environment creation and basic operations."""
    print("\nTesting environment creation...")
    
    try:
        from environments import GridWorldEnv
        env = GridWorldEnv(size=5)
        obs, info = env.reset()
        print(f"  ✓ GridWorldEnv created, observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✓ GridWorldEnv step executed")
        env.close()
    except Exception as e:
        print(f"  ✗ GridWorldEnv: {e}")
        traceback.print_exc()
        return False
    
    try:
        from environments import NavigationEnv
        env = NavigationEnv(width=10, height=10)
        obs, info = env.reset()
        print(f"  ✓ NavigationEnv created, observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✓ NavigationEnv step executed")
        env.close()
    except Exception as e:
        print(f"  ✗ NavigationEnv: {e}")
        traceback.print_exc()
        return False
    
    try:
        from environments import PlatformerEnv
        env = PlatformerEnv(width=20, height=10)
        obs, info = env.reset()
        print(f"  ✓ PlatformerEnv created, observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✓ PlatformerEnv step executed")
        env.close()
    except Exception as e:
        print(f"  ✗ PlatformerEnv: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_algorithm_creation():
    """Test algorithm creation."""
    print("\nTesting algorithm creation...")
    
    import numpy as np
    
    try:
        from rl_core.algorithms import DQN
        dqn = DQN(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = dqn.act(state)
        print(f"  ✓ DQN created, action: {action}")
    except Exception as e:
        print(f"  ✗ DQN: {e}")
        traceback.print_exc()
        return False
    
    try:
        from rl_core.algorithms import PPO
        ppo = PPO(state_dim=4, action_dim=2, continuous=False)
        state = np.random.randn(4).astype(np.float32)
        action = ppo.act(state)
        print(f"  ✓ PPO created, action: {action}")
    except Exception as e:
        print(f"  ✗ PPO: {e}")
        traceback.print_exc()
        return False
    
    try:
        from rl_core.algorithms import SAC
        sac = SAC(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = sac.act(state)
        print(f"  ✓ SAC created, action: {action}")
    except Exception as e:
        print(f"  ✗ SAC: {e}")
        traceback.print_exc()
        return False
    
    try:
        from rl_core.algorithms import A2C
        a2c = A2C(state_dim=4, action_dim=2, continuous=False)
        state = np.random.randn(4).astype(np.float32)
        action = a2c.act(state)
        print(f"  ✓ A2C created, action: {action}")
    except Exception as e:
        print(f"  ✗ A2C: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_nlp():
    """Test NLP goal parsing."""
    print("\nTesting NLP goal parsing...")
    
    try:
        from nlp import GoalParser
        parser = GoalParser()
        goal = parser.parse("reach the green target")
        print(f"  ✓ GoalParser created and parsed goal")
        print(f"    Goal type: {goal.goal_type}")
    except Exception as e:
        print(f"  ✗ GoalParser: {e}")
        traceback.print_exc()
        return False
    
    try:
        from nlp import RewardGenerator
        generator = RewardGenerator()
        print(f"  ✓ RewardGenerator created")
    except Exception as e:
        print(f"  ✗ RewardGenerator: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_replay_buffer():
    """Test replay buffer operations."""
    print("\nTesting replay buffer...")
    
    import numpy as np
    
    try:
        from rl_core import ReplayBuffer
        buffer = ReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(50):
            state = np.random.randn(4).astype(np.float32)
            action = np.array([0])
            reward = float(i)
            next_state = np.random.randn(4).astype(np.float32)
            done = i == 49
            buffer.push(state, action, reward, next_state, done)
        
        print(f"  ✓ ReplayBuffer created and populated with {len(buffer)} experiences")
        
        # Sample a batch
        batch = buffer.sample(16)
        print(f"  ✓ Sampled batch of size 16")
        
    except Exception as e:
        print(f"  ✗ ReplayBuffer: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_training_components():
    """Test training session creation."""
    print("\nTesting training components...")
    
    try:
        from training import TrainingSession
        session = TrainingSession(
            env_config={"env_type": "gridworld", "size": 5},
            agent_config={"algorithm": "dqn"},
            training_config={"total_timesteps": 100}
        )
        print(f"  ✓ TrainingSession created, ID: {session.session_id}")
    except Exception as e:
        print(f"  ✗ TrainingSession: {e}")
        traceback.print_exc()
        return False
    
    try:
        from training import TrainingManager
        manager = TrainingManager()
        print(f"  ✓ TrainingManager created")
    except Exception as e:
        print(f"  ✗ TrainingManager: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  RL-GYM Component Tests                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_environment_creation():
        all_passed = False
    
    if not test_algorithm_creation():
        all_passed = False
    
    if not test_nlp():
        all_passed = False
    
    if not test_replay_buffer():
        all_passed = False
    
    if not test_training_components():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

