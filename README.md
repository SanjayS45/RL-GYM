# RL-GYM 🏋️‍♂️

**Interactive Reinforcement Learning Training Platform**

*Inspired by AI Warehouse YouTube Channel*

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-brightgreen.svg" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/React-18+-61dafb.svg" alt="React">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 🎯 Overview

RL-GYM is a complete, interactive platform for training reinforcement learning agents. Watch agents learn in real-time, modify hyperparameters on the fly, and define custom environments with natural language goals.

### Key Features

- 🎮 **Real-time Visualization**: Watch your agents learn and improve
- 🧠 **Multiple RL Algorithms**: DQN, PPO, SAC, A2C out of the box
- 🌍 **Custom Environments**: GridWorld, Navigation, Platformer
- 💬 **Natural Language Goals**: Define objectives in plain English
- 📊 **Live Metrics**: Track training progress with beautiful charts
- 🎛️ **Interactive Controls**: Modify parameters during training
- 📁 **Dataset Support**: Import demonstrations and offline data

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Frontend                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐│
│  │ Environ │  │  Agent  │  │Training │  │ Visual  │  │Metrics ││
│  │  Setup  │  │ Params  │  │Controls │  │  izer   │  │  Panel ││
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬───┘│
│       └────────────┴────────────┼────────────┴────────────┘    │
│                                 │                               │
│                        WebSocket + REST API                     │
└─────────────────────────────────┼───────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────┐
│                           Backend                               │
│  ┌─────────────┐  ┌─────────────┴─────────────┐  ┌────────────┐│
│  │   RL Core   │  │      Training Manager     │  │  Datasets  ││
│  │  ┌───────┐  │  │  ┌────────┐  ┌─────────┐  │  │  ┌──────┐  ││
│  │  │  DQN  │  │  │  │Session │  │Callbacks│  │  │  │Loader│  ││
│  │  │  PPO  │  │  │  │ Manager│  │         │  │  │  │Valid.│  ││
│  │  │  SAC  │  │  │  └────────┘  └─────────┘  │  │  └──────┘  ││
│  │  │  A2C  │  │  └───────────────────────────┘  │            ││
│  │  └───────┘  │                                 │            ││
│  └─────────────┘  ┌───────────────────────────┐  └────────────┘│
│                   │      Environments          │               │
│  ┌─────────────┐  │  ┌─────────┐  ┌─────────┐ │               │
│  │     NLP     │  │  │GridWorld│  │ Navig.  │ │               │
│  │ Goal Parser │  │  │Platform │  │ Physics │ │               │
│  └─────────────┘  │  └─────────┘  └─────────┘ │               │
│                   └───────────────────────────┘               │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/SanjayS45/RL-GYM.git
cd RL-GYM

# Install all dependencies
make install
```

### Running the Application

```bash
# Start both backend and frontend (recommended)
make dev

# Or start them separately:
make backend  # http://localhost:8000
make frontend # http://localhost:5173
```

### Using Docker

```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## 📦 Project Structure

```
RL-GYM/
├── backend/                 # Python backend
│   ├── rl_core/            # RL algorithms and utilities
│   │   ├── algorithms/     # DQN, PPO, SAC, A2C
│   │   ├── base.py         # Base policy class
│   │   ├── networks.py     # Neural network architectures
│   │   ├── buffers.py      # Replay buffer
│   │   └── utils.py        # Utilities
│   ├── environments/       # Custom Gym environments
│   │   ├── grid_world.py   # Discrete navigation
│   │   ├── navigation.py   # Continuous navigation
│   │   └── platformer.py   # 2D platformer
│   ├── nlp/                # Natural language processing
│   │   ├── goal_parser.py  # Parse NL goals
│   │   └── reward_generator.py
│   ├── datasets/           # Dataset management
│   ├── training/           # Training orchestration
│   ├── api/                # FastAPI server
│   └── config.py           # Configuration
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── store/          # Zustand state
│   │   └── App.tsx         # Main app
│   └── package.json
├── examples/               # Example configurations
├── docker-compose.yml      # Docker setup
├── Makefile               # Development commands
└── README.md
```

---

## 🎮 Usage

### 1. Environment Setup

Choose from predefined environments:
- **GridWorld**: Discrete navigation on a grid
- **Navigation**: Continuous 2D navigation with lidar
- **Platformer**: 2D platformer with jumping

Or customize with obstacles and goals!

### 2. Agent Configuration

Select an algorithm and tune hyperparameters:

| Algorithm | Best For | Key Parameters |
|-----------|----------|----------------|
| **DQN** | Discrete actions | ε-greedy, replay buffer |
| **PPO** | Both action types | clip range, GAE λ |
| **SAC** | Continuous actions | entropy α, soft updates |
| **A2C** | Fast training | n-steps, value coefficient |

### 3. Natural Language Goals

Define goals in plain English:

```
"Reach the green target while avoiding red obstacles"
"Navigate to the goal using the shortest path"
"Collect all coins without falling off platforms"
```

### 4. Training

Start training and watch your agent learn:
- Real-time visualization
- Live metrics (reward, loss, episode length)
- Pause/resume/stop controls
- Speed adjustment

---

## 🔧 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/training/start` | POST | Start training session |
| `/training/stop/{id}` | POST | Stop training |
| `/training/status/{id}` | GET | Get session status |
| `/environments/list` | GET | List available environments |
| `/agents/algorithms` | GET | List available algorithms |
| `/datasets/upload` | POST | Upload dataset |

### WebSocket

Connect to `/training/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/training/ws');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // Handle training update
};
```

---

## 📊 Example Configurations

See the `examples/` directory for ready-to-use configurations:

- `gridworld_dqn.json` - DQN on GridWorld
- `navigation_ppo.json` - PPO on Navigation
- `platformer_sac.json` - SAC on Platformer
- `natural_language_goal.json` - Using NL goals

---

## 🧪 Testing

```bash
# Run backend tests
make test

# Run specific component tests
cd backend && python test_components.py
```

---

## 🛠️ Development

### Commands

```bash
make install         # Install dependencies
make dev             # Start development servers
make backend         # Start backend only
make frontend        # Start frontend only
make test            # Run tests
make build           # Build for production
make clean           # Clean build artifacts
```

### Adding a New Algorithm

1. Create a new file in `backend/rl_core/algorithms/`
2. Extend `BasePolicy` class
3. Implement `act()`, `learn()`, `save()`, `load()`
4. Export from `__init__.py`
5. Add to API routes

### Adding a New Environment

1. Create a new file in `backend/environments/`
2. Extend `gymnasium.Env`
3. Implement `reset()`, `step()`, `render()`
4. Export from `__init__.py`
5. Add to API routes

---

## 📈 Roadmap

- [ ] Multi-agent support
- [ ] Curriculum learning
- [ ] Model-based RL algorithms
- [ ] Custom reward shaping UI
- [ ] Training comparison tools
- [ ] Cloud deployment options

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by [AI Warehouse](https://www.youtube.com/@AIWarehouse) YouTube channel
- Built with [PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/)
- Environment design influenced by [Gymnasium](https://gymnasium.farama.org/)

---

<p align="center">
  Made with ❤️ for the RL community
</p>
