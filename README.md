# RL-GYM 🎮🤖

An interactive reinforcement learning training platform inspired by AI Warehouse's visual style. Train RL agents with real algorithms, watch them learn in real-time, and experiment with hyperparameters.

## Features

- **Real RL Algorithms**: PPO, DQN, SAC, A2C - no fake learning behavior
- **Live Visualization**: Watch agents learn in real-time with synchronized animations
- **Interactive Training**: Modify hyperparameters before and during training
- **Natural Language Goals**: Define objectives using plain English
- **Custom Environments**: Create environments with obstacles, walls, and goals
- **Dataset Support**: Upload demonstrations for pretraining or offline RL

## Architecture

```
RL-GYM/
├── backend/           # Python RL training server
│   ├── rl_core/       # RL algorithms (PPO, DQN, SAC, A2C)
│   ├── environments/  # Gym-style environments
│   ├── agents/        # Agent implementations
│   └── api/           # FastAPI server
├── frontend/          # React visualization UI
│   ├── components/    # UI components
│   ├── canvas/        # Training visualization
│   └── panels/        # Control panels
└── datasets/          # Sample datasets
```

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## License

MIT License

