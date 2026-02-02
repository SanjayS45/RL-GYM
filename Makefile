# RL-GYM Makefile
# Convenience commands for development and deployment

.PHONY: help install install-backend install-frontend dev backend frontend test clean build

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                    RL-GYM Commands                            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make install          Install all dependencies"
	@echo "  make install-backend  Install Python backend dependencies"
	@echo "  make install-frontend Install Node.js frontend dependencies"
	@echo ""
	@echo "  make dev              Start both backend and frontend (dev mode)"
	@echo "  make backend          Start backend server only"
	@echo "  make frontend         Start frontend dev server only"
	@echo ""
	@echo "  make test             Run backend tests"
	@echo "  make test-frontend    Run frontend tests"
	@echo ""
	@echo "  make build            Build frontend for production"
	@echo "  make clean            Clean build artifacts"
	@echo ""

# Install all dependencies
install: install-backend install-frontend
	@echo "✓ All dependencies installed"

# Install backend dependencies
install-backend:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "✓ Backend dependencies installed"

# Install frontend dependencies
install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✓ Frontend dependencies installed"

# Start development servers
dev:
	@echo "Starting RL-GYM in development mode..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo ""
	@make -j2 backend frontend

# Start backend server
backend:
	@echo "Starting backend server..."
	cd backend && python run.py --reload

# Start frontend dev server
frontend:
	@echo "Starting frontend dev server..."
	cd frontend && npm run dev

# Run backend tests
test:
	@echo "Running backend tests..."
	cd backend && python test_components.py

# Run frontend tests
test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm test

# Build frontend for production
build:
	@echo "Building frontend for production..."
	cd frontend && npm run build
	@echo "✓ Frontend built to frontend/dist/"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf frontend/dist
	rm -rf frontend/node_modules/.cache
	rm -rf backend/__pycache__
	rm -rf backend/**/__pycache__
	rm -rf backend/**/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Build artifacts cleaned"

# Development shortcuts
.PHONY: b f t
b: backend
f: frontend
t: test

