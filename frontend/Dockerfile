# RL-GYM Frontend Dockerfile
FROM node:18-alpine

# Set work directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy project files
COPY . .

# Expose port
EXPOSE 5173

# Run the development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

