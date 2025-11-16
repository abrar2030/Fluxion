#!/bin/bash\nset -euo pipefail

# Test script for Fluxion application
# This script runs tests for both backend and frontend

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Fluxion test suite...${NC}"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo -e "${BLUE}Creating Python virtual environment...${NC}"
  python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r code/requirements.txt > /dev/null

# Test backend
echo -e "${BLUE}Testing backend...${NC}"
cd code/backend
echo -e "${BLUE}Running API health check...${NC}"
curl -s http://localhost:5000/health > /dev/null
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Backend API is running and healthy${NC}"
else
  echo -e "${RED}Backend API is not running or not responding${NC}"
  echo -e "${BLUE}Starting backend for testing...${NC}"
  uvicorn app:app --host 0.0.0.0 --port 5000 &
  BACKEND_PID=$!
  sleep 3
  curl -s http://localhost:5000/health > /dev/null
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Backend API started successfully${NC}"
  else
    echo -e "${RED}Failed to start backend API${NC}"
    exit 1
  fi
fi

# Test frontend build
cd ../../frontend
echo -e "${BLUE}Testing frontend build...${NC}"
npm install > /dev/null
npm run build

if [ $? -eq 0 ]; then
  echo -e "${GREEN}Frontend build successful${NC}"
else
  echo -e "${RED}Frontend build failed${NC}"
  exit 1
fi

# Test integration
echo -e "${BLUE}Testing API integration...${NC}"
curl -s http://localhost:5000/ | grep "Welcome to Fluxion API" > /dev/null
if [ $? -eq 0 ]; then
  echo -e "${GREEN}API integration test passed${NC}"
else
  echo -e "${RED}API integration test failed${NC}"
fi

# Clean up
if [ ! -z "$BACKEND_PID" ]; then
  echo -e "${BLUE}Stopping test backend...${NC}"
  kill $BACKEND_PID
fi

echo -e "${GREEN}All tests completed!${NC}"
