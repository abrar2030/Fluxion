#!/bin/bash\nset -euo pipefail
# Run script for Fluxion project
# This script starts the application components

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Fluxion application...${NC}"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo -e "${BLUE}Creating Python virtual environment...${NC}"
  python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r code/requirements.txt > /dev/null

# Start backend application
echo -e "${BLUE}Starting backend service...${NC}"
cd code/backend
uvicorn app:app --host 0.0.0.0 --port 5000 &
APP_PID=$!
cd ../..

# Start frontend application
echo -e "${BLUE}Starting frontend service...${NC}"
cd frontend
npm install > /dev/null
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}Fluxion application is running!${NC}"
echo -e "${GREEN}Backend running with PID: ${APP_PID}${NC}"
echo -e "${GREEN}Frontend running with PID: ${FRONTEND_PID}${NC}"
echo -e "${GREEN}Backend API: http://localhost:5000${NC}"
echo -e "${GREEN}Frontend UI: http://localhost:5173${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"

# Handle graceful shutdown
function cleanup {
  echo -e "${BLUE}Stopping services...${NC}"
  kill $APP_PID
  kill $FRONTEND_PID
  echo -e "${GREEN}All services stopped${NC}"
  exit 0
}

trap cleanup SIGINT

# Keep script running
wait
