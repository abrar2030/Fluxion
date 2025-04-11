#!/bin/bash

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

# Start application
echo -e "${BLUE}Starting Fluxion application...${NC}"
cd code
source ../venv/bin/activate
pip install -r requirements.txt > /dev/null
python app.py &
APP_PID=$!
cd ..

echo -e "${GREEN}Fluxion application is running!${NC}"
echo -e "${GREEN}Application running with PID: ${APP_PID}${NC}"
echo -e "${GREEN}Access the application at: http://localhost:5000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"

# Handle graceful shutdown
function cleanup {
  echo -e "${BLUE}Stopping services...${NC}"
  kill $APP_PID
  echo -e "${GREEN}All services stopped${NC}"
  exit 0
}

trap cleanup SIGINT

# Keep script running
wait
