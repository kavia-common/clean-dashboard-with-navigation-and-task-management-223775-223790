#!/bin/bash
cd /home/kavia/workspace/code-generation/clean-dashboard-with-navigation-and-task-management-223775-223790/task_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

