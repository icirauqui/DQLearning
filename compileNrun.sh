#!/bin/bash

echo "Compile..."
/usr/bin/g++ -Werror -g /home/icirauqui/w0rkspace/DQLearning/main.cpp -o /home/icirauqui/w0rkspace/DQLearning/main

echo ""
echo "Run..."

if [[ $? == 0 ]]; then
	./main
fi
