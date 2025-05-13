#!/usr/bin/env bash

while getopts n:r:d:f:s:p: flag
do
    case "${flag}" in
        n) networkScenarios=${OPTARG};;
        r) simulationRuns=${OPTARG};;
        d) experimentsDir=${OPTARG};;
        f) fractionFit=${OPTARG};;
        s) startScenario=${OPTARG};;
        p) startRun=${OPTARG};;
    esac
done

# Run the simulations for the federated learning baseline
for (( i=startScenario; c<networkScenarios; i++ ))
do
    for (( j=startRun; j<simulationRuns; j++ ))
    do
        flwr run . --run-config "network-scenario=${i} simulation-run=${j} experiments-dir='${experimentsDir}' fraction-fit=${fractionFit}"
    done
    startRun=0
done