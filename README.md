# FINAL PROJECT: Evolutionary Optimization of Centipede Locomotion Using Genetic Algorithms

## Overview
This project implements an optimization loop, open-loop control, and reinforcement learning to simulate the evolution of a centipede generated in DiffTaichi. Inspired by naturally occurring evolutionary mechanisms like natural selection and random mutation, the program replicates survival-based selection and spontaneous feature selection through its integration of biological concepts with computational modeling to optimize centipede locomotion. 

## Modification & Implementation Details
The evolution of the centipede is driven by a pre-defined objective function: maximizing the distance traveled, or minimizing the loss function. The centipedes learn how to move forward using their body parameters over 100 iterations of reinforcement learning with this objective, and progress is tracked by visualizing their motion every ten iterations. Then, the parameters of centipedes that can learn how to move the farthest are used in the next generation of simulated centipedes. This is repeated for a total of five generations, with each consisting of four organisms with varying body heights and varying numbers of segments. To carry over parameters, an algorithm is adopted to produce organisms based on results of expected genetic and reproductive patterns: 

1. *Offspring 1*: Clone of the best-performing centipede in previous generation
2. *Offspring 2*: Mix of best-performing and second-best performing centipedes in previous generation (50% from each parent)
3. *Offspring 3*: Mix of second-best performing and third-best performing centipedes in previous generation (50% from each parent)
4. *Offspring 4*: Centipede generated with randomized parameters (mimicking random mutation)

This algorithm is used for all generations after the first, which consists of four randomly generated centipedes to serve as the original population. Body height is randomized from _ to _, and number of segments is randomized from 1 to 9 segments.

 The program models survival of the fittest by removing the worst-performing organism from every generation and forcing rapid convergence to the parameter values most optimal for the objective function. The mating patterns for Offsprings 2 and 3 are assumed based on the homophily principle, which states that animals tend to form relationships and interact with others who are most similar to them. "Most similar" centipedes were defined as two centipedes of the same performance level or adjacent performance levels.

## Limitations & Improvements
Due to DiffTaichi limitations on particle allocation, one of the organisms in the first generation was initialized to the maximum parameter values. This prevents the first generation from being completely random but still works for the purposes of this simulation because the remaining organisms are still randomly initialized, allowing natural selection to act on a diverse range of traits. 

Furthermore, the centipede does not travel as far as expected in the iterations allotted due to an even weight distribution across its skeleton, no matter the changes in parameters. This results in high stability but limits forward propulsion since there is no natural tendency for the centipede to tip or shift weight dynamically. The uniform distribution prevents the buildup of momentum in a singular direction, making movement more constrained. To mitigate this and observe more motion, future testing can incorporate a "head" on the centipede to introduce an uneven weight distribution, creating a natural forward bias that encourages movement. 

Due to time and computational constraints, only five generations could be simulated, but future iterations of this program should test evolution
over more generations to confirm convergence of the chosen parameters. Other improvements include: exploring a wider range of initial heights and segment numbers to determine if different optima emerge, incorporating environmental variations such as uneven terrain to test robustness of learning, fine-tuning learning rates and mutation parameters to enhance optimization, and incorporating other objective functions (i.e. energy efficiency, stability) to force a more successful evolution.

## Running the Simulation
1. Ensure Taichi is installed. (pip install taichi)
2. Run diffmpm.py. (python diffmpm.py)

Note: The program only advances to the next generation after the "Optimization Initial Velocity" plot is manually closed. 

## Relevant Links
View a summary of the program and its results [here](https://youtu.be/vieBQ8nmPiQ).\
View a quick teaser of the centipede evolution [here](https://drive.google.com/file/d/101_RovBEks5KobEzujfXNuMl3WJi9ZkE/view?usp=sharing).\
View observed program results [here](https://docs.google.com/document/d/1Vui7OamOJWye0zQlU2DKvVs7a8VYWwqOwMWXWA6XXNs/edit?usp=sharing).