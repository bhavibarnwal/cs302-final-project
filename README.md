# FINAL PROJECT: Evolutionary Optimization of Centipede Locomotion Using Genetic Algorithms

## Overview
This project implements an optimization loop, open-loop control, and reinforcement learning to simulate the evolution of a centipede generated in DiffTaichi. Inspired by naturally occurring evolutionary mechanisms like natural selection and random mutation, the program replicates survival-based selection and spontaneous feature selection through its integration of biological concepts with computational modeling to optimize centipede locomotion. 

## Modification & Implementation Details
The evolution of the centipede is driven by a pre-defined objective function: maximizing the distance traveled, or minimizing the loss function. Centipede evolution is simulated over 20 generations, with each consisting of four organisms with varying body heights and varying numbers of segments. Organisms in the first generation are generated with random body height and segment number values, while organisms in future generations are results of expected genetic and reproductive patterns:

1. *Offspring 1*: Clone of the best-performing centipede in previous generation
2. *Offspring 2*: Mix of best-performing and second-best performing centipedes in previous generation (50% from each parent)
3. *Offspring 3*: Mix of second-best performing and third-best performing centipedes in previous generation (50% from each parent)
4. *Offspring 4*: Centipede generated with randomized parameters (mimicking random mutation)

This algorithm models survival of the fittest by removing the worst-performing organism from every generation and forcing rapid convergence to the parameter values most optimal for the objective function. The mating patterns for Offsprings 2 and 3 are assumed based on the homophily principle, which states that animals tend to form relationships and interact with others who are most similar to them. "Most similar" centipedes were defined as two centipedes of the same performance level or adjacent performance levels. 

## Running the Simulation
1. Ensure Taichi is installed. (pip install taichi)
2. Run diffmpm.py. (python diffmpm.py)

Note: The program only advances to the next generation after the "Optimization Initial Velocity" plot is manually closed. 

## Relevant Links
View a summary of the program and its results [here](https://youtu.be/mdhfaDsPtEY).\
View a quick teaser of the centipede evolution [here](https://drive.google.com/file/d/1Q9l7S0edR8TR-SMY6os_ov7Ty7062Uyl/view?usp=sharing).\
View observed program results [here](https://docs.google.com/document/d/1pacFyVqc_3ryA5eGiVm-541kkJzkAr3z3UMGV0KAAU8/edit?usp=sharing).