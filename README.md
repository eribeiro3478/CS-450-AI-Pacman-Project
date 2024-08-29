Project Overview:
In our CS 450 Introduction to Artificial Intelligence course, we undertook a collaborative project based on the Pacman game originally developed by the University of California, Berkeley. The project was implemented using the Python programming language and focused on developing and optimizing intelligent agents to navigate the game environment.

Objective:
The primary objective of our project was to design and implement multiple AI agents that could effectively guide Pacman to win the game. The goal was to identify the shortest and fastest path for Pacman to consume all the food pellets on the board while avoiding contact with ghosts. We aimed to achieve this without any external assistance, ensuring that Pacman could autonomously make decisions that would lead to the highest possible score in the least amount of time.

Methodology:
We developed a custom evaluation function that assessed various factors critical to Pacman's success:
- Proximity to the Nearest Food Pellet: This factor measures the distance between Pacman and the closest food pellet. Minimizing this distance was essential for ensuring that Pacman could efficiently collect food.
- Distance from Ghosts and Their Scared State: This component evaluated how far Pacman was from the ghosts and whether the ghosts were in a "scared" state. Pacman would prioritize avoiding ghosts unless they were scared, in which case he could pursue them to gain extra points.
- Number of Food Pellets Remaining: The function considered the total number of food pellets left on the board. The agent was incentivized to clear the board as quickly as possible.
- Current Game Score: The overall game score was a crucial indicator of success. The function adjusted its strategy to maximize the score while balancing the risks of encountering ghosts.

Optimization Strategy:
The weights assigned to each of the above factors were carefully calibrated to balance aggressive food collection with cautious ghost avoidance. By tuning these weights, we aimed to create an agent that could navigate the game environment in a way that maximized the game score while minimizing the time taken and the risk of being caught by ghosts. Our experiments involved rigorous testing and iterative improvements to ensure that the agent's performance was both efficient and effective. The result was an intelligent Pacman agent capable of autonomously completing the game with optimal strategies, thus achieving the highest possible score in the shortest amount of time.
