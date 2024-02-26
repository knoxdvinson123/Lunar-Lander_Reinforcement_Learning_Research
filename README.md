# Lunar-Lander_Reinforcement_Learning_Research

This repository is a hands-on exploration of machine learning techniques, focusing on training a small lunar lander to master the art of controlled descent. The Lunar Lander environment, "LunarLander-v2," presents a thrilling challenge where the agent must utilize vertical and side thrusters to land the spacecraft safely between designated flags.

![image](https://github.com/knoxdvinson123/Lunar-Lander_Reinforcement_Learning_Research/assets/154300416/76f93f52-fe99-4c23-a945-85b9ee2ea765)


About Reinforcement Learning (RL):
- Reinforcement Learning empowers agents to learn optimal behavior by interacting with an environment. In our Lunar Lander scenario, the agent faces the task of precision landing, employing vertical and side thrusters strategically. The learning process involves iterative training sessions, with positive rewards for successful landings and negative feedback for unnecessary thruster usage.

Key Features:
- Implementation using OpenAI Gym for the Lunar Lander environment.
- Integration of stable_baselines_3 library for Reinforcement Learning algorithms, specifically the Advantage Actor-Critic (A2C) algorithm.
- Design of a reward system that encourages the agent to land precisely between designated flags.
  
Algorithm:
- For this Lunar Lander project, we have chosen the Advantage Actor-Critic (A2C) algorithm. A2C combines elements of both policy-based (Actor) and value-based (Critic) methods, making it a powerful and efficient choice for training agents in dynamic environments.

Environment:
- The project centers around the "LunarLander-v2" environment, where the lunar lander must navigate gravitational forces and employ thrusters judiciously to achieve a safe landing between specified flags.

Objectives:
- The agent gains positive rewards for accurate landings between flags, with substantial negative penalties for crashes. Each thruster usage incurs a small negative feedback, adding a strategic dimension to the learning process.
