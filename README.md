# Dependencies 
* [enviorment](https://retro.readthedocs.io/en/latest/getting_started.html)
* [models](https://stable-baselines.readthedocs.io/en/master/index.html)
* [mario](https://pypi.org/project/gym-super-mario-bros/)

# Stable Baselines

## Policy Types
https://stable-baselines.readthedocs.io/en/master/modules/policies.html

- MLP (Multi-layer perceptron)
  - MLPPolicy
    - Basic implementation, 2 layers of 64
  - MLPLstmPolicy
    >LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
    - The problem of Mario shouldn't need long term dependencies
  - MLPLnLstmPolicy
    - LSTM but input is normalized
- CNN
  - cnns are for images only


### Customizing Policies
We can customize by setting the parameters of the Policy class  
https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html

Ones we probably care about:

- n_env - (int) The number of environments to run
- n_steps - (int) The number of steps to run for each environment
- n_batch - (int) The number of batch to run (n_envs * n_steps)

## PPO2 Parameters

PPO hyper parameters explained: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

- learning_rate
- noptepochs - number of epochs
  
## Automatic Hyper-parameter Tuning

There is a project that created some pre-trained agents called [rl-zoo](https://github.com/araffin/rl-baselines-zoo).
They use a project called [Optuna](https://github.com/araffin/rl-baselines-zoo) to find the best hyper-parameters for the agents so we might want to use it too.