# Using MujocoCar Environment

The *MujocoCar* environment is modified from the [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) library. We're using the [Safe Navigation Goal 0](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_navigation/goal.html#goal0) task as the base task and modified it to sparse-reward setting.

## Installation

1. To use the modified *MujocoCar* environment, you need to install the *safety-gymnasium* library:

```shell
pip install safety-gymnasium
```

2. Copy our modified scripts to the corresponding folders if *safety-gymnasium* library:

```shell
cp ./goal_level0.py <safety_gymnasium_root>/tasks/safe_navigation/goal/goal_level0.py
```

Also, copy the following wrapper to the corresponding folder:

```shell
cp ./self_defined.py <safety_gymnasium_root>/wrappers/self_defined.py
```

Then the *MujocoCar* environment is ready to use.