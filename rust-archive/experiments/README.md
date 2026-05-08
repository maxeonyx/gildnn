# Research Experiments

This directory is for the actual experiments which are the eventual goal of this repository, and will be started after the base experiments are done, to ensure we have a working, flexible ML implementation before starting the more complicated stuff.

Like the base experiments, when writing new experiments, refactor out shared functionality from any other experiments (including base ones) to build the flexible `core` implementation, and run all experiments in `test` mode to test that they all still work.

The most important thing is that we take only one small step at a time, and make sure it's thoroughly working before moving on. We then want to reuse code while ensuring it's still testable on old experiments, to reduce the risk of bugs that leave our experiments with indeterminate results.
