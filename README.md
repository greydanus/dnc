DNC: Differentiable Neural Computer
=======

Implements DeepMind's third nature paper, [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html) by Graves et. al.

![DNC schema](copy/static/dnc_schema.png?raw=true)

Based on the paper's appendix, I sketched the [computational graph](https://docs.google.com/drawings/d/1Fc9eOH1wPw0PbBHWkEH39jik7h7HT9BWAE8ZhSr4hJc/edit?usp=sharing)

_This is a work in progress_
--------
I have a general framework and a couple Jupyter notebooks for debugging. This is not a finished project. It's still very much in the dev stage. I need to:
1. write unit tests
2. improve documentation/comments
3. run it on more difficult tasks
4. add some nice visualizations


Dependencies
--------
* All code is written in python 2.7. You will need:
 * Numpy
 * Matplotlib
 * [TensorFlow r1.0](https://www.tensorflow.org/api_docs/python/)