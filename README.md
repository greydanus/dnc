DNC: Differentiable Neural Computer
=======

See [blog post](https://greydanus.github.io/2017/02/27/differentiable-memory-and-the-brain/)

Implements DeepMind's third nature paper, [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html) by Graves et. al.

![Repeat copy results](static/repeat_copy_results.png?raw=true)

Based on the paper's appendix, I sketched the [computational graph](https://docs.google.com/drawings/d/1Fc9eOH1wPw0PbBHWkEH39jik7h7HT9BWAE8ZhSr4hJc/edit?usp=sharing)

I got the repeat-copy task to work ([Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/dnc/blob/master/repeat-copy/repeat-copy-nn.ipynb))

Dependencies
--------
* All code is written in python 2.7. You will need:
 * Numpy
 * Matplotlib
 * [TensorFlow r1.0](https://www.tensorflow.org/api_docs/python/)
