# Barycentric Rational Net

We consider neural networks with **barycentric rational activation functions**. The barycentric form of rational functions has demonstrated superior numerical approximation capabilities across a wide range of tasks. A representative work showcasing the power of this form is the AAA algorithm [@nakatsukasa2018aaa]. In this work, we leverage the exceptional numerical quality of the form

$$
R(x) \;=\;
\frac{\displaystyle\sum_{j=0}^n \frac{w_j\,y_j}{\,x - x_j\,}}
     {\displaystyle\sum_{j=0}^n \frac{w_j}{\,x - x_j\,}}
$$

as the nonlinear activation function in our neural network.

rational.py and experiment_1d.py are for rational NN;
rational_baryrat.py and experiment_baryrat.py are for baryrat NN
