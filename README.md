# Pytorch_GAN_MNIST
Pytorch implementation of Basic GAN in generating handwritten digits

## Background
Generative Adversarial Net (GAN) is a deep learning architecture framework that consists of 2 main components - Generator and Discriminator. It is an extremely powerful architecture in many content-generation tasks (e.g. image generation, Low-light image enhancement).
GAN was introduced in a paper (https://arxiv.org/pdf/1406.2661.pdf) by Ian Goodfellow and other researchers including Yoshua Bengio in 2014. Facebook's AI research director Yann LeCun once said adversarial training being "the most interesting idea in the last 10 years in ML"

## Training
In this project, I have carried out 300 epochs of training. To be honest, I did not succeed at the first time when I finished writing the code. I have encountered different problems (e.g. too high learning rate, overfitting of the Discriminator).
At last, the training went well. The losses are shown below:

## Results
Below is the evolution of the result in 300 epochs (sampled every 10 epochs)
