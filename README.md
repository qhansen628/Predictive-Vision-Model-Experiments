# Predictive-Vision-Model-Experiments
a collection of colab notebooks based on the idea of making a vision model work sequentially over saccades on an image. One of the goals are to test whether a process like pure predictive coding
can actually explain cognition, since most neuroscientists I see publically explaining consciousness and perception today use a predictive coding framework however I never see production ML
models that perform any tasks actually use hierarchical predictive coding: Predictive coding is where the at each time step the input layer is compared with a prediction layer, and only the error signal is passed up to the next layer. But the original motivation for these experiments is to test Freud's connectionist model outlined in chapter 7 of the interpretation of dreams, where he explains that the reduction 
of free excitation is the objective of the system... if you read this chapter you will get an impression of something like a predictive coding system, not exactly but its easy to see how the secondary process is a predictive inhibitory mechanism that inevitably comes from the original primary process of preventing a buildup of free excitation... Karl Friston formalized this as the free energy principle, however his formulation is very abstract and doesnt really provide a formulation that isnt directly bayesian or equivelant to simple RL functions. So instead the final goal of this project is to create a minimal recurrent neural network where the only objective is to minimize the networks expected future excitation. 

A final motivation for this project is the phenomenology of my visual experience and the experience of others. If you have ever misread a sentence by inserting a word you might have noticed this, but usually people report seeing the inserted word. It seems like when we look at something the perception is hallucinated, and if the error is high you will look at the place again until your mind perceives the correct object. This makes total sense if you think about human feats of hand eye coordination for fast moving objects, and clearly we know the brain is perfectly capable of producing visual experience in absense of stimuli every night when we dream. Finally 90% of the thalamocortical connections between the LGN and V1 are feed back connections... Unless those neurons somehow autodifferentiate for some biological backpropogarion, it is highly likely that current state of the art feed forward CNNs and ViTs have little in common with human vision this might increase misalighnment and adverserial attack vulnerabilities in production.

The network here is a simple RNN which is constructed more like a reserviour, so the input neurons, hidden neurons, and output neurons are held in a single vector and a single masked weight matrix is
used to update the neuron states. Through regularization to encourage sparsity the hidden neurons can create arbitrary recurrent graphs, but I am not hardcoding hierarchical architecture. Its not intended to
beat state of the art methods in this experiment I am simply testing the plausibility of these ideas.


# RL vision todo:
- Goal:
  - agent must return an accurate classification response on an image
- Environment: image
- Action space:
  - saccade positions: which determine which patch of image will be input to the model next
  - end episode and classify, ("speak" or "announce" "apprehend" "name") action: if this action is selected by the policy, the episode ends and R_T is given based on classifier loss (categorical crossentropy) 
- Preception Prediction Network:
  - Pred(saccade position,context)
  - recieves the saccade chosen by the policy and rnn context vector
    - it is a function which uses the context and next image position which predicts real pixel values
  - returns expected pixel values of patch recieved after taking saccade
  - The goal is for the model to make hypotheses about the image contents based on minimal information, and to learn an action 
- Classifier Network
 - Class(context)
 - returns class probability distribution
- Reward structure:
  - non terminal
   - at reward at non terminal state is based on error of the Prediction Network
   - the more unexpected visual input in response to an action more negative the reward.
     - ... what if pred net predicted reward too? if reward is calculated based on correctness of reward... does this become too meta? infinit recursion? but I believe reward prediction error is standard in RL 
   - maximizing total reward here is minimizing cumulative surprisal.
   - rewards are all negative but if prediction is perfect it is 0
  - terminal:
    - here a positive reward is recieved based on binary crossentropy of class networks...

Complication: mixing supervised learning with RL is required to train prediction network and classification network, unless the prediction/classification outputs were formulated as actions...policy parameters optimized for reward, preception net optimized for poss (KL on representation or MSE/something similar on pxel), classifier on categorical crossentropy??



