# Predictive-Vision-Model-Experiments
a collection of colab notebooks based on the idea of making a vision model 
work sequentially over saccades on an image


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



