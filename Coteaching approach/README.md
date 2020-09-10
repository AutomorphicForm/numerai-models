Details of the co-teaching method are presented in their paper: https://arxiv.org/abs/1804.06872
The code in this approach is largely adapted from https://github.com/bhanML/Co-teaching

Most notably, instead of discarding training examples which have not been selected, the loss function weighs them less but still ensures that the neural network is incentivized to account account for them.
