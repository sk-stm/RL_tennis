# Start of project

Try out [IQL](https://arxiv.org/pdf/1908.03963.pdf) (independedn Q-learning) since it's naive,
straight forward and works for small problems.

However, in the case of function approximation, especiallydeep neural network (DNN) it fails.
One of the main reasons for this failure is the need for the replay memory to stabilize the training
with DNNs, which does not work because the environment is non stationary for an agent because the other
agent is an undeterministic part of the environment.


# EXP 1
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

NOISE_REDUCTION_FACTOR = 0.99999
THETA = 0.15
MU = 0
SIGMA = 0.2

-> with lucky initialization it learns until 0.1 reward sometimes... very instable, also agents get stuck in
the corners pretty often.
-> try less noise but longer noise application

# Exp 2
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

NOISE_REDUCTION_FACTOR = 0.99999
THETA = 0.7
MU = 0
SIGMA = 0.1

-> doesn't learn much .. very slow progress -> more noise but at a lower initial weight

# Exp 3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

INITIAL_NOISE_WEIGHT = 0.8
NOISE_REDUCTION_FACTOR = 0.99999
THETA = 0.15
MU = 0
SIGMA = 0.2

-> some learning especially in the beginning -> stale after 20000 episodes
-> more consistent noise like in the beginning

fixed a bug that prevented the 2nd critic to be copied to it's target.
fixed a bug that didn't put the 2nd actor in eval mode before acting in the env.

# Exp 3

GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

INITIAL_NOISE_WEIGHT = 0.4
NOISE_REDUCTION_FACTOR = 0.999999
THETA = 0.3
MU = 0
SIGMA = 0.3

-> looks similar to the baseline (https://classroom.udacity.com/nanodegrees/nd893/parts/ec710e48-f1c5-4f1c-82de-39955d168eaa/modules/29462d31-10e3-4834-8273-45df5588bf7d/lessons/3cf5c0c4-e837-4fe6-8071-489dcdb3ab3e/concepts/9a05b852-7c76-48ee-acb7-807c0ebe57b9)
but with not such a big peak in the middle. -> Less noise or faster reduction

# Exp 4
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

INITIAL_NOISE_WEIGHT = 0.4
NOISE_REDUCTION_FACTOR = 0.9999
THETA = 0.15
MU = 0
SIGMA = 0.2

-> good in the beginning but diverged after -> more noise and more rapid decrease

# Exp 5
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

INITIAL_NOISE_WEIGHT = 1
NOISE_REDUCTION_FACTOR = 0.9999
THETA = 0.15
MU = 0
SIGMA = 0.2

-> goodish performance after 400 iterations ~ 0.5 noise weight then diverged
Didn't really get off the ground. average of -0.005 reward which is kind of ok I guess

# Exp 6
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

INITIAL_NOISE_WEIGHT = 0.5
NOISE_REDUCTION_FACTOR = 0.99999
THETA = 0.15
MU = 0
SIGMA = 0.2


-> changed the implementation so that the experiece replay is collected by each agent independently
Increased noise in the beginning and reduced it a bit faster
Reduced the memory buffer to 1000 because [here](https://arxiv.org/pdf/1908.03963.pdf) the they say that
"asically, the policy that generates the data for the replay memoryis different than the current policy
so that the learned policy of each agent can be misleading. Inorder to address this issue,
[38](https://arxiv.org/pdf/1605.06676v2.pdf)
disables the replay memorypart of the algorithm, or in [78](https://arxiv.org/pdf/1702.03037.pdf)
the old transitions are discarded and the replay memory uses only the recent experiences"

# Exp 7
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e3)
TAU = 1e-1
LR_ACTOR = 5e-3
LR_CRITIC = 5e-3
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

# NOISE PARAMETERS
INITIAL_NOISE_WEIGHT = 0.3
NOISE_REDUCTION_FACTOR = 0.9999
THETA = 0.0
MU = 0
SIGMA = 0.8

SAVE_DISTANCE = 0.005

-> trains and holds 0.005 rewards

# Exp 8

GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = int(1e6)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0.00001
UPDATE_EVERY = 5

# NOISE PARAMETERS
INITIAL_NOISE_WEIGHT = 0.2
NOISE_REDUCTION_FACTOR = 0.9999
THETA = 0.15
MU = 0
SIGMA = 0.1

SAVE_DISTANCE = 0.005

-> converges to ~0.02 reward nice!

# Exp 9

GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = int(1e6)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0.00001
UPDATE_EVERY = 5

# NOISE PARAMETERS
INITIAL_NOISE_WEIGHT = 0.2
NOISE_REDUCTION_FACTOR = 0.99999
THETA = 0.15
MU = 0
SIGMA = 0.1

SAVE_DISTANCE = 0.005

-> same same

Found the problem of the Ornstein Uhlenbeck process being calculated wrongly and therefore exploded
Wo wonder it didn't work out..

After that it worked with simply 2 ddpg agents.

