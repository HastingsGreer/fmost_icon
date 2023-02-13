import random
import torch

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks

input_shape = [1, 1, 192, 192, 192]

BATCH_SIZE=4
GPUS = 4
ITERATIONS_PER_STEP = 50000
#ITERATIONS_PER_STEP = 60

def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load(
        "/playpen-ssd/tgreer/fmost-preprocessed/fmost-9/fmost_192_train.trch"
    )
    batch_function = lambda : (make_batch(dataset), make_batch(dataset))
    cvpr_network.train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE)
