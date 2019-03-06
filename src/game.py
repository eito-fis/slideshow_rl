import numpy as np

def init(slides):
    '''
       0  1  2
      ________
    0|-1  1  1
    1| 1 -1  0
    2| 1  1 -1

    '''
    n_slides = len(slides)
    rewards = np.zeros((n_slides,n_slides))

    for i in range(n_slides):
        rewards[i,i] = -1. # can't go on intself
        for j in range(i+1,n_slides):
            same_tags = slides[i] & slides[j]
            tags_only_i = slides[i] - slides[j]
            tags_only_j = slides[j] - slides[i]
            rewards[i,j] = min(len(same_tags), len(tags_only_i), len(tags_only_j))
            rewards[j,i] = rewards[i,j]

    current_vector = np.zeros(n_slides)
    current_vector[0] = 1.

    considering_vector = np.zeros(n_slides)
    considering_vector[np.random.randint(1,n_slides)] = 1.

    state = (rewards, current_vector, considering_vector)
    return state

def step(state, go):
    rewards, current_vector, considering_vector = state

    current_i = np.argmax(current_vector)
    considering_i = np.argmax(considering_vector)

    if go:
        added_slide_i = considering_i

        reward = rewards[current_i,considering_i]
        # cannot go to the current slide anymore
        rewards[current_i] = -1.
        rewards[:,current_i] = -1.

        # change current position
        current_vector[current_i] = 0.
        current_vector[considering_i] = 1.

        current_i = considering_i
    else:
        reward = 0.
        added_slide_i = None

    done = np.max(rewards[current_i]) == -1
    if not done:
        # remove considering position
        considering_vector[considering_i] = 0.
        # select next considering position
        possible_slides = [slide_i for slide_i in range(len(current_vector)) if rewards[current_i,slide_i] != -1]
        considering_i = np.random.choice(possible_slides)
        considering_vector[considering_i] = 1.

    return state, reward, done

def init_index(slides):
    '''
       0  1  2
      ________
    0|-1  1  1
    1| 1 -1  0
    2| 1  1 -1

    '''
    n_slides = len(slides)
    rewards = np.zeros((n_slides,n_slides))

    for i in range(n_slides):
        rewards[i,i] = -1. # can't go on intself
        for j in range(i+1,n_slides):
            same_tags = slides[i] & slides[j]
            tags_only_i = slides[i] - slides[j]
            tags_only_j = slides[j] - slides[i]
            rewards[i,j] = min(len(same_tags), len(tags_only_i), len(tags_only_j))
            rewards[j,i] = rewards[i,j]

    current_vector = np.zeros(n_slides)
    current_vector[0] = 1.

    block_vector = np.zeros(n_slides)
    block_vector[0] = 1

    state = (rewards, current_vector, block_vector)
    return state


def step_index(state, action):
    rewards, current_vector, block_vector = state

    current_i = np.argmax(current_vector)

    # Catch for moving to invalid slice
    if (block_vector[action] == 1 or rewards[current_i][action] == -1):
        block_vector[action] = 1
        return state, 0, False
    reward = rewards[current_i, action]
    # cannot go to the current slide anymore
    rewards[current_i] = -1.
    rewards[:,current_i] = -1.

    block_vector[:] = 0
    current_vector[:] = 0
    current_vector[action] = 1
    #print("BLOCK_VECTOR: {}".format(block_vector))
    #print("REWARDS: {}".format(rewards))
    #print("REWARDS[ACTION]: {}".format(rewards[action]))
    #print("ACTION: {}".format(action))
    #print("REPLACING...: {}".format(np.where(rewards[action] == -1)))
    for index in np.where(rewards[action] == -1):
        block_vector[index] = 1
    #print("BLOCK_VECTOR AFTER: {}".format(block_vector))
    done = True if sum(block_vector) == len(block_vector) else False

    state = (rewards, current_vector, block_vector)
    return state, reward, done

def render(state, reward=None, done=None):
    render_format = \
        'State:\n' + \
        '- rewards:\n' + \
        '{}\n' + \
        '\n' + \
        '- current position:\n' + \
        '{}\n' + \
        '\n' + \
        '- considering position:\n' + \
        '{}\n'

    print(render_format.format(*state))
    if reward is not None: print('Reward: {}'.format(reward))
    if done is not None: print('Done: {}'.format(done))

    print('-----------------------')


def preprocess(_state):
    # Get data
    matrix = _state[0]
    vector_1 = _state[1]
    vector_2 = _state[2]

    # Reshape data
    matrix = matrix.reshape(1, len(matrix), len(matrix[0]), 1)
    vectors = np.concatenate((vector_1, vector_2), axis=None)
    vectors = vectors.reshape(1, len(vectors))
    
    return matrix, vectors

def play(model, gen, sample_size):
    while True:
        print("==========STARTING ROLLOUT==========")
        sample_photos = next(gen)

        # init game
        _state = init_index(sample_photos)
        _matrix_state, _vector_state = preprocess(_state)
        _done = False
        total_reward = 0
        count = 0

        while not _done and count < sample_size * 2:
            _predict = model.predict([_matrix_state, _vector_state], batch_size=1)[0]
            _action = np.argmax(_predict)
            print("==========")
            print("Action being taken: {}".format(_action))
            print("State:")
            print(_state[0])
            print(_state[1])
            print(_state[2])
            print("==========")
            _state, _reward, _done  = step_index(_state, _action)
            _matrix_state, _vector_state = preprocess(_state)
            total_reward += _reward
            count += 1
        print("Total reward: {}".format(total_reward))
        input()
        


if __name__ == '__main__':

    slides = [{0,1,2},{3,4,5},{5,0}]
    state = init(slides)
    render(state)

    # starts from first slide because assumes that the slides are shuffled
    slideshow = [0]
    total_reward = 0

    done = False
    while not done:
        go = np.random.choice([True,False])
        state,reward,done,added_slide_i = step(state, go)

        total_reward += reward
        if added_slide_i is not None: slideshow.append(added_slide_i)

        render(state,reward=reward,done=done)

    print('Slideshow: {}'.format(slideshow))
    print('Reward: {}'.format(total_reward))
