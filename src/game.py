import numpy as np

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

    current_matrix = np.zeros((n_slides, n_slides))
    current_matrix[0][0] = 1.

    block_matrix = np.zeros((n_slides, n_slides))
    block_matrix[0][0] = 1

    state = (rewards, current_matrix, block_matrix)
    return state

def step_index(state, action):
    rewards, current_matrix, block_matrix = state

    current_i = np.argmax(current_matrix) % len(current_matrix)

    # Catch for moving to invalid slice
    if (block_matrix[current_i][action] == 1):
        state = (rewards, current_matrix, block_matrix)
        return state, -0.1, False
    reward = rewards[current_i, action]
    # cannot go to the current slide anymore
    rewards[current_i] = -1.
    rewards[:,current_i] = -1.
    block_matrix[current_i] = 1
    block_matrix[:, current_i] = 1
    block_matrix[action][action] = 1

    current_matrix[:] = 0
    current_matrix[action][action] = 1
    #print("BLOCK_VECTOR: {}".format(block_matrix))
    #print("REWARDS: {}".format(rewards))
    #print("REWARDS[ACTION]: {}".format(rewards[action]))
    #print("ACTION: {}".format(action))
    #print("REPLACING...: {}".format(np.where(rewards[action] == -1)))
    #print("BLOCK_VECTOR AFTER: {}".format(block_matrix))
    done = False if any(0 in sublist for sublist in block_matrix) else True

    state = (rewards, current_matrix, block_matrix)
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
    adj_matrix = _state[0]
    current_matrix = _state[1]
    block_matrix  = _state[2]

    # Reshape data
    matrix = np.stack((adj_matrix, current_matrix, block_matrix), axis=-1)
    matrix = np.reshape(matrix, (1, len(adj_matrix), len(adj_matrix), 3))
    
    return matrix

def play(model, gen, sample_size):
    while True:
        print("==========STARTING ROLLOUT==========")
        sample_photos = next(gen)

        # init game
        _state = init_index(sample_photos)
        _matrix_state  = preprocess(_state)
        _done = False
        total_reward = 0
        count = 0

        while not _done and count < sample_size * 2:
            _predict = model.predict([_matrix_state], batch_size=1)[0]
            print("Probabilities: {}".format(_predict))
            _action = np.argmax(_predict)
            print("==========")
            print("Action being taken: {}".format(_action))
            print("State:")
            [print(l) for l in _state[0].tolist()]
            [print(l) for l in _state[1].tolist()]
            [print(l) for l in _state[2].tolist()]
            _state, _reward, _done  = step_index(_state, _action)
            _matrix_state  = preprocess(_state)
            print("Reward: {}".format(_reward))
            print("==========")
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
