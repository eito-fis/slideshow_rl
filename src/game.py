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

    current_matrix = np.zeros((n_slides, n_slides))
    current_matrix[0][0] = 1.

    considering_matrix = np.zeros((n_slides, n_slides))
    considering_matrix[0][1] = 1

    state = (rewards, current_matrix, considering_matrix)
    return state

def step(state, action):
    rewards, current_matrix, considering_matrix = state

    current_i = np.argmax(current_matrix) % len(current_matrix)
    considering_i = np.argmax(considering_matrix) % len(considering_matrix)
    
    if action:
        # If yes to considering slide

        reward = rewards[current_i, considering_i]

        # Penalize picking a square with no reward
        if reward == 0: reward = -1

        # cannot go to the current slide anymore
        rewards[current_i] = -1.
        rewards[:,current_i] = -1.

        # Set new current position
        current_matrix[current_i, current_i] = 0
        current_matrix[considering_i][considering_i] = 1

        # Set new considering position
        considering_matrix[current_i, considering_i] = 0
        
        current_i = considering_i
        # Done if all rewards in the row we move to our negative one
        # Otherwise, move our considering location to the first -1 slide
        if all(r == -1 for r in rewards[current_i]):
            done = True
        else:
            while rewards[current_i][considering_i] == -1:
                considering_i = 0 if (considering_i + 1 == len(rewards[current_i])) else considering_i + 1
            considering_matrix[current_i][considering_i] = 1
            done = False

        state = (rewards, current_matrix, considering_matrix)
        return state, reward, done
    else:
        # If no to considering slide

        # Move to next slide
        considering_matrix[current_i, considering_i] = 0
        
        # Look until we hit a non -1 value or until we come back to our current value
        considering_i = 0 if considering_i + 1 == len(rewards) else considering_i + 1
        while rewards[current_i][considering_i] == -1 and current_i != considering_i:
            considering_i = 0 if considering_i + 1 == len(rewards) else considering_i + 1
        considering_matrix[current_i][considering_i] = 1

        # Done if we've looped through all of our considerations
        done = True if considering_i == current_i else False

        # Reward for hitting no
        reward = 0

        state = (rewards, current_matrix, considering_matrix)
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
    considering_matrix  = _state[2]

    # Reshape data
    matrix = np.stack((adj_matrix, current_matrix, considering_matrix), axis=-1)
    matrix = np.reshape(matrix, (1, len(adj_matrix), len(adj_matrix), 3))
    
    return matrix

def play(model, gen, sample_size):
    while True:
        print("==========STARTING ROLLOUT==========")
        sample_photos = next(gen)

        # init game
        _state = init(sample_photos)
        _matrix_state  = preprocess(_state)
        _done = False
        total_reward = 0
        total_intrest = 0
        count = 0

        while not _done and count < sample_size * 2:
            _predict = model.predict([_matrix_state], batch_size=1)[0]
            print("==========")
            print("Proability: {}".format(_predict))
            _action = True if _predict >= 0.5 else False
            print("Action being taken: {}".format(_action))
            print("State:")
            [print(l) for l in _state[0].tolist()]
            print()
            [print(l) for l in _state[1].tolist()]
            print()
            [print(l) for l in _state[2].tolist()]
            _state, _reward, _done  = step(_state, _action)
            _matrix_state  = preprocess(_state)
            print("Reward: {}".format(_reward))
            print("==========")
            total_reward += _reward
            if _reward > 0: total_inrest += _reward
            count += 1
        print("Total reward: {}".format(total_reward))
        print("Total intrest: {}".format(total_intrest))
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
