# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from tensorflow.python.lib.io import file_io
from io import BytesIO

from collections import deque, Counter
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow import keras
from sklearn.cluster import KMeans

from src.game import init, step, step_index, init_index, preprocess, play

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def read_file(file_name):
    # Read file
    with open(file_name) as f:
        photos = f.readlines()
        photos = [x.strip() for x in photos]

    # Turn each photo into an list of its values
    photos.pop(0)
    photos = [x.split() for x in photos]

    # List of only the tags
    tags = [x[2:] for x in photos]
    tags = [item for sublist in tags for item in sublist]

    enc_photos = []
    photos_as_tags = [x[2:] for x in photos]

    for el in photos_as_tags:
        m = map(lambda x: tags.index(x), el)
        enc_photos.append(set(m))
    enc_photos = np.array(enc_photos)

    return enc_photos

def build_sample_generator(enc_photos, embedding_path, n_clusters, random):
    embedding_matrix = BytesIO(file_io.read_file_to_string(embedding_path, binary_mode=True))
    embedding_matrix = np.load(embedding_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_matrix)

    def sample_generator():
        while True:
            if random:
                yield np.random.choise(enc_photos, SAMPLE_SIZE)
            # pick cluster from probablity distribution of clusters (likely to pick higher density clusters)
            cluster_index = np.random.choice(kmeans.labels_, 1)
            cluster_indices = np.where(cluster_index == kmeans.labels_)[0]
            cluster_size = len(cluster_indices)
            # randomly sample from cluster if cluster is bigger than SAMPLE_SIZE
            if cluster_size >= SAMPLE_SIZE:
                sample_indices = np.random.choice(cluster_indices, SAMPLE_SIZE, replace=False)
                sample_ = enc_photos[sample_indices]
                yield sample_
            # sample entire cluster and fill remainder with random samples
            elif cluster_size > SAMPLE_SIZE * .8:
                sample_indices = cluster_indices
                remainder = SAMPLE_SIZE - cluster_size
                sample_indices = np.append(sample_indices, np.random.choice(np.where(cluster_index != kmeans.labels_)[0], remainder, replace=True))
                sample_ = enc_photos[sample_indices]
                yield sample_

    return sample_generator()

def build_model():
    # Build input layers
    input_matrix = keras.layers.Input(shape=[SAMPLE_SIZE, SAMPLE_SIZE, 1], name="input_matrix")
    input_vectors = keras.layers.Input(shape=[SAMPLE_SIZE * 2], name="input_vectors")

    # Conv2d layers
    conv1 = keras.layers.Conv2D(10, kernel_size=(1, SAMPLE_SIZE), activation='relu')(input_matrix)
    conv2 = keras.layers.Conv2D(10, kernel_size=(SAMPLE_SIZE, 1), activation='relu')(conv1)
    flatten = keras.layers.Flatten()(conv2)

    # Combine layers
    combine = keras.layers.concatenate([flatten, input_vectors])

    # Build dense layers
    x = keras.layers.Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform', name="layer_1", use_bias=False)(combine)
    x = keras.layers.Dense(units=1000 , activation='relu', kernel_initializer='glorot_uniform', name="layer_2", use_bias=False)(x)

    # Build output layers
    out = keras.layers.Dense(units=SAMPLE_SIZE, activation='softmax', kernel_initializer='RandomNormal', name="out")(x)

    # Build optimizer
    rms = keras.optimizers.RMSprop(lr=LEARNING_RATE)

    # Build and compile training model
    model = Model(inputs=[input_matrix, input_vectors], outputs=out)
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=rms,
        metrics=['accuracy'])

    return model
        
### FROM GOOGLE ###
def create_distance_callback(dist_matrix, max_reward):
  # Create the distance callback.

  def distance_callback(from_node, to_node):
    return int(max_reward - dist_matrix[from_node][to_node])

  return distance_callback

def gen_solution(matrix, tsp_size):
    # Make callbacks
    matrix = np.squeeze(matrix)

    # Consider distance as (max_reward - reward)
    # so the program prioritizes large rewards
    m_max = np.amax(matrix)
    dist_callback = create_distance_callback(matrix, m_max)

    # Make the solver
    routing = pywrapcp.RoutingModel(tsp_size, 1, 0)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Get solution
    solution = []
    node = routing.Start(0)
    while not routing.IsEnd(node):
        solution.append(node)
        node = assignment.Value(routing.NextVar(node))

    # Remove "useless" moves
    prev = solution[0]
    ret = []
    for index in range(1, len(solution) - 1):
        # The move is usefull if either moving to it gives us reward, or if our next move gives us reward
        # Otherwise, we can just skip it
        if (matrix[prev][solution[index]] != 0):
            ret.append(solution[index])
        elif (matrix[solution[index]][solution[index + 1]] != 0):
            ret.append(solution[index])
        prev = solution[index]

    # Check if last move is usefull
    if len(ret) > 0 and matrix[ret[-1]][solution[-1]] != 0:
        ret.append(solution[-1])

    return ret

def build_frames_generator(frames, photos_generator, verbose=False):

    def frames_generator():
        while True:
            # Play games until we have the amount of frames we want
            epoch_memory = []
            epoch_reward = []
            while len(epoch_memory) < frames:
                # Generate new game
                sample_photos = next(photos_generator)

                # init game
                _state = init_index(sample_photos)
                _matrix_state, _vector_state = preprocess(_state)
                game_memory = []
                _done = False

                # While there are no rewards in our sample, reroll
                while (np.amax(_matrix_state) == 0):
                    sample_photos = next(photos_generator)
                    _state = init_index(sample_photos)
                    _matrix_state, _vector_state = preprocess(_state)
                        
                # init solution
                _solved = gen_solution(_matrix_state, SAMPLE_SIZE)
                if verbose: print(_solved)

                # Take solution actions
                for index in _solved:
                    if verbose:
                        print(_state[0])
                        print(_state[1])
                        print(_state[2])
                    _action = index
                    _state, _reward, _done  = step_index(_state, _action)
                    if verbose:
                        print("Action being taken: {}".format(_action))
                        print("Reward: {}".format(_reward))
                        print("Done? {}".format(_done))
                    game_memory.append((_matrix_state, _vector_state, _reward, _action))
                    _matrix_state, _vector_state = preprocess(_state)

                # Once the game has finished, process the rewards then save to epoch_memory
                _m_s, _v_s, _rewards, _l = zip(*game_memory)
                epoch_reward.append(sum(_rewards))
                epoch_memory.extend(zip(_m_s, _v_s, _l))

            # Shuffle and return frames
            epoch_memory = [tuple(ex) for ex in np.array(epoch_memory)[np.random.permutation(len(epoch_memory))]]
            _matrixs, _vectors, _labels = zip(*epoch_memory)

            # Cut returning arrays down to frames length to give a costant expected shape for tensors
            _matrixs = np.array(_matrixs).reshape(len(_matrixs), SAMPLE_SIZE, SAMPLE_SIZE, 1)
            _vectors = np.array(_vectors)[:frames].reshape(ROLLOUT_SIZE, SAMPLE_SIZE * 2)
            _labels = np.squeeze(np.array(_labels))

            print(sum(epoch_reward) / len(epoch_reward))

            yield ({"input_matrix": _matrixs[:frames], "input_vectors": _vectors}, _labels[:frames])

    return frames_generator()

def build_dataset(gen):
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=({"input_matrix": tf.float32, "input_vectors": tf.float32}, tf.int32),
        output_shapes=({"input_matrix": (ROLLOUT_SIZE, SAMPLE_SIZE, SAMPLE_SIZE, 1), "input_vectors": (ROLLOUT_SIZE, SAMPLE_SIZE * 2)}, (ROLLOUT_SIZE))
        )
    dataset.batch(BATCH_SIZE).repeat()
    return dataset

def main(args):
    # Init global variables
    global BATCH_SIZE, LEARNING_RATE, SAMPLE_SIZE, ROLLOUT_SIZE

    BATCH_SIZE = args.batch_size
    ROLLOUT_SIZE = args.rollout_size
    SAMPLE_SIZE = args.sample_size
    LEARNING_RATE = args.learning_rate

    # Build model
    if args.restore == None:
        model = build_model()
    else:
        model = load_model(args.restore)

    # Read photos
    photos = read_file(args.file_name)

    # Build sample generator
    sample_generator = build_sample_generator(photos, args.image_embedding, args.cluster_size, args.random)

    # Build callbacks)
    tbCallBack = callbacks.TensorBoard(log_dir=args.output_dir, histogram_freq=0, write_graph=True, write_images=True)
    filepath = args.output_dir + "/{epoch:02d}.hdf5"
    saveCallBack = callbacks.ModelCheckpoint(filepath, monitor='val_loss', period=args.save_checkpoint_steps)

    # Build generator and dataset
    frame_generator = lambda: build_frames_generator(ROLLOUT_SIZE, sample_generator)
    dataset = build_dataset(frame_generator)

    # Train!
    model.fit(
        dataset,
        batch_size=ROLLOUT_SIZE,
        epochs=args.n_epoch,
        steps_per_epoch=1,
        verbose=1,
        callbacks=[tbCallBack, saveCallBack])

    play(model, sample_generator, SAMPLE_SIZE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('slideshow classifier')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=1000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=3000)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='log')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/pizza_output')
    parser.add_argument(
        '--file-name',
        type=str,
        default='data/a_example.txt')
    parser.add_argument(
        '--image-embedding',
        type=str,
        default='data/image_matrix.npy')
    parser.add_argument(
        '--cluster-size',
        type=int,
        default=20)
    parser.add_argument(
        '--rollout-size',
        type=int,
        default=5000)
    parser.add_argument(
        '--restore',
        type=str,
        default=None)
    parser.add_argument(
        '--play',
        default=False,
        action='store_true')
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=5)
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4)
    parser.add_argument(
        '--random',
        default=False,
        action='store_true')

    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
