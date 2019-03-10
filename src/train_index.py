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

import os
import argparse
from tensorflow.python.lib.io import file_io
from io import BytesIO

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model

import random
import numpy as np
from collections import deque
from sklearn.cluster import KMeans

from src.game import init, step, preprocess, play

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

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()

def build_sample_generator(enc_photos, embedding_path, n_clusters, random):
    embedding_matrix = BytesIO(file_io.read_file_to_string(embedding_path, binary_mode=True))
    embedding_matrix = np.load(embedding_matrix)

    kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=1).fit(embedding_matrix)

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
    input_matrix = keras.layers.Input(shape=[SAMPLE_SIZE, SAMPLE_SIZE, 3], name="input_matrix")
    reward = keras.layers.Input(shape=[1], name="reward")

    # Conv2d layers
    prev = input_matrix
    for _ in range(NUM_CONVS):
        conv = keras.layers.Conv2D(FILTERS, kernel_size=(1, SAMPLE_SIZE * KERNAL_MULT), activation='relu', padding=PADDING)(prev)
        prev = conv
    flatten = keras.layers.Flatten()(conv)

    # Build dense layers
    prev = flatten
    for width in LAYERS:
        x = keras.layers.Dense(units=width, activation='relu', kernel_initializer='RandomNormal', use_bias=True)(prev)
        prev = x

    # Build output layers
    out = keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal', name="out")(x)

    # Build optimizer
    rms = keras.optimizers.RMSprop(lr=LEARNING_RATE)

    # Define custom loss function
    def custom_loss(y_true, y_pred):
        cross_entropy = K.binary_crossentropy(y_true, y_pred)
        return K.mean(cross_entropy * reward, keepdims=True)

    # Build and compile training model
    model_train = Model(inputs=[input_matrix, reward], outputs=out)
    model_train.compile(loss=custom_loss, optimizer=rms)
    model_train.summary()

    # Build prediction model
    model_predict = Model(inputs=[input_matrix], outputs=out)
    
    return model_train, model_predict

def build_frames_generator(frames, photos_generator, model_predict, verbose=False):

    def frames_generator():
        while True:
            # Set metrics collection
            epoch_memory = []
            epoch_reward = []
            epoch_intrest = []
            epoch_predict = []
            # Play games until we have the amount of frames we want
            while len(epoch_memory) < frames:
                # Generate new game
                sample_photos = next(photos_generator)

                # init game
                _state = init(sample_photos)
                _matrix_state  = preprocess(_state)
                game_memory = []
                _done = False

                # While there are no rewards in our sample, reroll
                while (np.amax(_matrix_state[:, :, :, 0]) == 0):
                    sample_photos = next(photos_generator)
                    _state = init(sample_photos)
                    _matrix_state  = preprocess(_state)
                        
                count = 0
                while not _done and count < SAMPLE_SIZE ** 2 * 2:
                    _predict = model_predict.predict([_matrix_state], batch_size=1)[0, 0]
                    epoch_predict.append(_predict)
                    #print("Model proabilities: {}".format(_predict))
                    _action = np.random.choice([True, False], p=[_predict, 1 - _predict], replace=False)

                    # Take action
                    _state, _reward, _done = step(_state, _action, NEGATIVE_REWARD)

                    # Log information
                    game_memory.append((_matrix_state, _reward, _action))
                    
                    # Update reward and observation
                    _matrix_state  = preprocess(_state)
                    count += 1

                # Process the rewards
                _m_s, _rewards, _labels = zip(*game_memory)
                _prwd = np.array(discount_rewards(_rewards))
                #_prwd -= _prwd.mean()
                #_std = _prwd.std()
                #_prwd /= _std if _std > 0 else 1

                # Save the gamestates, rewards and actions
                epoch_reward.append(sum(_rewards))
                epoch_intrest.append(sum([r for r in _rewards if r != -1]))
                epoch_memory.extend(zip(_m_s, _prwd, _labels))

            # Shuffle and return frames
            epoch_memory = [tuple(ex) for ex in np.array(epoch_memory)[np.random.permutation(len(epoch_memory))]]
            _matrixs, _rewards, _labels = zip(*epoch_memory)

            # Cut returning arrays down to frames length to give a costant expected shape for tensors
            _matrixs = np.array(_matrixs).reshape(len(_matrixs), SAMPLE_SIZE, SAMPLE_SIZE, 3)
            _labels = np.squeeze(np.array(_labels))
            _rewards = np.squeeze(np.array(_rewards))

            #yield {"input_matrix": _matrixs[:frames], "input_vectors": _vectors, "reward": _rewards[:frames]}, _labels[:frames], sum(epoch_reward) / len(epoch_reward)
            yield [_matrixs[:frames], _rewards[:frames]], _labels[:frames], \
                    sum(epoch_reward) / len(epoch_reward), \
                    sum(epoch_predict) / len(epoch_predict), \
                    sum(epoch_intrest) / len(epoch_intrest)

    return frames_generator()

def main(args):
    # Init globals
    global ROLLOUT_SIZE, LEARNING_RATE, OBSERVATION_DIM, GAMMA, SAMPLE_SIZE, MEMORY_SIZE, NEGATIVE_REWARD, LAYERS, FILTERS, PADDING, NUM_CONVS, KERNAL_MULT

    # Training variables
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    SAMPLE_SIZE = args.sample_size
    ROLLOUT_SIZE = args.rollout_size
    NEGATIVE_REWARD = args.negative_reward

    # Dense layer variables
    LAYERS = [int(neurons) for neurons in args.layers.split(',')]

    # Conv variables
    FILTERS = args.filters
    PADDING = args.padding
    NUM_CONVS = args.n_conv
    KERNAL_MULT = args.kernal_size_mult

    # Read photos
    photos = read_file(args.file_name)

    # Tensorflow summary writing setup
    summary_loss = tf.placeholder(dtype=tf.float32, shape=())
    summary_reward = tf.placeholder(dtype=tf.float32, shape=())
    summary_prediction = tf.placeholder(dtype=tf.float32, shape=())
    summary_intrest = tf.placeholder(dtype=tf.float32, shape=())
    tf.summary.scalar('loss', summary_loss)
    tf.summary.scalar('reward', summary_reward)
    tf.summary.scalar('prediction', summary_prediction)
    tf.summary.scalar('intrest', summary_intrest)
    merged_summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        # Build models
        model_train, model_predict = build_model()

        # Build sample generator
        sample_generator = build_sample_generator(photos, args.image_embedding, args.n_clusters, args.random)

        # Build frame generator
        frame_generator = build_frames_generator(ROLLOUT_SIZE, sample_generator, model_predict)

        # Build summary writer
        summary_path = "{}/summary".format(args.log_dir)
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        # Iterate for n_epochs 
        for e in range(args.n_epoch):
            # Play the games and generate the frames
            print("Generating data...")
            features, targets, average_reward, average_prediction, average_intrest = next(frame_generator)

            #Train!
            print("Training...")
            loss = model_train.train_on_batch(features, targets)
            print("Epoch {}:\nLoss = {}\nAverage Prediction = {}\nAverage Reward = {}\nAverage Intrest = {}".format(e, loss, average_prediction, average_reward, average_intrest))
            if e % args.save_checkpoint_steps:
                # Write scalar summaries
                feed_dict = {summary_loss: loss, summary_reward: average_reward, summary_prediction: average_prediction, summary_intrest: average_intrest}
                summary = sess.run(merged_summaries, feed_dict=feed_dict)
                summary_writer.add_summary(summary, e)

                # Save model
                save_path_train = "{}/model_train{}.h5".format(args.log_dir, e)
                save_path_predict = "{}/model_predict{}.h5".format(args.log_dir, e)
                model_train.save(save_path_train)
                model_predict.save(save_path_predict)
            print("================================")

        play(model_predict, sample_generator, SAMPLE_SIZE, NEGATIVE_REWARD)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('slideshow rl')
    # Hyperparameter arguments
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=1000)
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=50)
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10)
    parser.add_argument(
        '--rollout-size',
        type=int,
        default=500)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99)
    parser.add_argument(
        '--negative-reward',
        type=float,
        default=-1)

    # Conv arguments
    parser.add_argument(
        '--filters',
        type=int,
        default=10)
    parser.add_argument(
        '--padding',
        type=str,
        default='same')
    parser.add_argument(
        '--n-conv',
        type=int,
        default=1)
    parser.add_argument(
        '--kernal-size-mult',
        type=int,
        default=1)

    # Comma seperated string detailing the topology of our dense layers
    parser.add_argument(
        '--layers',
        type=str,
        default="1000,1000")

    # Saving arguments
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=10)
    parser.add_argument(
        '--restore',
        type=str,
        default=None)

    # Input data arguments
    parser.add_argument(
        '--file-name',
        type=str,
        default='data')
    parser.add_argument(
        '--image-embedding',
        type=str,
        default=None)
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None)
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/slideshow_rl')

    # Mode arguments
    parser.add_argument(
        '--play',
        default=False,
        action='store_true')
    parser.add_argument(
        '--random',
        default=False,
        action='store_true')

    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
