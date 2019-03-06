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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model

import random
import numpy as np
from collections import deque

from src.game import init, step, init_index, step_index, preprocess, play

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
    input_matrix = keras.layers.Input(shape=[SAMPLE_SIZE, SAMPLE_SIZE, 1], name="input_matrix")
    input_vectors = keras.layers.Input(shape=[SAMPLE_SIZE * 2], name="input_vectors")
    reward = keras.layers.Input(shape=[1], name="reward")

    # Conv2d layers
    conv1 = keras.layers.Conv2D(10, kernel_size=(1, SAMPLE_SIZE), activation='relu')(input_matrix)
    conv2 = keras.layers.Conv2D(10, kernel_size=(SAMPLE_SIZE, 1), activation='relu')(conv1)
    flatten = keras.layers.Flatten()(conv2)

    # Combine layers
    combine = keras.layers.concatenate([flatten, input_vectors])

    # Build dense layers
    x = keras.layers.Dense(units=1000, activation='relu', kernel_initializer='RandomNormal', name="layer_1", use_bias=False)(combine)
    x = keras.layers.Dense(units=1000 , activation='relu', kernel_initializer='RandomNormal', name="layer_2", use_bias=False)(x)

    # Build output layers
    out = keras.layers.Dense(units=SAMPLE_SIZE, activation='softmax', kernel_initializer='RandomNormal', name="out")(x)

    # Build optimizer
    rms = keras.optimizers.RMSprop(lr=LEARNING_RATE)
    #rms = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
    #rms = keras.optimizers.Adam(lr=LEARNING_RATE)

    # Build distribution stratedgy
    #distribution = tf.contrib.distribute.MirroredStrategy()

    # Define custom loss function
    def custom_loss(y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        #cross_entropy = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        #cross_entropy = keras.losses.binary_crossentropy(y_true, y_pred)
        #loss = reward * cross_entropy
        #cross_entropy = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        return K.mean(cross_entropy * reward, keepdims=True)
        #return cross_entropy * reward

    # Build and compile training model
    model_train = Model(inputs=[input_matrix, input_vectors, reward], outputs=out)
    #model_train.compile(loss=custom_loss, optimizer=rms, distribute=distribution)
    model_train.compile(loss=custom_loss, optimizer=rms)

    # Build prediction model
    model_predict = Model(inputs=[input_matrix, input_vectors], outputs=out)
    model_predict._make_predict_function()
    
    return model_train, model_predict

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
                        
                count = 0
                while not _done and count < SAMPLE_SIZE * 2:
                    _predict = model_predict_global.predict([_matrix_state, _vector_state], batch_size=1)[0]
                    _action = np.random.choice(range(SAMPLE_SIZE), p=_predict, replacement=False)

                    # Take action
                    _state, _reward, _done = step_index(_state, _action)

                    # Log information
                    game_memory.append((_matrix_state, _vector_state, _reward, _action))
                    
                    # Update reward and observation
                    _matrix_state, _vector_state = preprocess(_state)
                    total_reward += _reward
                    count += 1

                # Process the rewards
                _m_s, _v_s, _rewards, _labels = zip(*game_memory)
                _prwd = np.array(discount_rewards(_rewards))
                _prwd -= _prwd.mean()
                _std = _prwd.std()
                _prwd /= _std if _std > 0 else 1

                # Save the gamestates, rewards and actions
                epoch_reward.append(sum(_rewards))
                epoch_memory.extend(zip(_m_s, _v_s, _prwd, _labels))

            # Shuffle and return frames
            epoch_memory = [tuple(ex) for ex in np.array(epoch_memory)[np.random.permutation(len(epoch_memory))]]
            _matrixs, _vectors, _rewards, _labels = zip(*epoch_memory)

            # Cut returning arrays down to frames length to give a costant expected shape for tensors
            _matrixs = np.array(_matrixs).reshape(len(_matrixs), SAMPLE_SIZE, SAMPLE_SIZE, 1)
            _vectors = np.array(_vectors)[:frames].reshape(ROLLOUT_SIZE, SAMPLE_SIZE * 2)
            _labels = np.squeeze(np.array(_labels))
            _labels = np.squeeze(np.array(_rewards))

            yield {"input_matrix": _matrixs[:frames], "input_vectors": _vectors, "reward": _rewards[:frames]}, _labels[:frames], sum(epoch_reward) / len(epoch_reward)

    return frames_generator()

def main(args):
    # Init globals
    global ROLLOUT_SIZE, LEARNING_RATE, OBSERVATION_DIM, GAMMA, SAMPLE_SIZE, MEMORY_SIZE

    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    SAMPLE_SIZE = args.sample_size
    ROLLOUT_SIZE = args.rollout_size

    # Read photos
    photos = read_file(args.file_name)

    # Build sample generator
    sample_generator = build_sample_generator(photos, args.image_embedding, args.n_clusters, args.random)

    # Build generator and dataset
    memory, dataset = build_dataset()
    frame_generator = build_frames_generator(ROLLOUT_SIZE, sample_generator)

    # Tensorflow summary writing setup
    summary_loss = tf.placeholder(dtype=tf.float32, shape=())
    summary_reward = tf.placeholder(dtype=tf.float32, shape=())
    tf.summary.scalar('loss', summary_loss)
    tf.summary.scalar('reward', summary_reward)
    merged_summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        # Build models
        model_train, model_predict = build_model()

        # Iterate for n_epochs 
        for e in range(args.n_epoch):
            # Play the games and generate the frames
            print("Generating data...")
            features, targets, average_batch_reward = next(frame_generator)

            #Train!
            print("Training...")
            loss = model_train.train_on_batch(features, targets)
            print("Epoch {}:\nLoss = {}\nAverage Reward = {}".format(e, loss, average_batch_reward))
            if e % args.save_checkpoint_steps:
                summary = sess.run(merged_summaries, feed_dict={summary_loss: loss, summary_reward: average_batch_reward})
                summary_writer.add_summary(summary, e)
                save_path = "{}/model_{}.h5".format(args.log_dir, e)
                model_train.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pizza cutter trainer')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=1000)
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=25)
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20)
    parser.add_argument(
        '--rollout-size',
        type=int,
        default=5000)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-3)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9)

    parser.add_argument(
        '--period',
        type=int,
        default=5)
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=10)

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
        default='/tmp/pizza_output')

    parser.add_argument(
        '--restore',
        type=str,
        default=None)
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
