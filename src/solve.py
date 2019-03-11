import argparse
from io import BytesIO

import numpy as np
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import load_model

from sklearn.cluster import KMeans

from src.game import init, step, preprocess

def best_cluster(labels, n_clusters, sample_size):
	# Find cluster_size of each cluster
	cluster_sizes = np.array([len(np.where(i == labels)[0]) for i in range(n_clusters)])

        # Find cluster with size closest to sample_size
	cluster_index = np.argmin(abs((cluster_sizes / sample_size) - 1))

        # Save chosen cluster's size
	cluster_size = cluster_sizes[cluster_index]

        # Save indices corrosponding to chosen cluster
	cluster_indices = np.where(cluster_index == labels)[0]

	return cluster_index, cluster_size, cluster_indices

def sample_generator(enc_photos, embedding_matrix, embedding_indices, sample_size):
        # Heuristic to calculate the number of clusters we will be using so that we have less clusters the less slides we have left
	n_clusters = min(int(len(enc_photos.items()) / sample_size), 100)
	
        # Cluster our embeddings
	kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=1).fit(embedding_matrix)

        # Select the cluster we'll be using
	cluster_index, cluster_size, cluster_indices = best_cluster(kmeans.labels_, n_clusters, sample_size)
	other_cluster_indices = np.where(cluster_index != kmeans.labels_)[0]

        # Get first vertical image's index
	first_vertical = np.argmax(embedding_indices >= 500)

	remove_indices = []
	original_indices = [[] for _ in range(sample_size)]
	sample = [set() for _ in range(sample_size)]
        # Build our sample
	for i in range(sample_size):
		if i < cluster_size: # If cluster has remaining photos, then pick from cluster
			sample_index = np.random.choice(cluster_indices, 1)[0]
			original_embedding_index = embedding_indices[sample_index]

			remove_indices.append(sample_index)
			original_indices[i].append(original_embedding_index)

			sample[i] = enc_photos.get(original_embedding_index)
			cluster_indices = np.delete(cluster_indices, np.where(sample_index == cluster_indices)[0])
		else: # if cluster has no remaining photos, pick from other cluster
			sample_index = np.random.choice(other_cluster_indices, 1)[0]
			original_embedding_index = embedding_indices[sample_index]

			remove_indices.append(sample_index)
			original_indices[i].append(original_embedding_index)

			sample[i] = enc_photos.get(original_embedding_index)
			other_cluster_indices = np.delete(other_cluster_indices, np.where(sample_index == other_cluster_indices)[0])

		if original_embedding_index >= 500: # if vertical (first 500 horizontal, last 500 vertical)
			sample_index = np.random.choice(other_cluster_indices[other_cluster_indices >= first_vertical], 1)[0]
			original_embedding_index = embedding_indices[sample_index]

			remove_indices.append(sample_index)
			original_indices[i].append(original_embedding_index)

			sample[i] |= enc_photos.get(original_embedding_index)
			other_cluster_indices = np.delete(other_cluster_indices, np.where(sample_index == other_cluster_indices)[0])

	return sample, original_indices, remove_indices, cluster_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser("solver")
    parser.add_argument("--image-matrix", type=str, default="embeddings/100dim__10mil__image_embedding.npy")
    parser.add_argument("--input-file", type=str, default="data/c_memorable_moments.txt")
    parser.add_argument("--output-file", type=str, default="out/submission.txt")
    parser.add_argument("--input-model", type=str, default="../data/model.h5")
    args = parser.parse_args()

    # Load model
    model = load_model(args.input_model)

    # Load file
    with file_io.FileIO(args.input_file, 'r') as f:
        photos = f.readlines()[1:]
        photos = [photo.strip().split() for photo in photos]

    # Build list of all tags
    tags = [x[2:] for x in photos]
    tags = [item for sublist in tags for item in sublist]

    # Build line index -> tags dictionary
    enc_photos = dict()
    photos_as_tags = [x[2:] for x in photos]

    index = 0
    for el in photos_as_tags:
        m = map(lambda x: tags.index(x), el)
        enc_photos[index] = set(m)
        index += 1

    # Load Slide2vec embedding
    embedding_matrix = BytesIO(file_io.read_file_to_string(args.image_matrix, binary_mode=True))
    embedding_matrix = np.load(embedding_matrix)
    embedding_indices = np.array([i for i in range(len(embedding_matrix))])

    sample_size = 5
    full_reward = 0
    all_slides = []

    # While we have sufficient slides left to connect:
    while len(embedding_matrix) > sample_size * 2:

        # GENERATE SAMPLE
        sample_, sample_indices, remove_indices, cluster_size = sample_generator(enc_photos, embedding_matrix, embedding_indices, sample_size)

        state = init(sample_)
        matrix_state  = preprocess(state)
        done = False

        # starts from first slide because assumes that the slides are shuffled
        slideshow = [0]
        total_reward = 0

        # Play the game!
        while not done:
            predict = model.predict([matrix_state], batch_size=1)[0]
            action = True if predict >= 0.5 else False
            state, reward, done, added_slide_i = step(state, action)
            matrix_state  = preprocess(state)

            total_reward += reward if reward > 0 else 0
            if added_slide_i is not None: slideshow.append(added_slide_i)
        
        # Save data
        for slide in slideshow:
            all_slides.append(sample_indices[slide])
        result = [photo for slide in slideshow for photo in sample_indices[slide]]
        full_reward += total_reward

        print('Slideshow: {}'.format(result))
        print('Reward: {}'.format(total_reward))

        # REMOVE INDICES #
        embedding_matrix = np.delete(embedding_matrix, remove_indices, axis=0)
        embedding_indices = np.delete(embedding_indices, remove_indices, axis=0)
        for sample_slide in sample_indices:
            for photo in sample_slide:
                enc_photos.pop(photo)

    # Write to file for Google
    print("Starting write...")
    with file_io.FileIO(args.output_file, 'a') as f:
        f.write(str(len(all_slides)) + "\n")
        for images in all_slides:
            f.write(" ".join(str(image) for image in images) + "\n")
    print("Write done!")

    print('Total Reward: {}'.format(full_reward))
