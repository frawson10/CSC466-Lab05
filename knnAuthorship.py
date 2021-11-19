import pandas as pd
import sys
from collections import Counter, defaultdict
import numpy as np

def knn(tfidf_data, words_data, k: int):

    dftf = tfidf_data.copy(deep=True)
    dfwords = words_data.copy(deep=True)


    data_distance = defaultdict(list)
    result = []
    all_vector = dftf.loc[:, 1:]
    all_vector = all_vector.fillna(0)
    all_vector_np = np.array(all_vector)
    all_vector_np = all_vector_np.astype(np.float64)
    all_vector_norm = np.linalg.norm(all_vector_np, axis=1)
    for index, row in dftf.iterrows():
        current_vector = row
        current_vector = current_vector.fillna(0.0)
        current_vector_np = np.array(current_vector[1:])
        current_vector_np = current_vector_np.astype(np.float64)

        closest_distance = cosine_similarity(all_vector_np, current_vector_np, all_vector_norm, k + 1)
        # closest_distance = okapi()
        try:
            closest_distance = np.delete(closest_distance, np.where(closest_distance == index)) #want to exclude the computation to its own vector
        except:
            pass

        all_text_files = [dftf[0][value] for value in closest_distance]

        data_distance[row[0]] = all_text_files

        index += 1



    # for index, row in distancedf.iterrows():
    #
    #     sorted_similarity = sorted(value,reverse=True)
    #     closest_point = sorted_value[0:k]
    #     closest_point_class_variable = [data[x[1]][d.index_class_variable] for x in closest_point]
    #     counter_result = Counter(closest_point_class_variable)
    #     most_popular_class = max(counter_result, key=counter_result.get)
    #     result.append(most_popular_class)
    return data_distance
#
#
def cosine_similarity(all_vector, vector, all_vector_norm, k):
    numerators = np.dot(all_vector, vector)
    vector_normalize = np.linalg.norm(vector)
    denominators = all_vector_norm * vector_normalize
    all_distances = numerators / denominators
    closest_distances = np.argpartition(all_distances, -k)[-k:]
    return closest_distances

def okapi(): #return the closest distance in terms of index excluding itself
    pass


def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == '__main__':

    tfidf_file = sys.argv[1]
    words_file = sys.argv[2]
    ground_truth_file = sys.argv[3]
    k = int(sys.argv[4])
    tfidf_data = pd.read_csv(tfidf_file, header=None, index_col=False, low_memory=False)
    tfidf_data = tfidf_data[3:]
    tfidf_data = tfidf_data.reset_index(drop=True) #reset index of row to start at 0
    words_data = pd.read_csv(words_file, header=None, index_col=False, low_memory=False)

    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip('\n')
            file_author = line.split(',')
            ground_truth[file_author[0]] = file_author[1]

    document_classifier = knn(tfidf_data, words_data, k)


    predicted_result = {}
    for document, closest_neighbors in document_classifier.items():
        closest_neighbors_authors = [ground_truth[neighbor] for neighbor in closest_neighbors]
        plural_neighbor = most_common(closest_neighbors_authors)
        predicted_result[document] = plural_neighbor


    with open(tfidf_file.replace('.csv','') + '_predicted.csv', 'w') as f:
        for document, predicted_neighbor in predicted_result.items():
            f.write(document + ',' + predicted_neighbor)
            f.write('\n')

    actual_result = list(ground_truth.values())
    knn_result = list(predicted_result.values())

    i = 0
    correct_classified = 0
    while i < len(knn_result):
        if knn_result[i] == actual_result[i]:
            correct_classified += 1
        i += 1

    y_actu = pd.Series(actual_result, name='Actual')
    y_pred = pd.Series(knn_result, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print(df_confusion)

    print('Total Number Of Records Classified: ', len(knn_result))
    print('Total Number Of Records Correctly Classified: ', correct_classified)
    print('Total number of records Incorrectly Classified: ', len(knn_result) - correct_classified)
    print('Overall accuracy and error rate of the classifier: ', correct_classified/len(knn_result) * 100, '%')


    # import pdb; pdb.set_trace()