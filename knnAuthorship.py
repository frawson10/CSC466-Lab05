from typing import List
from Data import Data
from collections import defaultdict
import sys
import pandas as pd
from collections import Counter
def knn(d: Data, k: int):
    data_distance = defaultdict(list)
    data = d.data
    result = []
    for index, current_data in enumerate(data):
        other_data = data[0:index] + data[index+1:]
        for j , other in enumerate(other_data):
            distance_computed = eucledian_distance(d, current_data, other)
            if j < index:
                data_distance[index].append((distance_computed, j))
            else:
                data_distance[index].append((distance_computed,j + 1))

    for index, value in data_distance.items():
        sorted_value = sorted(value, key=lambda x: x[0])
        closest_point = sorted_value[0:k]
        closest_point_class_variable = [data[x[1]][d.index_class_variable] for x in closest_point]
        counter_result = Counter(closest_point_class_variable)
        most_popular_class = max(counter_result, key=counter_result.get)
        result.append(most_popular_class)
    return result


def eucledian_distance(d: Data, current_data: List[any], other_data: List[any]):
    i = 0
    distance_total = 0
    while i < len(current_data):
        attribute = d.attributes[i]
        categorical = d.categorical_numerical[attribute] == 'categorical'
        if not categorical: #means numeric

            try:
                current_value = float(current_data[i])
                other_value = float(other_data[i])
                distance_total = distance_total + (current_value - other_value) ** 2
            except:
                i += 1
                continue
        else: #for categorical
            #just compute it using ascii value of each word and add them up
            current_value = list(current_data[i])
            other_value = list(other_data[i])
            # import pdb; pdb.set_trace()
            similarity = jaccard_similarity(current_value, other_value)
            distance = 1 - similarity
            distance_total = distance_total + distance
        i += 1
    return distance_total ** (0.5)

def jaccard_similarity(current_value, other_value):
    intersection = len(list(set(current_value).intersection(list(set(other_value)))))
    union = (len(current_value) + len(other_value) - intersection)
    return float(intersection) / union


if __name__ == '__main__':
    filename = sys.argv[1]
    k = sys.argv[2]
    data = Data(filename, [])
    knn_result = knn(data, int(k))
    actual_result = list(map(lambda x: x[data.index_class_variable], data.data))

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