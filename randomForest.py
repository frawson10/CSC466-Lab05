from Data import Data
from InduceC45 import algorithm_c45, Node
from collections import Counter
from classify import classify_data
import sys
import random
import pandas as pd

def random_forest(d: Data, attribute_amount: int, data_point_amount: int, tree_amount: int, threshold, ground_truth: dict, gain_ratio):
    trees = []
    for i in range(1, tree_amount):
        # import pdb; pdb.set_trace()
        randAttr = random.sample(d.attributes, attribute_amount)
        # randDP = random.sample(d.data, k)
        randDP = [random.choice(d.data) for x in range(data_point_amount)]

        # import pdb; pdb.set_trace()
        head_node = Node('')
        tree = algorithm_c45(d, randDP, randAttr, threshold, head_node, gain_ratio, ground_truth)
        trees.append(tree)
    return trees

def trees_evaluation(d, trees):
    classified_result = []
    final_result = []

    for tree in trees:

        classified_result.append(classify_data(d, tree.__dict__))

    for i in range(len(d.data)):
        trees_classified_result = [a[i] for a in classified_result]
        counter_result = Counter(trees_classified_result)
        most_popular_class = max(counter_result, key=counter_result.get)

        final_result.append(most_popular_class)
    return final_result




if __name__ == '__main__':
    filename = sys.argv[1]
    ground_truth_file = sys.argv[2]
    m = int(sys.argv[3])  # NumAttributes
    k = int(sys.argv[4])  # NumDataPoints
    n = int(sys.argv[5])  # NumTrees
    threshold = float(sys.argv[6])
    gain_ratio = sys.argv[7] == '1'

    data = Data(filename, [])

    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip('\n')
            file_author = line.split(',')
            ground_truth[file_author[0]] = file_author[1]

    if k > len(data.data):
        k = len(data.data)
    trees = random_forest(data, m, k, n, threshold, ground_truth, gain_ratio)
    result = trees_evaluation(data, trees)

    actual_result = list(ground_truth.values())

    i = 0
    correct_classified = 0
    while i < len(result):
        if result[i] == actual_result[i]:
            correct_classified += 1
        i += 1

    y_actu = pd.Series(actual_result, name='Actual')
    y_pred = pd.Series(result, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print(df_confusion)

    print('Total Number Of Records Classified: ', len(result))
    print('Total Number Of Records Correctly Classified: ', correct_classified)
    print('Total number of records Incorrectly Classified: ', len(result) - correct_classified)
    print('Overall accuracy and error rate of the classifier: ', correct_classified/len(result) * 100, '%')

    # import pdb; pdb.set_trace()