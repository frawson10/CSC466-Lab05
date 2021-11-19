from Data import Data
import copy
import sys
import pandas as pd
def classify_data(d: Data, classify_tree):
    classified_result = {}
    for index, data in enumerate(d.data):
        copy_classify_tree = copy.deepcopy(classify_tree)
        decision_flag = False
        while decision_flag == False:
            if 'decision' in copy_classify_tree:
                classified_result[index] = copy_classify_tree['decision']
                break
            var = copy_classify_tree['var']
            categorical = (d.categorical_numerical[var] == 'categorical')

            var_index = d.attributes.index(var)
            data_value = data[var_index]
            if categorical:
                for edge in copy_classify_tree['edges']:
                    edge_value = edge['edge']
                    if edge_value['value'] == data_value:
                        if 'leaf' in edge_value:
                            decision = edge_value['leaf']['decision']
                            classified_result[index] = decision
                            decision_flag = True
                        else:
                            copy_classify_tree = edge_value['node']
                        break
            else:
                for edge in copy_classify_tree['edges']:
                    edge_value = edge['edge']
                    if data_value <= edge_value['value'] and edge_value['direction'] == 'le':
                        if 'leaf' in edge_value:
                            decision = edge_value['leaf']['decision']
                            classified_result[index] = decision
                            decision_flag = True
                        else:
                            copy_classify_tree = edge_value['node']
                        break
                    elif data_value > edge_value['value'] and edge_value['direction'] == 'gt':
                        if 'leaf' in edge_value:
                            decision = edge_value['leaf']['decision']
                            classified_result[index] = decision
                            decision_flag = True
                        else:
                            copy_classify_tree = edge_value['node']
                        break


    return classified_result





if __name__ == '__main__':

    filename = sys.argv[1]
    import json
    try:
        imported_tree = sys.argv[2]
    except:
        imported_tree = filename.replace('.csv','') + '-results.out'
    with open(imported_tree) as f:
        classify_json = json.load(f)
    data = Data(filename, [])
    classified_result = classify_data(data, classify_json['node'])
    classified_result = list(classified_result.values())
    actual_result = list(map(lambda x: x[data.index_class_variable], data.data))

    i = 0
    correct_classified = 0
    while i < len(classified_result):
        if classified_result[i] == actual_result[i]:
            correct_classified += 1
        i += 1

    y_actu = pd.Series(actual_result, name='Actual')
    y_pred = pd.Series(classified_result, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print(df_confusion)

    print('Total Number Of Records Classified: ', len(classified_result))
    print('Total Number Of Records Correctly Classified: ', correct_classified)
    print('Total number of records Incorrectly Classified: ', len(classified_result) - correct_classified)
    print('Overall accuracy and error rate of the classifier: ', correct_classified/len(classified_result) * 100, '%')
