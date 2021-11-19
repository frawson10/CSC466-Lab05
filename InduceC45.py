from Data import Data
# from Node import Node, LeafNode, Edge
import copy
import math
import sys
class Node:
    def __init__(self,value):
        self.var = value
        self.edges = []

class LeafNode:
    def __init__(self,decision, p ):
        self.decision = decision
        self.p = p


def  algorithm_c45(d: Data, data: [], a: [], threshold: float, t: Node, gain_ratio : bool, ground_truth: dict):

    d_value = data
    documents = list(map(lambda x: x[d.index_class_variable], d_value))
    d_class = [ground_truth[x] for x in documents]
    #homogenous
    if len(set(d_class)) == 1:
        leaf_node = LeafNode(d_class[0], 1.0)
        t = leaf_node

    elif a == []:
        most_occurence_class = find_most_frequent_label(d_class)
        p = d_class.count(most_occurence_class) / len(d_class)
        leaf_node = LeafNode(most_occurence_class, p)
        t = leaf_node
    else:
        if gain_ratio:
            new_attribute, alpha, numerical_flag = selectSplittingAttribute_information_gain_ratio(d,data,a,threshold)
        else:
            new_attribute, alpha, numerical_flag = selectSplittingAttribute_information_gain(d,data,a,threshold)

        if new_attribute == None:
            most_occurence_class = find_most_frequent_label(d_class)
            p = d_class.count(most_occurence_class) / len(d_class)
            leaf_node = LeafNode(most_occurence_class,p)
            t = leaf_node
        else:
            # r = Tree(new_attribute)
            t.var = new_attribute
            attribute_index = d.attributes.index(new_attribute)
            categorical = (d.categorical_numerical[new_attribute] == 'categorical')
            # import pdb; pdb.set_trace()
            if categorical:
                for domain in d.attribute_map[new_attribute]:
                    filter_data = list(filter(lambda x: x[attribute_index] == domain, data))
                    if filter_data != []:
                        excluding_attr = list(filter(lambda x: x != new_attribute,a))
                        tv = Node('')
                        tv = algorithm_c45(d, filter_data, excluding_attr, threshold, tv, gain_ratio, ground_truth)
                        if type(tv) == Node:
                            edge = {'edge':
                                        {
                                            'value' : domain,
                                            'node': tv.__dict__
                                        }
                                    }
                        else:
                            edge = {'edge':
                                {
                                    'value': domain,
                                    'leaf': tv.__dict__
                                }
                            }
                        t.edges.append(edge)

                    else:

                        most_occurence_class = find_most_frequent_label(d_class)
                        p = d_class.count(most_occurence_class) / len(d_class)
                        leaf_node = LeafNode(most_occurence_class, p)
                        edge = {'edge':
                                {
                                    'value': domain,
                                    'leaf': leaf_node.__dict__
                                }
                            }
                        t.edges.append(edge)
            else:
                tv_left = Node('')
                filtered_left_data = list(filter(lambda x: x[attribute_index] <= alpha, data))

                tv_right = Node('')
                filtered_right_data = list(filter(lambda x: x[attribute_index] > alpha, data))


                if len(filtered_right_data) > 0:
                    tv_left = algorithm_c45(d, filtered_left_data, a, threshold, tv_left, gain_ratio, ground_truth)
                    tv_right = algorithm_c45(d, filtered_right_data, a, threshold, tv_right, gain_ratio, ground_truth)

                    if type(tv_left) == Node:
                        left_edge = {'edge':
                            {
                                'value': alpha,
                                'direction': 'le',
                                'node': tv_left.__dict__
                            }
                        }
                    else:
                        left_edge = {'edge':
                            {
                                'value': alpha,
                                'direction': 'le',
                                'leaf': tv_left.__dict__
                            }
                        }

                    if type(tv_right) == Node:
                        right_edge = {'edge':
                            {
                                'value': alpha,
                                'direction': 'gt',
                                'node': tv_right.__dict__
                            }
                        }
                    else:
                        right_edge = {'edge':
                            {
                                'value': alpha,
                                'direction': 'gt',
                                'leaf': tv_right.__dict__
                            }
                        }
                    t.edges.append(left_edge)
                    t.edges.append(right_edge)
                else:
                    left_data_class = list(map(lambda x: x[d.index_class_variable], filtered_left_data))
                    import pdb; pdb.set_trace()
                    most_occurence_class = find_most_frequent_label(left_data_class)
                    p = left_data_class.count(most_occurence_class) / len(left_data_class)
                    leaf_node = LeafNode(most_occurence_class, p)
                    edge = {'edge':
                        {
                            'value': alpha,
                            'leaf': leaf_node.__dict__
                        }
                    }
                    t.edges.append(edge)


    return t


def find_most_frequent_label(d):
    map_class = {}
    for value in d:
        if value not in map_class:
            map_class[value] = 1
        else:
            map_class[value] += 1

    return max(map_class, key=map_class.get)



def selectSplittingAttribute_information_gain(d: Data, data: [], a: [],threshold: float):
    total_enthropy = calculate_entropy(d, data)
    enthropy_map = {}
    alpha_attribute = {}
    for attr in a:
        categorical = d.categorical_numerical[attr]
        if categorical == 'numerical': #numerical
            alpha = find_best_split(d,data,attr)
            attr_enthropy = calculate_entropy_numeric_attribute(d,data,alpha,attr)
            alpha_attribute[attr] = alpha
        else:
            attr_enthropy = gain_attr(d, data, attr)
        enthropy_map[attr] = total_enthropy - attr_enthropy


    if enthropy_map:
        if d.class_variable in enthropy_map:
            del enthropy_map[d.class_variable]
        if enthropy_map:
            max_attribute = max(enthropy_map, key=enthropy_map.get)
            if enthropy_map[max_attribute] > threshold:
                return max_attribute, alpha_attribute.get(max_attribute,0), d.categorical_numerical[max_attribute] == 'numerical'

    return None, None, None

def selectSplittingAttribute_information_gain_ratio(d: Data, data: [],a: [],threshold: float):
    total_enthropy = calculate_entropy(d, data)
    enthropy_map = {}
    gain_ratio = {}
    alpha_attribute = {}
    for attr in a:
        categorical = d.categorical_numerical[attr]
        if categorical == 'numerical':
            alpha = find_best_split(d, data,attr)
            attr_enthropy = calculate_entropy_numeric_attribute(d, data, alpha, attr)
            alpha_attribute[attr] = alpha
        else:
            attr_enthropy = gain_attr(d, data, attr)
        enthropy_map[attr] = total_enthropy - attr_enthropy
        # if homogenous
        entropy_attr = calculate_entropy_attributel(d, data, attr)
        if entropy_attr != 0:
            gain_ratio[attr] = enthropy_map[attr] / entropy_attr

    if gain_ratio:
        if d.class_variable in gain_ratio:
            del gain_ratio[d.class_variable]
        if gain_ratio:
            max_attribute = max(gain_ratio, key=gain_ratio.get)
            if enthropy_map[max_attribute] > threshold:
                return max_attribute, alpha_attribute.get(max_attribute, 0), d.categorical_numerical[max_attribute] == 'numerical'

    return None, None, None

def gain_attr(d: Data, data: [], attribute):
    result = 0
    for domain in d.attribute_map[attribute]:
        attribute_index = d.attributes.index(attribute)
        filtered_data = list(filter(lambda x: x[attribute_index] == domain , data))
        result = result + ((len(filtered_data)/len(data)) * calculate_entropy(d, filtered_data))

    return result

def calculate_entropy(d: Data, data : []):
    result = 0

    map_class = {}
    for value in data:
        class_value = value[d.attributes.index(d.class_variable)]
        if class_value not in map_class:
            map_class[class_value] = 1
        else:
            map_class[class_value] += 1

    total_size = len(data)
    for value in map_class.values():
        result = result + (int(value)/total_size * math.log(int(value)/total_size, 2))
    return result * -1


def calculate_entropy_attributel(d: Data, data : [], attribute):
    result = 0

    map_class = {}
    for value in data:
        class_value = value[d.attributes.index(attribute)]
        if class_value not in map_class:
            map_class[class_value] = 1
        else:
            map_class[class_value] += 1

    total_size = len(data)
    for value in map_class.values():
        result = result + (int(value)/total_size * math.log(int(value)/total_size, 2))
    return result * -1

def calculate_entropy_numeric_attribute(d: Data, data : [], alpha, attribute):
    index_attribute = d.attributes.index(attribute)
    d_less = list(filter(lambda x: x[index_attribute] <= alpha, data))
    d_greater = list(filter(lambda x: x[index_attribute] > alpha, data))
    enthropy_d_less = calculate_entropy(d, d_less)
    enthropy_d_more = calculate_entropy(d, d_greater)
    result = len(d_less) / len(data) * enthropy_d_less + len(d_greater)/len(data)*enthropy_d_more
    return result



def find_best_split(d: Data, data: [], currAttr):
    # initialize data structures
    gain = {}
    total_entropy = calculate_entropy(d, data)
    alpha = {}
    domain_attributes = d.attribute_map[currAttr]
    domain_attributes = sorted(domain_attributes)

    for index, value in enumerate(domain_attributes):
        alpha[index] = domain_attributes[index] # DEBUG >> WHAT IS ALPHA
        entropy_value = calculate_entropy_numeric_attribute(d,data,value,currAttr)
        gain[index] = total_entropy - entropy_value

    best = max(gain, key=gain.get)
    return alpha[best]

        # class_variable_left_side =






if __name__ == '__main__':
    #
    filename = sys.argv[1]
    threshold = sys.argv[2]
    gain_ratio = sys.argv[3] == '1'
    try:
        restricted_file = sys.argv[4]
        file_reader = open(restricted_file, 'r')
        line = file_reader.readline()
        line = line.strip('\n')
        line = line.replace(' ', '')
        optional_file = line.split(',')
    except:
        optional_file = []
    omitted_attributes = []
    for index, value in enumerate(optional_file):
        if value == '0':
            omitted_attributes.append(index)
    data = Data(filename, omitted_attributes)
    import pdb; pdb.set_trace()
    head_node = Node('')
    algorithm_c45(data, data.data, data.attributes, float(threshold), head_node, gain_ratio)
    final_json = {'dataset': filename,
                  'node': head_node.__dict__}
    import json
    out_file = filename.replace('.csv','') + '-results.out'
    with open(out_file, 'w') as file:
        file.write(json.dumps(final_json, indent = 5))
    # print(json.dumps(final_json, indent=5))

