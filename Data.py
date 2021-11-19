from collections import defaultdict
class Data:
    def __init__(self,file,row_omited: list):
        self.row_omitted = row_omited
        self.attributes, self.different, self.class_variable, self.data, self.attribute_map, self.categorical_numerical = self.returnData(file)
        self.index_class_variable = self.attributes.index(self.class_variable)

    def returnData(self,file):
        header = []
        data = []
        different = []
        class_variable = ''
        attribute_map = defaultdict(list)
        categorical_numerical = {}
        fileReader = open(file, 'r')
        current_line = 0
        for line in fileReader.readlines():
            line = line.strip('\n')
            if current_line == 0:
                header = line.split(',')
                header.remove('')
            elif current_line == 1 :
                different = list(map(int,line.split(',')))
                for index, value in enumerate(different):
                    if value == -1:
                        self.row_omitted.append(index)
                # import pdb; pdb.set_trace()
                different = list(filter(lambda x: x != -1, different))
                new_header = []
                if self.row_omitted is not []:
                    for index, value in enumerate(header):
                        if index not in self.row_omitted:
                            new_header.append(value)
                    header = new_header
                #c
                for index, attribute in enumerate(header):
                    if different[index] == 0:
                        categorical_numerical[attribute] = 'numerical'
                    else:
                        categorical_numerical[attribute] = 'categorical'
            elif current_line == 2:
                class_variable = line
            else:
                split_data = line.split(',')
                split_data.remove('')
                if self.row_omitted is not []:
                    new_data = []
                    for index, value in enumerate(split_data):
                        if index not in self.row_omitted:
                            new_data.append(value)
                    split_data = new_data
                if split_data != [''] and split_data != []:
                    data.append(split_data)
                    for index, value in enumerate(split_data):
                        # import pdb; pdb.set_trace()
                        attribute = header[index]
                        if attribute not in attribute_map:
                            attribute_map[attribute].append(value)
                        else:
                            if value not in attribute_map[attribute]:
                                attribute_map[attribute].append(value)

            current_line += 1

        return header, different, class_variable, data, attribute_map, categorical_numerical
