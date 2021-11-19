import sys
import os
from collections import defaultdict
from nltk.stem import PorterStemmer
from math import log
pos = PorterStemmer()
# returns tokens found from string text
def tokenizer(text):
    tokens = text.split(' ')
    for index, token in enumerate(tokens):
        # import pdb; pdb.set_trace()
        tokens[index] = pos.stem(token)
    return tokens


# main function to iterate through each .txt file in the input directory
def textVectorizer(dirname):
    # list of lines to add to the groundtruth file
    ground_truth = []
    # set of words found to avoid duplicates
    words = set()

    for authorname in os.listdir(dirname):
        import pdb; pdb.set_trace()
        folderdir = ''.join((dirname, '/', authorname))
        for filename in os.listdir(folderdir):
            # building ground_truth list
            truth_line = ','.join((filename, authorname))
            ground_truth.append(truth_line)

            # list of tokens in current .txt file
            filetokens = []
            filedir = ''.join((folderdir, '/', filename))
            with open(filedir, 'r') as f:
                filetokens = tokenizer(f.read())
                #filetokens = list(f)
                #filetokens = f.read().strip().split(' ')
                print(filetokens)
                exit()
                # for line in f:
                #     filetokens = list()
                #     print(line)
                #     exit()

    # add ground truth list to a file and output the groundtruth.csv
    #print(len(ground_truth))
    with open ('groundtruth.csv', 'w') as f:
        for line in ground_truth:
            f.write(line)
            f.write('\n')

def textVectorizer2(dirname, stop_list):
    # list of lines to add to the groundtruth file
    ground_truth = []
    # set of words found to avoid duplicates
    words = set()
    documents = {}
    tf_documents = {}

    stop_word_file = sys.argv[2].split('/')[1].replace('.txt', '')
    tf_idf_out = sys.argv[1] + '_tf_idf_' + stop_word_file + '.csv'
    words_out = sys.argv[1] + '_words_' + stop_word_file + '.csv'


    for authorname in os.listdir(dirname):
        folderdir = ''.join((dirname, '/', authorname))
        for filename in os.listdir(folderdir):
            token_map = defaultdict(int)
            # building ground_truth list
            truth_line = ','.join((filename, authorname))
            ground_truth.append(truth_line)

            # list of tokens in current .txt file
            filetokens = []
            filedir = ''.join((folderdir, '/', filename))
            with open(filedir, 'r') as f:
                all_lines = f.read()
                all_lines = all_lines.replace('\n', ' ')
                all_lines = all_lines.replace("\'s", '')
                all_lines = "".join(list(map(lambda x: x if x.isalpha() or x == ' ' else '', all_lines)))

                #stem all lines
                filetokens = tokenizer(all_lines)
                #filetokens = list(f)
                for token in filetokens:
                    if token not in stop_list:
                        if token != '':
                            words.add(token)
                            token_map[token] += 1


            documents[filename] = token_map

    #documents -> 5000 documents
    #each document -> words in that document and occurence count within that document

    # import pdb; pdb.set_trace()
    total_documents = len(documents)
    documents_word = defaultdict(int)
    idf = {}

    for document, tokens in documents.items():
        #compute tf
        total_words = len(tokens)
        tf = {}
        for word in words:
            occurence_word_in_document = tokens[word] #raw frequency of word within certain document
            tf[word] = occurence_word_in_document / total_words
            if occurence_word_in_document != 0:
                documents_word[word] += 1 #document frequency of word
        tf_documents[document] = tf

    #compute idf and create a words file at the same time
    with open(words_out, 'w') as f:
        for word, occurence in documents_word.items():
            idf[word] = log(total_documents /  occurence)
            f.write(word + ',')

    #output tf*idf file
    with open(tf_idf_out, 'w') as f:
        #first line -> attributes name
        f.write('Document,')
        for index in range(len(words)):
            f.write('Attribute{0},'.format(index))
        f.write('\n')
        #second line -> each attribute's type
        f.write(str(len(tf_documents))) #how many documents there are
        for index in range(len(words)):
            f.write(',0') #everything is numerical so that's why every attribute is 0
        f.write('\n')
        f.write('Document') #class variable
        f.write('\n')
        for document, tf_words in tf_documents.items():
            f.write(document + ',')
            for word, tf_value in tf_words.items():
                idf_value = idf[word]
                f.write(str(tf_value * idf_value) + ',')
            f.write('\n')

    # import pdb; pdb.set_trace()



    # add ground truth list to a file and output the groundtruth.csv
    #print(len(ground_truth))
    with open (sys.argv[1] + '_GroundTruth.csv', 'w') as f:
        for line in ground_truth:
            f.write(line)
            f.write('\n')

def parse_stop_words(filename):
    with open(filename) as f:
        all_lines = f.read()
        return all_lines.split('\n')



if __name__ == '__main__':
    dirname = sys.argv[1]
    stopword = sys.argv[2]

    stop_list = parse_stop_words(stopword)
    stop_list = list(filter(lambda x: x != '', stop_list))
    # import pdb; pdb.set_trace()
    textVectorizer2(dirname, stop_list)