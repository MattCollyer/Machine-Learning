from preprocess import *
import numpy as np

###################################################################
# Read the data
#   - There are two data files, one negative and one positive
#   - Each line is a new sentence
#   - Each token is seperated by a space
#   - Write a function that will take in a positive file and a negative file,
#     and return a list of sentences (lists of tokens)
#            and a list of labels (1 for positive, 0 for negative)



if __name__ == "__main__":
    print("Part 1 Test")
    test_source, test_target = words_from_file("data/pos_sample.txt", "data/neg_sample.txt")
    print("Test source:", test_source[:3])
    print("Test target:", test_target[:3])
    print()

# Should print:

# Part 1 Test
# Test source: [['well', ',', 'another', 'popular', 'phrase', 'of', 'the', "90's", 'is', '"', 'all', 'good', 'things', 'must', 'come', 'to', 'an', 'end', ',', '"', 'and', 'this', 'stays', 'true', 'for', 'oscar', 'as', 'well', '.'], ['charles', 'walks', 'in', 'on', 'amy', 'and', 'oscar', 'having', 'a', 'drink', 'one', 'night', ',', 'as', 'oscar', 'and', 'amy', 'have', 'become', 'great', 'friends', ',', 'but', 'he', "doesn't", 'seem', 'to', 'mind', '.'], ['neve', 'is', 'delightful', 'as', 'her', 'conflicted', 'character', ',', 'who', 'feels', 'love', 'for', 'oscar', ',', 'but', 'knows', ',', 'based', 'on', 'rumors', ',', 'that', 'he', 'is', 'gay', '.']]
# Test target: [1, 1, 1] 

###################################################################
# Build the vocabulary
#  - the vocabulary should consist of every word that appears at least 3x in the examples
#  - each word should be associated with a unique integer id
#  - <oov> should be used for every word that only appears once and should have a single id


if __name__ == "__main__":
    print("Part 2 Test")
    test_vocab = build_vocabulary([['hi', 'there', '!'], ['what', 'are', 'you', 'doing', 'over', 'there', '?'], ['there', 'is', 'an', 'apple']])
    print("Example vocabulary:", test_vocab)
    
    data_vocab = build_vocabulary(test_source)
    
    print("Full dictionary length", len(data_vocab))
    print("ID for 'awesome'", data_vocab['awesome'])
    print()


# Should print: 

# Part 2 Test
# Example vocabulary: {'<oov>': 0, 'there': 1}
# Full dictionary length 5236
# ID for 'awesome' 1472


###################################################################
# Convert the array of strings to an array of ints
#  - Using the vocabulary dictionary, convert each token into the approprate key
#  - If the token is not in the dictionary, use <oov>


if __name__ == "__main__":
    print("Part 3 Test")
    
    test_vocab = {'<oov>': 0, 'there': 1}
    test_sentence_ints = sentence_to_ints(['hi', 'there', '!'], test_vocab)
    print("Test converted sentence:", test_sentence_ints)
    int_sentences = [sentence_to_ints(sentence, data_vocab) for sentence in test_source]

    print("Sample positive sentence:", test_source[3])
    print("... as ints:", int_sentences[3])
    print()


# Should print: 

# Part 3 Test
# Test converted sentence: [0, 1, 0]
# Sample positive sentence: ['the', 'bottom', 'line', ':', 'three', 'to', 'tango', 'is', 'a', 'light', ',', 'sharp', ',', 'snappy', 'romantic', 'comedy', 'with', 'a', 'superb', 'ending', ',', 'and', 'great', 'stars', '.']
# ... as ints: [7, 57, 58, 59, 60, 16, 0, 9, 33, 61, 2, 62, 2, 0, 63, 64, 65, 33, 66, 67, 2, 19, 39, 68, 26]

# ##################################################################
# Vectorize the sentences using Bag of Words
#  - Turn each sentence into a vector of V dimensions
#    where V is the size of the vocabulary dictionary
#  - Each element of the vector should be the count of the words in the sentence,
#    that correspond to that index

if __name__ == "__main__":
    print("Part 4 Test")
    
    test_vocab = {'<oov>': 0, 'there': 1}
    print("Test BoW:", bag_of_words_vector(test_sentence_ints, len(test_vocab)))
    
    bow_sentences = [bag_of_words_vector(sentence, len(data_vocab)) for sentence in int_sentences]

    print("Sample sentence as BoW:", bow_sentences[3][:25])
    print()

# Should print:

# Part 4 Test
# Test BoW: [2, 1]
# Sample sentence as BoW: [2, 0, 3, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]

###################################################################
# Put it all together
#  - Read the examples in from the file
#  - construct the vocabulary for the corpus
#  - convert the examples into lists of ints
#  - return the tuple (X, Y), where X is a numpy matrix of integers num_samples x vocab_size
#      and Y is a vector of 0 or 1s


if __name__ == "__main__":
    #I CHANGED THIS FUNCTION SINCE THIS TEST THIS WILL FAIL BUT ITS OK I SWEAR ITS OK.
    print("Part 5 Test")
    data = load_and_preprocess_data("data/pos_sample.txt", "data/neg_sample.txt")
    print("X:", data['X'])
    print("Y:", data['Y'])
    print()

# Should print:

# Part 5 Test
# X: [[ 0  2  2 ...  0  0  0]
# [ 0  0  2 ...  0  0  0]
# [ 2  0  4 ...  0  0  0]
# ...
# [12  0  3 ...  0  0  0]
# [ 3  0  1 ...  0  0  0]
# [ 1  0  1 ...  0  0  0]]
# Y: [1 1 1 ... 0 0 0]