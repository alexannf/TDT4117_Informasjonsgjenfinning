#####################
## PRINT FUNCTIONS ##
#####################


# prints every element in a list, possible to define boundaries and a line-separator symbol
def print_all(list, separator=None, highest_i=None, lowest_i=None):
    lowest = 0
    highest = len(list)
    if lowest_i:
        lowest = lowest_i
    if highest_i:
        highest = highest_i
    for i in range(lowest, highest):
        print(list[i])
        if separator:
            print(separator)


# prints every element in a list, using the repr representation (whitespace visible)
def print_all_repr(list, separator=None, highest_i=None, lowest_i=None):
    lowest = 0
    highest = len(list)
    if lowest_i:
        lowest = lowest_i
    if highest_i:
        highest = highest_i
    for i in range(lowest, highest):
        print(repr(list[i]))
        if separator:
            print(separator)


# prints every (key, value)-tuple in a dictionary, possible to define boundaries
def dictionary_print(dict, highest_i=None, lowest_i=None):
    lowest = 0
    highest = len(dict)
    if lowest_i:
        lowest = lowest_i
    if highest_i:
        highest = highest_i
    for i in range(lowest, highest):
        print(str(i) + " " + str(dict.get(i)))


# topics is gensim related, this allows us to print the first "n" topics
def print_topics(n):
        for i in range(0, n):
            print("topic "+str(i+1)+": "+str(lsi_model.show_topic(i)))

# gives us a nicely formatted print for our already trimmed and sorted-by-best vector
def print_relevant_topics(relevant_vector):
    all_lsi_topics = lsi_model.show_topics()
    for pair in relevant_vector:
        print("[Topic ", pair[0], "]")
        print((all_lsi_topics[pair[0]][1]))

# formats, restricts and prints the length of the paragraphs in the list of (paragraph, key) tuples
def print_relevant_paragraphs(paragraph_tuples_list, line_restriction):
    all_text = ""
    for paragraph_tuple in paragraph_tuples_list:
        whitespace_number = 0
        paragraph, paragraph_number = paragraph_tuple[0], paragraph_tuple[1]+1
        paragraph_text = "[paragraph %s]" % paragraph_number
        paragraph_by_line = paragraph.split("\n")
        for line in paragraph_by_line:
            if(whitespace_number >= line_restriction+1):
                # break the loop before we print a line beyond the given restriction
                break
            else:
                # we need to put the whitespace character "back in", so we get a natural printing
                paragraph_text += line + "\n"
                whitespace_number += 1
        all_text += "\n" + paragraph_text
    print(all_text)



############
## TASK 1 ##
############

# 1.0
import random; random.seed(123)


# 1.1
import codecs
f = codecs.open("pg3300.txt", "r", "utf-8")


# helper function for next task
def file_to_list(file, split_phrase):
    lines = file.readlines()
    text = ""
    for line in lines:
        text += line
    # closing the file, for good measure
    file.close()
    list = text.split(split_phrase)
    return list

# 1.2 Partitioning file into list of lists of paragraphs (representing documents)
# splitting by "\n\r" (line-separator indicating its between paragraphs)
paragraphs = file_to_list(f, "\n\r")


# helper functions for next task
def remove_empty_paragraphs(par_list):
    filtered_list = []
    for paragraph in par_list:
        # handles empty strings and pure whitespace strings
        if paragraph:
            filtered_list.append(paragraph)
    return filtered_list

def filter_by_phrase(phrase, par_list):
    filtered_list = []
    for paragraph in par_list:
        if phrase.lower() not in paragraph.lower():
            filtered_list.append(paragraph)
    return filtered_list

# 1.3 filter out empty parentheses and all paragraphs containing "Gutenberg"
text_paragraphs = remove_empty_paragraphs(paragraphs)
filtered_paragraphs = filter_by_phrase("Gutenberg", text_paragraphs)


# helper function for the next task
def tokenize(par_list):
    tokenized_list = []
    for paragraph in par_list:
        # paragraph.split() is going to return a list for each paragraph. Appending it will make a list of lists
        tokenized_list.append(paragraph.lower().split())
    # a list of lists
    return tokenized_list

# 1.4 Tokenize paragraphs
tokenized_paragraphs = tokenize(filtered_paragraphs)


# 1.5, 1.6 & 1.7 clean, stem and count tokens (words) in our list of list of tokens
import string
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
stemmer = PorterStemmer()
freqDist = FreqDist()

def clean_and_stem(tokenized_par_list):
        cleaned_and_stemmed_list = []
        for paragraph in tokenized_par_list:
            cleaned_and_stemmed_paragraph = []
            for word in paragraph:
                # we don't want to check words consisting only of punctuation or whitespace
                if word.strip(string.punctuation + "\n\r\t"):
                    cleaned_and_stemmed_paragraph.append(
                        # 1.5 & 1.6, cleaning punctuation and whitespace from words, and stemming them
                        stemmer.stem(word.strip(string.punctuation + "\n\r\t")).lower())
                    # 1.7 adding 1 to the word-count of this word
                    freqDist[word] += 1
            if cleaned_and_stemmed_paragraph:
                cleaned_and_stemmed_list.append(cleaned_and_stemmed_paragraph)
        return cleaned_and_stemmed_list


# function call that finishes the preprocessing process
preprocessed_paragraphs = clean_and_stem(tokenized_paragraphs)


############
## TASK 2 ##
############

# 2.1 (1) building a dictionary creating integer-word mappings
import gensim
dictionary = gensim.corpora.Dictionary(preprocessed_paragraphs)

# only used for testing
dictionary_with_stopwords = gensim.corpora.Dictionary(preprocessed_paragraphs)


# helper functions for next task
def stem_stopwords(stopword_list):
    stemmed_stopwords = []
    for stopword in stopword_list:
        stemmed_stopwords.append(stemmer.stem(stopword).lower())
    return stemmed_stopwords

def remove_stopwords(stopword_list, dict):
    stopword_ids = []
    for stopword in stopword_list:
        # if the stopword exists in our dictionary
        if stopword in dict.values():
            # add the dictionary-id to our list of stopword-ids
            stopword_ids.append(dictionary.token2id[stopword])
    # filter tokens removes all ids in our stopword_ids list and reorganizes the dictionary if needed
    dict.filter_tokens(stopword_ids)

# 2.1 (2) filter stopwords using given stopwords from file "common-english-words.txt"
stopword_file = codecs.open("common-english-words.txt", "r", "utf-8")

# we can reuse function from 1.1 but split by "," instead. It only works because its only one line in the doc, but w/e
stopwords = file_to_list(stopword_file, ",")
stemmed_stopwords = stem_stopwords(stopwords)

# mutates the input dictionary (the one we defined above) to not contain stopwords
remove_stopwords(stemmed_stopwords, dictionary)


# helper function for the next task
def map_to_BOW(par_list, dict):
    paragraph_mappings_list = []
    for paragraph in par_list:
        paragraph_mappings_list.append(dict.doc2bow(paragraph))
    return paragraph_mappings_list

# 2.2 mapping our paragraphs in lists of preprocessed words to (dict-id, id-count) tuples (bag of words)
# the whole collection of bag of words is called a corpus
corpus = map_to_BOW(preprocessed_paragraphs, dictionary)



############
## TASK 3 ##
############

# 3.1 creating a TF-IDF model based on our corpus (transitively based on all the important parts of our document)
tfidf_model = gensim.models.TfidfModel(corpus)

# 3.2 converting our boring word counts in the bag of words to more meaningful TF-IDF weights, based on our tfidf-model
tfidf_corpus = tfidf_model[corpus]

# 3.3 creating a matrix that lets us calculate similarities between paragraphs and queries (with raw word count BOW)
similarity_matrix = gensim.similarities.MatrixSimilarity(corpus)

# 3.4 repeating steps in 3.1-3.3 using tfidf-corpus to create lsi-model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[tfidf_corpus]
lsi_similarity_matrix = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# 3.5
"""
 results after print_topics(3) (see bottom of the code for print statements)

 topic 1: [('labour', 0.15477563572840125), ('price', 0.14408495433082372), 
            ('produc', 0.13503229199222272), ('employ', 0.13483987253672042), 
            ('countri', 0.13079677263588554), ('capit', 0.126820964758121), 
            ('trade', 0.12440049710041262), ('tax', 0.11909823868057161), 
            ('land', 0.11813898491032215), ('upon', 0.1128775955699408)]
            
 topic 2: [('labour', -0.297056541235689), ('price', -0.25549054520054026), 
            ('trade', 0.21859512591378727), ('rent', -0.21233575452319642), 
            ('coloni', 0.19719495410936988), ('land', -0.1725887563414334), 
            ('foreign', 0.15962331626723392), ('duti', 0.15789054105745426), 
            ('quantiti', -0.15293957292273963), ('wage', -0.1528805702140827)]
            
 topic 3: [('silver', 0.3166707982011145), ('gold', 0.25744593878413335), 
            ('price', 0.24510725904707598), ('capit', -0.2242681866541281), 
            ('coin', 0.20626813426287297), ('employ', -0.17905716342117106), 
            ('stock', -0.16338669813510923), ('money', 0.16092246121593112), 
            ('revenu', -0.14612793779110148), ('profit', -0.14079089457638216)]

 Interpretation of the first 3 topics:
 
 I know that LSI is about analyzing the documents and the terms in the corpus and analyze frequency of pairs of words
 to produce a set of concepts, which is what i think we call topics here.
 
 I think topic 1 means that "document 1"/"paragraph 1" is most closely related to the topic 'labour', then 'price', 
 then 'produc' etc. 
 
 The paired values in the topic-tuples are their "relevance score" which says how well the document fits in to the
 given topic. The relevance score is actually an absolute value, even though we have represented negative numbers here
 we note that the biggest absolute values always come first.
 
"""


############
## TASK 4 ##
############

# helper function for the next task
def preprocessed(q):
    cleaned_and_stemmed_list = []
    words = q.split()
    for word in words:
        cleaned_and_stemmed_list.append(
            # cleans, stems and removes punctuation and whitespace from words
            stemmer.stem(word.strip(string.punctuation + "\n\r\t").lower()))
    return cleaned_and_stemmed_list

# 4.1 preprocessing the given query and turning it into a vector containing (word, count)-tuples
preprocessed_query = preprocessed("What is the function of money?")
preprocessed_query_test = preprocessed("How taxes influence Economics?")

# doc2bow will return single list (vector)
BOW_vector = dictionary.doc2bow(preprocessed_query)
BOW_vector_test = dictionary.doc2bow(preprocessed_query_test)


# helper function for the next task
def report_weights(query_vector):
    for pair in query_vector:
        weight = pair[1]
        word = dictionary.get(pair[0])
        print(word, ": ", "%0.2f" % weight)

# 4.2 giving our plain BOW values new weights according to our TF-IDF model and removing words not in dictionary
tfidf_vector = tfidf_model[BOW_vector]
tfidf_vector_test = tfidf_model[BOW_vector_test]

# report_weights(tfidf_vector)
# Prints this to console:
    # money :  0.32
    # function :  0.95

# report_weights(tfidf_vector_test)
# Prints this to console:
    # influenc :  0.52
    # tax :  0.26
    # econom :  0.82


# 4.3 Report top 3 most relevant paragraphs

# a matrix, or index, that makes it possible for us to compare documents to documents or queries to documents
tfidf_similarity_matrix = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# returns a tuple containing the paragraph and the "paragraph-number"
def get_most_relevant_paragraphs(q, matrix, num):
    docs2similarity = enumerate(matrix[q])
    # gives us a list of the "num" most relevant paragraphs, sorted by relevance
    sorted_paragraphs = sorted(docs2similarity, key=lambda kv: -kv[1])[:num]
    relevant_paragraph_ids = list(map(lambda t: t[0], sorted_paragraphs))
    relevant_paragraphs = []
    for key in relevant_paragraph_ids:
        paragraph_and_key = (filtered_paragraphs[key], key)
        relevant_paragraphs.append(paragraph_and_key)
    return relevant_paragraphs

# retrieving the most relevant paragraphs according to the TF-IDF based similarity matrix
tfidf_relevant = get_most_relevant_paragraphs(tfidf_vector, tfidf_similarity_matrix, 3)
itfidf_relevant_test = get_most_relevant_paragraphs(tfidf_vector_test, tfidf_similarity_matrix, 3)


# print_relevant_paragraphs(tfidf_relevant_test, 5)
# print_relevant_paragraphs(tfidf_relevant, 5)


# 4.4 giving our TF-IDF values new weights according to our LSI model

lsi_vector = lsi_model[tfidf_vector]
lsi_vector_test = lsi_model[tfidf_vector_test]

lsi_similarity_matrix = gensim.similarities.MatrixSimilarity(lsi_model[corpus])


def get_most_relevant_topics(vector, num):
    # gives us a list of the "num" most relevant paragraphs, sorted by relevance
    sorted_vectors = sorted(vector, key=lambda kv: -abs(kv[1]))[:num]
    return sorted_vectors

# retrieving the most relevant topic according to the LSI based similarity matrix
lsi_relevant = get_most_relevant_topics(lsi_vector, 3)
lsi_relevant_test = get_most_relevant_topics(lsi_vector_test, 3)

lsi_relevant_paragraphs = get_most_relevant_paragraphs(lsi_vector, lsi_similarity_matrix, 3)


##############
## PRINTING ##
##############


amount = 5

print("\n1.2] - first " + str(amount) + " paragraphs from the article, no whitespace adjustment:")
print("---------------------------------------------------------------------------------------------------------------")
print_all(paragraphs, None, amount)

print("\n\n1.2] - first " + str(amount) + " paragraphs from the article, with whitespace adjustment:")
print("---------------------------------------------------------------------------------------------------------------")
print_all(text_paragraphs, None, amount)

print("\n\n1.3] - first " + str(amount) + ' paragraphs from the article, "Gutenberg" removed:')
print("---------------------------------------------------------------------------------------------------------------")
print_all(filtered_paragraphs, None, amount)

print("\n\n1.4] - first " + str(amount) + ' paragraphs from the article, tokenized:')
print("---------------------------------------------------------------------------------------------------------------")
print_all(tokenized_paragraphs, None, amount)

print("\n\n1.5]-1.7] - first " + str(amount) + ' paragraphs from the article, after preprocessing is finished:')
print("---------------------------------------------------------------------------------------------------------------")
print_all(preprocessed_paragraphs, None, amount)

print("\n\n2.1] - first " + str(amount) + ' integer-word mappings in the dictionary created on our corpus:')
print("---------------------------------------------------------------------------------------------------------------")
dictionary_print(dictionary_with_stopwords, amount)

print("\n\n2.1] - first " + str(amount) + ' mappings in the dictionary with stopwords removed:')
print("---------------------------------------------------------------------------------------------------------------")
dictionary_print(dictionary, amount)

print("\n\n2.2] - BOW vectors from the first " + str(amount) + " paragraphs:")
print("---------------------------------------------------------------------------------------------------------------")
print_all(corpus, None, amount)

print("\n\n3.2] - TFIDF-weighted vectors from the first " + str(amount) + " paragraphs:")
print("---------------------------------------------------------------------------------------------------------------")
print_all(tfidf_corpus, None, amount)

print("\n\n3.4] - LSI-weighted vectors from the first " + str(amount) + " paragraphs:")
print("---------------------------------------------------------------------------------------------------------------")
print_all(lsi_corpus, None, amount)

amount_of_topics = 3
print("\n\n3.5] - displaying the %s first topics in our LSI model" % amount_of_topics)
print("---------------------------------------------------------------------------------------------------------------")
print_topics(amount_of_topics)

print("\n")
print("---------------------------------------------------------------------------------------------------------------")
query = "What is the function of money?"

print("4.1] - our given query:\t\t\t\t\t\t\t\t\t"+query)
print("4.1] - our given query after preprocessing:\t\t\t\t" + str(preprocessed_query))
print("4.1] - our given query transformed to BOW-form:\t\t\t" + str(BOW_vector))

print("\n\n4.2] - our given query transformed to BOW-form with TD-IDF weighting:\t\t" + str(tfidf_vector))
print("4.2] - alternative representation:")
report_weights(tfidf_vector)

print('\n\n4.3] - the 3 best matching (TF-IDF) paragraphs in our corpus from the query: "%s"' % query)
print_relevant_paragraphs(tfidf_relevant, 5)

print("\n\n4.3] - the 3 best matching (LSI) paragraphs in our corpus from the query: %s" % query)
print("Topic form:")
print_relevant_topics(lsi_relevant)
print("\n\n4.3] - top paragraphs given our LSI vector and our LSI corpus:")
print_relevant_paragraphs(lsi_relevant_paragraphs, 5)

























"""
# 4.4


# TESTKODE FOR Å SJEKKE AT DET FUNGERER LIKT SOM I OPPGAVEN
# test_query = "How taxes influence Economics?"
# test_query = preprocessing(test_query)
# test_query = dictionary.doc2bow(test_query)
# test_tfidf = tfidf_model[test_query]
# test_lsi = lsi[test_tfidf]
# sorted_test = (sorted(test_lsi, key=lambda kv: -abs(kv[1]))[:3] ) #[(3, 0.1236889420871775), (5, 0.08609030455385876), (9, 0.08523104132289301)]
# all_test_topics = (lsi.show_topics())
# for i in sorted_test:
#     print("[Topic ", i[0], "]")
#     print(all_test_topics[i[0]][1])
#printer følgende til konsoll, bekrefter at testquery gir samme output som i oppgavebeskrivelsen:
# [Topic  3 ]
# -0.467*"tax" + -0.201*"rent" + 0.193*"trade" + 0.166*"capit" + 0.154*"foreign" + 0.154*"employ" + -0.151*"upon" + 0.137*"quantiti" + 0.137*"labour" + 0.134*"manufactur"
# [Topic  5 ]
# 0.383*"tax" + 0.217*"capit" + 0.162*"foreign" + 0.137*"duti" + 0.133*"trade" + 0.130*"consumpt" + 0.127*"upon" + 0.126*"export" + 0.120*"profit" + 0.118*"home"
# [Topic  9 ]
# 0.314*"tax" + -0.258*"bank" + 0.206*"coloni" + -0.182*"land" + 0.181*"labour" + -0.156*"corn" + 0.154*"wage" + -0.149*"manufactur" + -0.145*"rent" + -0.140*"export"
# TESTKODE SLUTT




lsi_query = lsi[query_tfidf]
sorted_lsi = (sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3] )
all_topics = lsi.show_topics()
for i in sorted_lsi:
    print("[Topic ", i[0], "]")
    print((all_topics[i[0]][1]))

# Får følgende resultat skrevet til konsoll:
# [Topic  4 ]
# -0.262*"bank" + 0.251*"price" + -0.233*"capit" + -0.232*"circul" + -0.188*"gold" + -0.184*"money" + 0.181*"corn" + 0.141*"import" + -0.140*"coin" + -0.140*"revenu"
# [Topic  20 ]
# -0.305*"expens" + -0.215*"work" + 0.192*"interest" + -0.182*"bounti" + -0.167*"bank" + 0.156*"money" + -0.152*"coin" + 0.125*"stock" + -0.124*"mine" + 0.124*"revenu"
# [Topic  16 ]
# 0.309*"circul" + -0.219*"increas" + -0.199*"cent" + -0.194*"per" + -0.192*"coin" + 0.157*"mine" + 0.146*"money" + 0.145*"coloni" + -0.142*"industri" + 0.141*"materi"
"""