# SOHAIL AHMED KHAN

from collections import Counter 
import re
import string
import sys


#Preprocessing corpus for unigrams and also getting the words and their frequencies using Counter
def get_unigram_counts(filepath):
    filepath = filepath
    with open(filepath) as fp:  
        sentences = []
        for line in (fp):
            #Adding <s> start and stop </s> in every sentence.
            line = '<s> ' + line + ' </s>'

            #Lower casing the sentences and splitting words over spaces
            words = line.lower().split()

            #Removing Punctuation
            sentence = [x for x in words if x not in (string.punctuation + '--' + '...' + '-' + '?')]
            sentences.extend(sentence)
    #Getting frequencies of words in Dictionary
    unigram_counts = Counter(sentences)
    return unigram_counts, len(sentences)


#Preprocessing the questions .txt file
def get_questions(Q_filepath):
    Q_filepath = Q_filepath
    with open(Q_filepath) as fp: 
        questions = []
        for line in (fp):
            #Seprating the candidate words at the end of each question
            words = line.replace('/', " ").lower().split()
            sentence = [x for x in words if x not in (string.punctuation + '--' + '...' + '-' + '?')]
            questions.append(sentence)
    return questions

#Preprocessing corpus for bigrams, removing punctuation and also getting the bigrams and their frequencies using Counter
def get_bigram_counts(filepath):
    filepath = filepath
    bigram = []
    with open(filepath) as fp:  
        sentences = []
        for line in (fp):
            line = '<s> ' + line + ' </s>'
            words = line.lower().split()
            sentence = [x for x in words if x not in (string.punctuation + '--' + '...' + '-' + '?')]
            sentences.extend(sentence)

        #Getting two consecutive words
        key2 = 1
        for i in range(len(sentences)-1):
            bigram.append(sentences[i] + " " + sentences[key2])
            key2 += 1
    #Getting the frequencies of Bigrams        
    bigram_counts = Counter(bigram)
    return bigram_counts, len(bigram)

#Method to get bigram predictions and their accuracies
def bigram(bigram_counts, total_bigram_count_in_corpus, unigram_counts, total_word_count_in_corpus, questions):

    last = []
    second_last = []
    list1 = []
    list2 = []
    bigram_probs = {}
    question_bigram_second_last = []
    question_bigram_last = []
    correct_words = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']
    
    #Getting the possible candidate words in two different lists
    for w in questions:
        last.append(w[len(w)-1])
        second_last.append(w[len(w)-2])

    #Getting the questions by appending the two candidate words alternatively in two different lists
    for i in range(len(questions)):
        strn = ''
        strn = ' '.join(questions[i][:-2]) 
        #Adding start <s> and stop </s> symbols in questions   
        list1.append('<s> ' + strn.replace('____', second_last[i]) + ' </s>')
        list2.append('<s> ' + strn.replace('____', last[i]) + ' </s>')

    #Calculating Bigram probabilities
    for words, freq in bigram_counts.items():
        word = words.split()
        bigram_probs[words] = freq / unigram_counts[word[0]]

    #Getting the possible sentence for one candidate word
    using_second_last_word = []
    for i in list1:
        key2 = 1
        words = i.split()
        for j in range(len(words)-1):
            using_second_last_word.append(words[j] + " " + words[key2])
            key2 += 1
        question_bigram_second_last.append(using_second_last_word)
        using_second_last_word = []

    #Calculating the probabilities for sentence with one possible candidate word
    prob_second_last = []
    for i in question_bigram_second_last:
        result = 1
        for j in i:
            if j in bigram_probs.keys():
                result *= bigram_probs[j]
            else:
                result = 0
        prob_second_last.append(result)

    #Getting the possible sentence for other candidate word
    using_last_word = []
    for i in list2:
        key2 = 1
        words = i.split()
        for j in range(len(words)-1):
            using_last_word.append(words[j] + " " + words[key2])
            key2 += 1
        question_bigram_last.append(using_last_word)
        using_last_word = []

    #Calculating the probabilities for sentence with other possible candidate word
    prob_last = []
    for i in question_bigram_last:
        result = 1
        for j in i:
            if j in bigram_probs.keys():
                result *= bigram_probs[j]
            else:
                result = 0
        prob_last.append(result)


    bigram_sentences = []
    accuracy = 0
    high_probs = []
    selected_word = []

    #Comparing the calculated probabilites and incrementing the accuracies for correctly predicted words
    for j in range(len(prob_last)):
        
        if prob_second_last[j] > prob_last[j]:
            bigram_sentences.append(list1[j])
            high_probs.append(prob_second_last[j])
            selected_word.append(second_last[j])
            if second_last[j] == correct_words[j]:
                accuracy += 1
                
        elif(prob_second_last[j] < prob_last[j]):
            bigram_sentences.append(list2[j])
            high_probs.append(prob_last[j])
            selected_word.append(last[j])
            
            if last[j] == correct_words[j]:
                accuracy += 1
        
        elif(prob_second_last[j] == prob_last[j] and prob_second_last[j] != 0.0 and prob_last[j] != 0.0):
            bigram_sentences.append(list1[j])
            high_probs.append(prob_second_last[j])
            selected_word.append(second_last[j])
            if second_last[j] == correct_words[j]:
                accuracy += 0.5
        
        else:
            bigram_sentences.append(list2[j])
            selected_word.append(last[j])
            high_probs.append(prob_last[j])

    #Printing the results for Bigram Model
    print("\n Results for BIGRAM Model\n")
    sentences = []
    for i in range(len(bigram_sentences)):
        bigram_sentences[i] = bigram_sentences[i].replace('<s> ', "")
        bigram_sentences[i] = bigram_sentences[i].replace(' </s>', ".")
        sentences.append(bigram_sentences[i])

    for i in range(len(sentences)):
        print([i+1], sentences[i], "---- Choosen Word: ", '"',selected_word[i],'"', " : Having Probability: ", high_probs[i])
    print("\n --------------------------------------------------------------------------- \n")
    print('Accuracy of Bigram Model: ', accuracy, ' out of ', len(bigram_sentences))


def bigram_with_smoothing(bigram_counts, total_bigram_count_in_corpus, unigram_counts, total_word_count_in_corpus, questions):
    
    last = []
    second_last = []
    list1 = []
    list2 = []
    bigram_probs_smooth = {}
    question_bigram_second_last = []
    question_bigram_last = []

    #Getting the possible candidate words in two different lists
    for w in questions:
        last.append(w[len(w)-1])
        second_last.append(w[len(w)-2])

    #Getting the questions by appending the two candidate words alternatively in two different lists
    for i in range(len(questions)):
        strn = ''
        strn = ' '.join(questions[i][:-2])    
        list1.append('<s> ' + strn.replace('____', second_last[i]) + ' </s>')
        list2.append('<s> ' + strn.replace('____', last[i]) + ' </s>')

    #Calculating Bigram probabilities and adding smoothing
    for words, freq in bigram_counts.items():
        word = words.split()
        bigram_probs_smooth[words] = (freq + 1 ) / (unigram_counts[word[0]] + len(unigram_counts))

    
    #Getting the possible sentence for one candidate word
    using_second_last_word = []
    for i in list1:
        key2 = 1
        words = i.split()
        for j in range(len(words)-1):
            using_second_last_word.append(words[j] + " " + words[key2])
            key2 += 1
        question_bigram_second_last.append(using_second_last_word)
        using_second_last_word = []

    #Calculating the probabilities for sentence with one possible candidate word
    prob_second_last = []
    for i in question_bigram_second_last:
        result = 1
        for j in i:
            if j in bigram_probs_smooth.keys():
                result *= bigram_probs_smooth[j]
            else:
                word = j.split()
                if word[0] in unigram_counts.keys():
                    result *= 1 / (unigram_counts[word[0]] + len(unigram_counts))
                else:
                    result *= 1 / len(unigram_counts)
        prob_second_last.append(result)

    
    #Getting the possible sentence for other candidate word
    using_last_word = []
    for i in list2:
        key2 = 1
        words = i.split()
        for j in range(len(words)-1):
            using_last_word.append(words[j] + " " + words[key2])
            key2 += 1
        question_bigram_last.append(using_last_word)
        using_last_word = []

    #Calculating the probabilities for sentence with other possible candidate word
    prob_last = []
    for i in question_bigram_last:
        result = 1
        for j in i:
            if j in bigram_probs_smooth.keys():
                result *= bigram_probs_smooth[j]
            else:
                word = j.split()
                if word[0] in unigram_counts.keys():
                    result *= 1 / (unigram_counts[word[0]] + len(unigram_counts))
                else:
                    result *= 1 / len(unigram_counts)
        prob_last.append(result)            

    bigram_sentences = []
    accuracy = 0
    high_probs = []
    selected_word = []
    correct_words = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']


    #Comparing the calculated probabilites and incrementing the accuracies for correctly predicted words
    for j in range(len(prob_last)):
        if prob_second_last[j] > prob_last[j]:
            bigram_sentences.append(list1[j])
            high_probs.append(prob_second_last[j])
            selected_word.append(second_last[j])
            if second_last[j] == correct_words[j]:
                accuracy += 1

        elif(prob_second_last[j] < prob_last[j]):
            bigram_sentences.append(list2[j])
            high_probs.append(prob_last[j])
            selected_word.append(last[j])
            if last[j] == correct_words[j]:
                accuracy += 1

        elif(prob_second_last[j] == prob_last[j] and prob_second_last[j] != 0.0 and prob_last[j] != 0.0):
            bigram_sentences.append(list1[j])
            high_probs.append(prob_second_last[j])
            selected_word.append(second_last[j])
            if second_last[j] == correct_words[j]:
                accuracy += 0.5

        else:
            bigram_sentences.append(list2[j])
            selected_word.append(last[j])
            high_probs.append(prob_last[j])


    #Printing the results for Bigram Model
    print("\n Results for BIGRAM with Smoothing Model\n")
    sentences = []
    for i in range(len(bigram_sentences)):
        bigram_sentences[i] = bigram_sentences[i].replace('<s> ', "")
        bigram_sentences[i] = bigram_sentences[i].replace(' </s>', ".")
        sentences.append(bigram_sentences[i])

    for i in range(len(sentences)):
        print([i+1], sentences[i], "---- Choosen Word: ", '"',selected_word[i],'"', " : Having Probability: ", high_probs[i])
    print("\n --------------------------------------------------------------------------- \n")
    print('Accuracy of Bigram with Smoothing Model: ', accuracy, ' out of ', len(bigram_sentences))



def unigrams(unigram_counts, total_word_count_in_corpus, questions):
    
    last = []
    secondlast = []
    unigram_counts['____'] = 1
    words = []
    probs = []
    accuracy = 0
    correct_words = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']

    #Reading questions and calculating probabilities for words present in questions using the unigram counts
    for w in questions:
        
        last = w[len(w)-1]
        second_last = w[len(w)-2]

        #Getting probability of  one candidate word
        prob_last = unigram_counts[last]/total_word_count_in_corpus

        #Getting probability of the other candidate word
        prob_second_last = unigram_counts[second_last]/total_word_count_in_corpus
        
        #Getting probabilities of question sentences with the two possible candidate words
        for i in range(len(w)-2):
            prob_last *= unigram_counts[w[i]]/total_word_count_in_corpus
            prob_second_last *= unigram_counts[w[i]]/total_word_count_in_corpus
        
        #Comparing the probabilities calculated above based on the two possible candidate words, incrementing accuracies
        #for correctly predicted words
        if prob_last > prob_second_last:
            probs.append(prob_last)
            words.append(last)
            if last in correct_words:
                accuracy += 1
            
        elif prob_second_last > prob_last:
            probs.append(prob_second_last)
            words.append(second_last)
            if second_last in correct_words:
                accuracy += 1
        
        elif prob_second_last == prob_last and prob_second_last != 0.0 and prob_last != 0.0:
            probs.append(prob_last)
            words.append(last)
            if second_last in correct_words:
                accuracy += .5
            
        elif prob_second_last == prob_last and prob_second_last == 0.0 and prob_last == 0.0:
            probs.append(prob_last)
            words.append(last)
    
    #Returning results to the main function
    return words, probs, accuracy


def main():
    #Getting file paths from the command line arguments
    filepath =  sys.argv[1]
    Q_filepath =  sys.argv[2]

    print("\n----------------- Working..." + "\t Please Wait... -----------------")

    #Getting Unigram Counts
    unigram_counts = []
    unigram_counts, total_word_count_in_corpus = get_unigram_counts(filepath)

    #Getting Bigram Counts
    bigram_counts = {}
    bigram_counts, total_bigram_count_in_corpus = get_bigram_counts(filepath)
    
    #Getting preprocessed Questions
    questions = get_questions(Q_filepath)

    #Getting results from the Unigram Method and printing the results
    words, probs, accuracy = unigrams(unigram_counts, total_word_count_in_corpus, questions)
    print("\n Results for UNIGRAM Model\n")
    for i in range(len(questions)):
        strn = ''
        strn = ' '.join(questions[i][:-2])    
        print([i+1],strn.replace('____', words[i]), '--- Choosen word: ',words[i],' having probability ', probs[i])
    print("\n --------------------------------------------------------------------------- \n")
    print('Accuracy of Unigram Model: ', accuracy, ' out of ', len(questions))
    print("\n --------------------------------------------------------------------------- \n")

    #Printing results from Bigram Method
    bigram(bigram_counts, total_bigram_count_in_corpus, unigram_counts, total_word_count_in_corpus, questions)
    print("\n --------------------------------------------------------------------------- \n")
    
    #Printing results from Bigram_with_smoothing Method
    bigram_with_smoothing(bigram_counts, total_bigram_count_in_corpus, unigram_counts, total_word_count_in_corpus, questions)
    print("\n --------------------------------------------------------------------------- \n")
    
if __name__ == '__main__':
    main()
