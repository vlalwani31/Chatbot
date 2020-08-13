import os
import yaml
import numpy as np
from keras import layers, activations, models, preprocessing, utils


# Tokenize Sentences
def tokenize_sentences(list_of_sentences):
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(list_of_sentences)
    return tokenizer.texts_to_sequences(list_of_sentences), tokenizer.word_index

# Get Maximum length out of list of sequences
def max_length_token_list(list_of_token):
    length_list = []
    for token_seq in list_of_token:
        length_list.append(len(token_seq))
    return np.array(length_list).max()


# Collect all the Question and Answers
dir_path = './NLP_data/simple_chatbot_data'
files_list = os.listdir(dir_path + os.sep)

questions = []
answers = []
for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if (len(con) > 2):
            questions.append(con[0])
            replies = con[1:]
            ans = ''
            for reply in replies:
                ans += ' ' + reply
            answers.append(ans)
        elif (len(con) > 1):
            questions.append(con[0])
            answers.append(con[1])

# Tokenize all the Questions
tokenized_questions, question_dict = tokenize_sentences(questions)

# Get Maximum length of the Question
question_max_length = max_length_token_list(tokenized_questions)
print('Max Question Length: ', question_max_length)

# Pad Questions
padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=question_max_length, padding='post')
encoder_input_data = np.array(padded_questions)

num_question_tokens = len(question_dict) + 1
print('Number of question tokens: ', num_question_tokens)


# Pad Answers with <start> and <end>
Answer_lines = []
for line in answers:
    Answer_lines.append('<START> ' + line + ' <END>')
# Tokenize all the Answers
tokenized_answers, answer_dict = tokenize_sentences(Answer_lines)

# Get Maximum length of the Answers
answer_max_length = max_length_token_list(tokenized_answers)
print('Max Answer Length: ', answer_max_length)

# Pad Answers
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=answer_max_length, padding='post')
decoder_input_data = np.array(padded_answers)

num_answer_tokens = len(answer_dict) + 1
print('Number of answer tokens: ', num_answer_tokens)

# Removing <START> and turning input into one hot vector
decoder_target_data = []
for token_seq in tokenized_answers:
    decoder_target_data.append(token_seq[1:])
padded_answers = preprocessing.sequence.pad_sequences(decoder_target_data, maxlen=answer_max_length, padding='post')
one_hot_answers = utils.to_categorical(padded_answers, num_answer_tokens)
decoder_target_data = np.array(one_hot_answers)
