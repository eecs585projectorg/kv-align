import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')  
nltk.download('words')  
nltk.download('punkt_tab')
nltk.download('wordnet')

stemmer = SnowballStemmer("english")

def count_valid_words(tokens):
    valid_words = set(words.words())
    valid_word_count = sum(1 for token in tokens if token.lower() in valid_words or stemmer.stem(token.lower()) in valid_words or token in [".", ",", "!", "?"])
    return valid_word_count

def get_data_plot(mode, model_name, limit_2gram):
    with open(f'final_evaluations/readability_data/{model_name}_{mode}_10000_cache_64_limit_{limit_2gram}.txt', 'r') as file:
        text = file.read()

    print(f"mode {mode}, model_name {model_name}, limit_2gram {limit_2gram}")
    tokens = word_tokenize(text)

    print("num tokens:", len(tokens) // 10 * 10)

    is_readable = []

    bad_num = 0
    for i in range(0, len(tokens) // 10 * 10, 10):
        count = count_valid_words(tokens[i:i+10])
        if count < 5:
            bad_num += 1
            is_readable.append(False)
        else:
            is_readable.append(True)

    print("Number of Unreadable Chunks", bad_num)

    np.save(f"final_evaluations/readability_data/is_readable_model_{model_name}_mode_{mode}_limit_2gram_{limit_2gram}.npy", np.array(is_readable))

    # Number of elements in the array
    N = len(is_readable)

    # Create the x positions for the bar segments (just use an array from 0 to N)
    x_positions = np.arange(N)

    # Create the bar plot
    _, ax = plt.subplots(figsize=(10, 2))  # Wide but thin plot

    # Loop over the array and create the segments for each value (True/False)
    for i in range(N):
        color = 'green' if is_readable[i] else 'red'  # Choose color based on True/False
        ax.bar(x_positions[i], 1, width=1, color=color)

    # Adjust the plot limits to make the bar look continuous
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)

    # Hide the y-axis and ticks
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks_position('none')

    ax.set_title(f'Readability of model {model_name} with mode {mode} and limit_2gram {limit_2gram}')

    plt.savefig(f"final_evaluations/readability_data/readability_plot_model_{model_name}_mode_{mode}_limit_2gram_{limit_2gram}.png")
    plt.show()

for mode in ["kv", "sw"]:
    for model_name in ["gpt2", "rope"]:
        for limit_2gram in [True, False]:
            get_data_plot(mode, model_name, limit_2gram)
