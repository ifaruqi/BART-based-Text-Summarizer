#!/usr/bin/env python
# coding: utf-8

# <h1><center>Text Summarization Python Code Using PDF File As An Input

# Note: Please execute this .ipynb file on ***google colab*** so that the text summarization interface included at the end of this file works best.

# The code below explain the steps to perform text summarization with PDF file as an input. The user will be required to upload the file and then pick the summarization method (Extractive or Abstractive).
# 
# The code will include the following functionalities:
# * Taking PDF file as an input
# * Performing data cleaning 
# * Summarizing the text content
# * Simple web interface for the user to try on

# ***

# # Section 1 - Packages and Dependencies

# Note: All packages and dependencies used in this code file are explained in the Report

# Capture and quiet are added in every chunk to prevent warning message to appear. This might not be the best practice, as the user may not be able to see warning or error message. It is done for the purpose of tidyness.

# ### Installing Required Packages

# In[19]:


get_ipython().run_cell_magic('capture', '', 'pip install PyDictionary --quiet\n')


# In[10]:


get_ipython().run_cell_magic('capture', '', 'pip install summa --quiet\n')


# In[3]:


get_ipython().run_cell_magic('capture', '', 'pip install spacy --quiet\n')


# In[4]:


get_ipython().run_cell_magic('capture', '', 'pip install transformers --quiet\n')


# In[5]:


get_ipython().run_cell_magic('capture', '', 'pip install torch --quiet\n')


# In[6]:


get_ipython().run_cell_magic('capture', '', 'pip install sentencepiece --quiet\n')


# In[7]:


get_ipython().run_cell_magic('capture', '', 'pip install pdfplumber --quiet\n')


# In[12]:


get_ipython().run_cell_magic('capture', '', 'pip install ipywidgets\n')


# In[14]:


get_ipython().run_cell_magic('capture', '', 'pip install Pillow==9.0.0 --quiet\n')


# In[23]:


get_ipython().run_cell_magic('capture', '', 'pip install nltk\n')


# ### Loading Packages

# In[24]:


# for string manipulation and regular expressions
import re
import string

# Natural Language Processing libraries and modules
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
import spacy
import heapq
from summa import summarizer

# interactive displays and user interface components
from IPython.display import display, HTML
import ipywidgets as widgets

# hugging face transformers library for nlp and deep learning models
from transformers import pipeline
from transformers import AutoTokenizer

# library to import a pdf file
import pdfplumber
import logging


# In[25]:


get_ipython().run_cell_magic('capture', '', '!python -m spacy download en_core_web_sm\n')


# In[27]:


get_ipython().run_cell_magic('capture', '', "nlp = spacy.load('en_core_web_sm')\nnltk.download('stopwords')\nnltk.download('punkt')\nnltk.download('wordnet')\nnltk.download('omw-1.4')\n")


# In[28]:


# Load the transformer and tokenizer for Abstractive Text Summarization
from transformers import pipeline
from transformers import AutoTokenizer


# ***

# # Section 2 - Functions for Extractive and Abstractive Summarization

# ## Section 2.1: Extractive Summarization

# The below function i.e. ***extractive_pdf()*** performs extractive text summarization by assigning scores to sentences based on the frequencies of non-stopwords in the text and selecting the top-scoring sentences as the summary.
# Please refer to the [link](https://github.com/Amey-Thakur/TEXT-SUMMARIZER/blob/main/nltk_summarization.py) for the motivation of the function below.

# 
# 
# *   Modify ***line # 37*** in the below code to change the number of sentences in the extractive summary.
# *   We used the logic of using 30 as a cutoff in the code is to limit the length of the sentences that are considered for scoring. If a sentence exceeds 30 words, it will be excluded from the scoring process and will not contribute to the final summary; the reason for using this cutoff is to prioritize shorter sentences, as longer sentences may contain more information and be less indicative of the main points of the text. By focusing on shorter sentences, the code aims to extract concise and important information for the summary.
# Consider modifying ***line # 29*** if you wish to change the cut-off
# 
# 

# In[29]:


def extractive_pdf(cleaned_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(cleaned_text):
        if word not in stopWords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_list = nltk.sent_tokenize(cleaned_text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


# ## Section 2.2: Abstractive Summarization

# The below function i.e. ***abstractive_pdf()*** performs abstractive text summarization by using BART (Bidirectional and AutoRegressive Transformers) architecture and has been pretrained on a large corpus of text data.
# 
# In this project, DistilBART model is used. The model is pre-trained with CNN Daily News data. Sources: __[DistilBART](https://huggingface.co/sshleifer/distilbart-cnn-12-6)__

# In[45]:


def abstractive_pdf(all_text):
    import warnings
    warnings.filterwarnings('ignore')

    import logging
    # Set the logging level to suppress the warning
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Initialize the model for summarization
    summarization = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    tokens = tokenizer.tokenize(all_text)
    
    # If the tokens exceed the maximum limit for BART model (1024 tokens), split the text into smaller chunks
    max_length = 1000  # Maximum number of tokens in each chunk
    tokenized_text = tokenizer.encode(all_text, add_special_tokens=False)
    text_chunks = []
    current_chunk = []
    
    # Split the text into chunks
    for token in tokenized_text:
        current_chunk.append(token)
        if len(current_chunk) >= max_length:
            chunk = tokenizer.decode(current_chunk)
            text_chunks.append(chunk.strip())
            current_chunk = []
    
    # Add the remaining tokens as the last chunk
    if current_chunk:
        chunk = tokenizer.decode(current_chunk)
        text_chunks.append(chunk.strip())
    
    # Process each chunk and generate summaries
    chunk_summaries = []
    for chunk in text_chunks:
        summary = summarization(chunk, max_length=300, min_length=30, num_beams=8, do_sample=True)[0]["summary_text"]
        chunk_summaries.append(summary.strip())
    
    # Merge the summaries into a single text
    merged_summary = " ".join(chunk_summaries)

    # Generate the final summary
    if len(merged_summary) > 1000:
        final_summary = summarization(merged_summary, max_length=len(merged_summary), min_length=300, num_beams=8, do_sample=True)[0]["summary_text"]
    else:
        final_summary = merged_summary
        
    return final_summary


# ***

# ## Section 3: Input PDF and Text Cleaning

# ### Extracting Text from A PDF File

# In[92]:


def input_pdf(file_name):
    # Read pdf file and extract the text
    with pdfplumber.open(file_name) as pdf:
        all_text = ''
        for page in pdf.pages:
            text = page.extract_text()
            all_text += text
            
    return all_text


# ### Cleaning Extracted Text from A PDF File

# In[177]:


def text_cleaning(all_text):
    
    reference_pattern = r"(?im)^\s*\d+\.\s*references|bibliography|reference|\nReferences\b"

    matches = re.findall(reference_pattern, all_text)
    if matches:
        last_match = matches[-1]
        start_index = all_text.rfind(last_match)
        clean_text = all_text[:start_index]
    else:
        clean_text = all_text
        
    
    # Step 1: Remove numbers in brackets and at the end
    clean_text = re.sub(r'\[\d+\]|\s\d+$', ' ', clean_text)

    # Step 2: Remove HTML links
    clean_text = re.sub(r'\b(?:https?://|www\.)\S+\b', '', clean_text)

    # Step 3: Remove email addresses
    clean_text = re.sub(r'\S+@\S+', '', clean_text)

    # Step 4: Regular expression pattern to identify headers, footers, and page numbers
    header_footer_pattern = r'^\s*\d+\s*|\s*\d+\s*$|^\s*\w+\s*|\s*\w+\s*$'
    clean_text = re.sub(header_footer_pattern, '', clean_text, flags=re.MULTILINE)

    # Step 5: Perform additional text cleaning (e.g., remove extra spaces)
    clean_text = ' '.join(clean_text.split())

    # Step 6: Identify and remove specific patterns or content that are irrelevant to summarization
    clean_text = re.sub(r'\b(?:advertisement|footnote)\b', '', clean_text, flags=re.IGNORECASE)
    
        
    # Step 8: Remove metadata of an academic paper
    metadata_patterns = [
        r"Journal of [A-Za-z\s]+",  # Example pattern for journal names
        r"Volume [0-9]+",  # Example pattern for volume numbers
        r"Issue [0-9]+",  # Example pattern for issue numbers
        r"Author: [A-Za-z\s]+",  # Example pattern for author names
        r"Keywords: [A-Za-z\s]+"# Add more patterns as needed
    ]

    for pattern in metadata_patterns:
        clean_text = re.sub(pattern, "", clean_text)
        
    
    return clean_text
    


# ### Taking a PDF Input from Users

# For convenience, it is recommended to put the file in the same directory as this .ipynb file. Then, when prompted to provide the file path, the user can then write : ./filename.pdf. The '.' indicate the current directory, and the rest indicate that the file is in the directory.

# In[187]:


# Ask the user to provide the file path
file_path = input("Enter the path to the PDF file: ")

# Extract all the text from the pdf file
all_text = input_pdf(file_path)

# Clean the extracted text
clean_text = text_cleaning(all_text)


# ***

# ## Section 4: Web Interface for Text Summarization

# ### Web Interface

# The code below handles form submissions and performs text summarization based on the selected type. It cleans the input text by removing numbers, special characters, and extra spaces. Depending on the summarization type chosen ('Extractive' or 'Abstractive'), it generates the corresponding type of summary using the 'cleaned text'. The resulting summarized text is displayed in HTML format along with a label indicating the type of summarization.

# In[33]:


def handle_form_submission(form_data):
    summarization_type = form_data.get('summarization_type', '')

    # Perform summarization based on the selected type
    if summarization_type == 'Extractive':
        summarized_text = extractive_pdf(clean_text)
        summarization_label = 'Extractive Summarized Text:'
    else:
        summarized_text = abstractive_pdf(clean_text)
        summarization_label = 'Abstractive Summarized Text:'

    summary_html = f'''
    <h3>{summarization_label}</h3>
    <p>{summarized_text}</p>
    '''

    display(HTML(summary_html))


# The below code sets up a user interface for a text summarization application. It includes a submit button, input text area, and a dropdown menu to select the type of summarization. When the submit button is clicked, the input text and selected summarization type are captured, and the handle_form_submission function is called with the form data. The code also displays a heading and the UI elements for input and selection.
# 
# Instruction to use the interface:
# 
# **Step 1:** Make sure you already upload the pdf file you want to extract. This can be done by running the previous chunk of code (Taking a PDF Input from A User).
# 
# **Step 2:** Select the type of summarization: Choose the summarization type from the dropdown menu labeled "Summarization Type". You have two options to choose from: "Extractive" or "Abstractive". Select the type that best suits your requirements or preferences.
# 
# **Step 3:** Click the submit button: This will trigger the summarization process. The system will generate a summary based on your input and selected type, and it will be displayed below in HTML format along with a label indicating the type of summarization.

# In[195]:


def on_submit_button_clicked(b):
    form_data = {
        'summarization_type': summarization_type.value,
        'all_text': all_text
    }
    handle_form_submission(form_data)


submit_button = widgets.Button(description='Submit')
submit_button.on_click(on_submit_button_clicked)

summarization_type = widgets.Dropdown(description='Summarization Type:',
                                      options=['Extractive', 'Abstractive'])

# Display the heading
heading = widgets.HTML("<h1>PDF Summarizer Group 16</h1>")
display(heading)
display(summarization_type)
display(submit_button)


# In[190]:


all_text


# In[191]:


clean_text


# <h1><center>End of The Code
