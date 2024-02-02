# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gradio as gr
import requests
import matplotlib.pyplot as plt
import os
import numpy as np 
import torchvision.transforms as transforms
import pickle
from model import model
from PIL import Image
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
import sklearn

def compute_mean_embeddings(s, model, words_list, dim=300):
    """
    Compute the mean embedding of the text by averaging the embedding for each word
    for a given model and word list
    """
    s = s.lower()
    emb_list = [model[w] for w in s if w in words_list]
    if emb_list != []:
      return np.mean(emb_list, axis=0)
    else:
      return np.zeros(dim)

#Read the bag of word vectorizer 
with open('vectorizer_bow.pkl', 'rb') as file:
    vectorizer_bow = pickle.load(file)

#Read the pickle file containing the description and title of movies
with open('movie_set.pkl', 'rb') as file:
    movie_set = pickle.load(file)

#Get image path list of every image
image_list = np.sort(os.listdir(os.path.join("MLP-20M", "MLP-20M")))
image_path_list  = np.array([os.path.join("MLP-20M", "MLP-20M", i) for i in image_list])
#Define normalization transformation of images
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])
##Extract the word2vec model outisde the function in order to avoid to do it for every request
#Define the path of the glove file 
glove_file = ('glove.6B.300d.txt')
#Convert the glove embeddings in word2vec word2vec in a temporary file
word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
#Load the Word2vec model
model_word2vec = KeyedVectors.load_word2vec_format(word2vec_glove_file)
#Retrieve the list of words in our Word2vec model
words_list_word2vec = model_word2vec.index_to_key



def find_similar_movies_image(input_image, nombre_recommendations):
    #If a number of recommendation is provided, convert it from str to int
    if nombre_recommendations:
        nb_reco = int(nombre_recommendations)
    #Otherwise, set the default number of reco by 5
    else:
        nb_reco = 5
    # Use the pre-trained network to extract features from the input image
    image = Image.fromarray(input_image)
    tensor = transform(image)
    features = model(tensor.unsqueeze(0))
    # Convert the features to list to pass it to the api via a json format
    print("flag1 model")
    vector = features.tolist()
    print("flag vector")
    response = requests.post("http://annoy-db:5000/recom_image", json={'vector': vector, 'nb_reco': nb_reco})
    print("flag postrep")
    if response.status_code == 200:
        #Get the indexes from the api response
        indices = response.json()
        print("flag ok200")
        # Retrieve paths for the indices
        paths = image_path_list[indices]
        print("flag dfdf")
        # Plot the images
        fig, axs = plt.subplots((nb_reco-1)//4 + 1, 4, figsize=(20, 6*((nb_reco-1)//4 + 1)), squeeze = False)
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i//4, i%4].imshow(img)
            axs[i//4, i%4].axis('off')
        #Remove empty plots
        for i in range(nb_reco, 4 * ((nb_reco-1)//4 + 1)):
            axs[i // 4, i % 4].axis('off')
        print("flag pre ret fig")
        return fig
    else:
        print("flag error")
        return "Error in API request"
    
def find_similar_movies_text(input_text, nombre_recommendations, methode_embeddings):
    #If a number of recommendation is provided, convert it from str to int
    if nombre_recommendations:
        nb_reco = int(nombre_recommendations)
    #Otherwise, set the default number of reco by 5
    else:
        nb_reco = 5
    print("flag1 model")
    #Compute the embeddings of our text for the selectionned images
    if methode_embeddings=='Bag of word':
        desc_embeddings = vectorizer_bow.transform([input_text])
        desc_vect = desc_embeddings.toarray().flatten()
        
    elif methode_embeddings=='Word2vec':
        #Compute the mean embeddings of the text
        desc_vect = compute_mean_embeddings(input_text, model_word2vec, words_list_word2vec)
    print("flag embedd calculated")  
    #Send the variables to the api
    response = requests.post("http://annoy-db:5000/recom_txt", 
                             json={'vector' : desc_vect.tolist(), 
                                   'method' : methode_embeddings, 'nb_reco' : nb_reco})
    print("flag postrep")
    if response.status_code == 200:
        #Get the indexes from the api
        indices = response.json()
        print("flag ok200")
        title_date = ''
        for i,j in enumerate(indices) :
            title_date+= str((f'Recommendation n°{i+1} : '+movie_set.iloc[j]["title"]+
                              '\n'+'Date : '+ movie_set.iloc[j]["release_date"]+' \n'+ 
                              'Synopsis : '+movie_set.iloc[j]["overview"]+' \n\n'))
        print('flag got movies descs')
        #Return the whole string and remove the last 2 lines break
        return title_date[:-4]
    else:
        print("flag error")
        return "Error in API request"
    


iface_im = gr.Interface(
    fn=find_similar_movies_image,
    inputs=["image", gr.Textbox(type="text", label="Nombre de recommendations souhaitées")],
    outputs="plot",
    title="Système de recommendation de films",
    description="Upload a movie poster image to find similar movies based on their posters."
)

iface_txt = gr.Interface(
    fn=find_similar_movies_text,
    inputs=["text", gr.Textbox(type="text", label="Nombre de recommendations souhaitées"),
            gr.Radio(["Bag of word", "Word2vec"], label="Sélectionnez la méthode")],
    outputs="text",
    title="Système de recommendation de films",
    description="Upload a movie poster image to find similar movies based on their posters."
)


demo = gr.TabbedInterface([iface_im, iface_txt], ["Image", "Texte"])
demo.launch(server_name="0.0.0.0")




