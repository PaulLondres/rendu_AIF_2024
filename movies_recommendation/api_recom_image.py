from flask import Flask, jsonify, request
from annoy import AnnoyIndex

app = Flask(__name__)

@app.route('/recom_image', methods=['POST'])
def recom_image():
    #Load the annoy database
    dim_image = 576
    annoy_index_image = AnnoyIndex(dim_image, 'angular')
    annoy_index_image.load('annoy_base_image.ann')
    print("flag : api_enter")
    vector = request.json['vector'] # Get the vector from the request
    print("flag : api got json")
    nb_reco = request.json.get('nb_reco')
    closest_indices = annoy_index_image.get_nns_by_vector(vector[0], nb_reco) # Get the 2 closest elements indices
    print("flag : end of api")
    return jsonify(closest_indices) # Return the reco as a JSON

@app.route('/recom_txt', methods=['POST'])
def recom_txt():
    print("flag : api_enter")
    vector = request.json['vector'] # Get the vector from the request
    print("flag : api got json")
    method = request.json['method']
    nb_reco = request.json.get('nb_reco')
    #Load the annoy database of the selectionned method
    if method=='Bag of word':
        dim_text_bow = 20000
        annoy_base = AnnoyIndex(dim_text_bow,'euclidean')
        annoy_base.load('annoy_bow.ann')
    elif method=='Word2vec':
        dim_text_w2v = 300
        annoy_base = AnnoyIndex(dim_text_w2v,'euclidean')
        annoy_base.load('annoy_w2v.ann')
    # print(vector, type(vector))
    closest_indices = annoy_base.get_nns_by_vector(vector, nb_reco) # Get the 2 closest elements indices
    print("flag : end of api")
    return jsonify(closest_indices) # Return the reco as a JSON



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    
