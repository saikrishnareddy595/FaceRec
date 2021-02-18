import face_recognition as fr
from flask import Flask, render_template, request , jsonify
import os
import numpy as np

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        #print(fnames)
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("./faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded
def classify_face(im):
    global faces
    #print(faces)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = im[:,:,::-1]
    face_locations = fr.face_locations(img)
    #print(face_locations)
    unknown_face_encodings = fr.face_encodings(img, face_locations)
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names

#flask app logic
app = Flask(__name__, template_folder='templates', static_folder='static_files')
app.secret_key="i am the secret key"
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

@app.route("/")
def get_data():
    global faces
    faces=get_encoded_faces()
    result={ i:k for i,k in enumerate(faces.keys()) }
    #print(result)
    return render_template('front_page.html',result=result)

@app.route("/learning",methods=["POST"])
def learning():
    print(request.files)
    image=request.files['testimage']
    if image:
        image=image.save('test.jpg')
        img = fr.load_image_file('./test.jpg')
        #print(img.shape)
        #print("in here",img)
        c_face = classify_face(img)
        result={'number_of_persons':len(c_face) ,'person_names': list(c_face)}
        return render_template('display_page.html',result=result)
    else:
        render_template("<h1>Enter Valid Image File <h1>")




@app.route("/about")
def about_page():
    return render_template('about_page.html')

if __name__=="__main__":
    app.run(debug=True)
