from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2, base64, os, secrets
import numpy as np
from deepface import DeepFace
from datetime import datetime

# https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Get the current directory path
current_directory = os.path.dirname(os.path.abspath(__file__))

# Configure SQLite database
db_file_path = os.path.join(current_directory, "users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_file_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(6), nullable=True)
    check_in = db.Column(db.DateTime, default=datetime.now)
    check_out = db.Column(db.DateTime, default=datetime.now)
    image = db.Column(db.Text, nullable=False)


# Create database tables
with app.app_context():
    db.create_all()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def is_face(img):
    try:
        DeepFace.analyze(img, actions=["gender"])
        return True
    except ValueError as e:
        print("Face could not be detected:", e)
        return False


def check_face(image_data, username):
    try:
        user = User.query.filter_by(username=username).first()
        reference_img = user.image
        decoded_data = base64.b64decode(image_data.split(",")[1])
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        reference_img = base64.b64decode(reference_img.split(",")[1])
        nparr_ref = np.frombuffer(reference_img, np.uint8)
        ref_frame = cv2.imdecode(nparr_ref, cv2.IMREAD_COLOR)

        # Ensure both frames are in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)

        return DeepFace.verify(frame, ref_frame)["verified"]
    except ValueError as e:
        print("Face could not be detected:", e)
        return False


@app.route("/")
def index():
    if "user_id" in session:
        user = User.query.get(session["user_id"])

        # Decode the image data from base64
        img_data = user.image.split(",")[1]
        img_binary = base64.b64decode(img_data)

        # Create a data URI for the image
        img_uri = "data:image/jpg;base64," + base64.b64encode(img_binary).decode()

        return render_template(
            "index.html",
            username=user.username,
            gender=user.gender,
            image=img_uri,
            check_in=user.check_in,
            check_out=user.check_out,
        )
    else:
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        image_data = request.form["image"]
        username = request.form["username"]

        user = User.query.filter_by(username=username).first()
        if user:
            if is_face(image_data):
                if check_face(image_data, username):
                    session["user_id"] = user.id
                    return jsonify({"is_match": True})
                else:
                    return jsonify({"is_match": False})
            else:
                return jsonify({"no_face": True})
        else:
            return jsonify({"user_not_found": True})
    else:
        if "user_id" in session:
            print("User already logged in. Redirecting to index.")
            return redirect(url_for("index"))
        else:
            return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        image_data = request.form["image"]
        username = request.form["username"]
        check_in_str = request.form["check_in"]
        check_out_str = request.form["check_out"]

        # Convert date strings to Python date objects
        check_in = datetime.strptime(check_in_str, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_str, "%Y-%m-%d")

        if not User.query.filter_by(username=username).first():

            if is_face(image_data):
                new_user = User(
                    username=username,
                    image=image_data,
                    check_in=check_in,
                    check_out=check_out,
                )

                db.session.add(new_user)
                db.session.commit()

                decoded_data = base64.b64decode(image_data.split(",")[1])
                nparr = np.frombuffer(decoded_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                gen_inf = DeepFace.analyze(frame, actions=["gender"])
                if isinstance(gen_inf, list):
                    gen_inf = gen_inf[0]

                gender = gen_inf.get("dominant_gender")

                new_user.gender = gender
                db.session.commit()

                return jsonify({"is_available": True})
            else:
                return jsonify({"no_face": True})
        else:
            return jsonify({"is_available": False})
    else:
        return render_template("register.html")


@app.route("/check_face", methods=["POST"])
def check_face_route():
    if request.method == "POST":
        image_data = request.form["image"]
        username = request.form["username"]

        is_match = check_face(image_data, username)
        return jsonify({"is_match": is_match})


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(port=8000, debug=True)
