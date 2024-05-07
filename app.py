from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import base64
import numpy as np
import secrets
from deepface import DeepFace
from datetime import datetime

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///userfaceid.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    check_in = db.Column(db.DateTime, default=datetime.now)
    check_out = db.Column(db.DateTime, default=datetime.now)
    image = db.Column(db.Text, nullable=False)


# Create database tables
with app.app_context():
    db.create_all()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def check_face(image_data, username):
    user = User.query.filter_by(username=username).first()
    if user:
        reference_img = user.image
        decoded_data = base64.b64decode(image_data.split(",")[1])
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        reference_img = base64.b64decode(reference_img.split(",")[1])
        nparr_ref = np.frombuffer(reference_img, np.uint8)
        ref_frame = cv2.imdecode(nparr_ref, cv2.IMREAD_COLOR)

        print("Captured Frame Shape:", frame.shape)
        print("Reference Frame Shape:", ref_frame.shape)

        # Ensure both frames are in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)

        print("Captured Frame Shape:", frame.shape)
        print("Reference Frame Shape:", ref_frame.shape)

        verification_result = DeepFace.verify(frame, ref_frame)
        print(verification_result)
        return verification_result["verified"]
    else:
        print("No users found in the database.")
        return False


@app.route("/")
def index():
    if "user_id" in session:
        user = User.query.get(session["user_id"])
        return render_template(
            "index.html",
            username=user.username,
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

        if check_face(image_data, username):
            user = User.query.filter_by(username=username).first()
            session["user_id"] = user.id  # dummy user
            return jsonify({"is_match": True})
        else:
            return jsonify({"is_match": False})
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

            new_user = User(
                username=username,
                image=image_data,
                check_in=check_in,
                check_out=check_out,
            )
            db.session.add(new_user)
            db.session.commit()

            # Save image data to a file for inspection
            with open("registered_image.jpg", "wb") as f:
                f.write(base64.b64decode(image_data.split(",")[1]))

            return jsonify({"is_available": True})

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
