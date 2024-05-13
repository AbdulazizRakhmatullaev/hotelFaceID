from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2, base64, os, secrets
import numpy as np
from deepface import DeepFace
from datetime import datetime

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
    image = db.Column(db.Text, nullable=False)

    username = db.Column(db.String(30), nullable=False)
    first_name = db.Column(db.String(20), nullable=False)
    last_name = db.Column(db.String(25), nullable=False)
    dob = db.Column(db.DateTime)
    passport = db.Column(db.String(9), nullable=False)

    hotel_name = db.Column(db.String, nullable=False)
    room_num = db.Column(db.Integer, nullable=False)

    check_in = db.Column(db.DateTime)
    check_out = db.Column(db.DateTime)


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
            first_name=user.first_name,
            last_name=user.last_name,
            dob=user.dob,
            hotel_name=user.hotel_name,
            room_num=user.room_num,
            passport=user.passport,
            image=img_uri,
            check_in=user.check_in,
            check_out=user.check_out,
        )
    else:
        return redirect(url_for("admin"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        image_data = request.form["image"]
        username = request.form["username"]

        user = User.query.filter_by(username=username).first()
        if user:
            if is_face(image_data):
                if check_face(image_data, username):
                    if user.username == "admin":
                        session["admin"] = user.id
                    else:
                        session["user_id"] = user.id

                    return jsonify({"is_match": True})
                else:
                    return jsonify({"is_match": False})
            else:
                return jsonify({"no_face": True})
        else:
            return jsonify({"user_not_found": True})
    else:
        if ("user_id" in session) or ("admin" in session):
            print("User already logged in. Redirecting to index.")
            return redirect(url_for("index"))
        else:
            return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        image_data = request.form["image"]

        username = request.form["username"]
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        passport = request.form["passport"]

        hotel_name = request.form["hotel_name"]
        room_num = request.form["room_num"]

        dob_str = request.form["dob"]
        check_in_str = request.form["check_in"]
        check_out_str = request.form["check_out"]

        # Convert date strings to Python date objects
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        check_in = datetime.strptime(check_in_str, "%Y-%m-%dT%H:%M")
        check_out = datetime.strptime(check_out_str, "%Y-%m-%dT%H:%M")

        if not User.query.filter_by(username=username).first():
            if is_face(image_data):
                new_user = User(
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    passport=passport,
                    dob=dob,
                    hotel_name=hotel_name,
                    room_num=room_num,
                    image=image_data,
                    check_in=check_in,
                    check_out=check_out,
                )

                db.session.add(new_user)
                db.session.commit()

                return jsonify({"user_is_available": True})
            else:
                return jsonify({"no_face": True})
        else:
            return jsonify({"user_is_available": False})
    else:
        return render_template("register.html")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if "admin" in session:
        users = User.query.filter(User.username!="admin").all()

        return render_template("admin.html", users=users)
    else:
        return redirect(url_for("login"))


@app.route("/check_face", methods=["POST"])
def check_face_route():
    if request.method == "POST":
        image_data = request.form["image"]
        username = request.form["username"]

        is_match = check_face(image_data, username)
        return jsonify({"is_match": is_match})


@app.route("/logout", methods=["POST"])
def logout():

    if "user_id" in session:
        ses = "user_id"
    else:
        ses = "admin"

    session.pop(ses, None)

    return redirect(url_for("login"))

@app.route("/myaccount/delete", methods=["POST"])
def delete_my_acc():

    if request.method == "POST":
        username = request.form.get("username")
        user = User.query.filter_by(username=username).first()

        # delete the white boi
        db.session.delete(user)
        session.pop("user_id", None)

        db.session.commit()

        redirect(url_for("login"))
    return redirect(url_for("index"))

# for admins to delete accounts
@app.route("/admin/account/delete", methods=["POST"])
def delete_acc():

    if request.method == "POST":
        username = request.form.get("username")
        user = User.query.filter_by(username=username).first()

        # delete the white boi
        db.session.delete(user)
        db.session.commit()

        redirect(url_for("admin"))
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
