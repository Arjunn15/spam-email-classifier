import json

def load_users():
    with open("users.json", "r") as file:
        return json.load(file)

def authenticate(email, password):
    users = load_users()
    return email in users and users[email]["password"] == password

def get_user_name(email):
    users = load_users()
    return users[email]["name"]
