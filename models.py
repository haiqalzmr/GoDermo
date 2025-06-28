# models.py
# -*- coding: utf-8 -*-
"""
SQLite models for GoDermo app: Users + Prediction history
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255))
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    heatmap_path = db.Column(db.String(255))
    mobilenet_prob = db.Column(db.String(100))
    efficientnet_prob = db.Column(db.String(100))
    densenet_prob = db.Column(db.String(100))
    ensemble_prob = db.Column(db.String(100))
    report_path = db.Column(db.String(255))

def add_user(first_name, last_name, email, password, age=None, gender=None):
    user = User(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password=generate_password_hash(password),
        age=age,
        gender=gender
    )
    db.session.add(user)
    db.session.commit()
    return user

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

def validate_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user.password, password):
        return user
    return None

