# auth.py
# -*- coding: utf-8 -*-
"""
Flask routes for signup, login, logout.
"""

from flask import (Blueprint, flash, redirect, render_template, request,
                   session, url_for)

from models import add_user, get_user_by_email, validate_user

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = validate_user(email, password)
        if user:
            session['user_id'] = user.id
            session['user_name'] = user.first_name
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))  # Redirect to main page after login
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        age = request.form.get('age')
        gender = request.form.get('gender')
        if get_user_by_email(email):
            flash('Email already registered.', 'warning')
        else:
            add_user(first_name, last_name, email, password, age, gender)
            flash('Account created. Please login.', 'success')
            return redirect(url_for('auth.login'))
    return render_template('signup.html')

@auth.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('auth.login'))
