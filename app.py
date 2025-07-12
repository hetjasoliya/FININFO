from flask import Flask
from flask import render_template
from flask_wtf import FlaskForm
from flask_wtf import CSRFProtect
from flask_modals import Modal
from wtforms import DateField, StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
from flask import redirect, url_for
from flask import request
from flask_bootstrap import Bootstrap5
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from flask import session
import io
import base64
from pretty_html_table import build_table
from openai import OpenAI
from os import getenv
from pydantic import BaseModel
from typing import List
import asyncio
from flask import Response



pd.set_option('display.precision', 2)
app = Flask(__name__)
app.secret_key = 'just_a_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Maninagar%403257@localhost/fininfo'
db = SQLAlchemy(app)
CSRFProtect(app)
Bootstrap5(app)
modal = Modal(app)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY", "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"),
)


def stream_chat_response(prompt, instructions1):
    
    try:
        
        response = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "system", "content": instructions1}
            ],
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {content}\n\n"
                
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"

class chatbot_form(FlaskForm):
    prompt = StringField('Prompt', validators=[DataRequired()],)
    submit = SubmitField('Submit')

def save_data(symbol, startdate, enddate, interval1):
    ticker = yf.Ticker(symbol)
    df2 = ticker.history(start=startdate, end=enddate, interval=interval1)
    df2.reset_index(inplace=True)
    if interval1 == '1d' or interval1 == '1wk' or interval1 == '1mo':
        df2['Date'] = pd.to_datetime(df2['Date'])
    else:
        df2['Date'] = pd.to_datetime(df2['Datetime'])
    df2['Day'] = df2['Date'].dt.day
    df2['Month'] = df2['Date'].dt.month
    df2['Year'] = df2['Date'].dt.year
    for _, row in df2.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']
        output_dir = f'SE_Data/{symbol}_data/{year}/{month}/{day}'
        os.makedirs(output_dir, exist_ok=True)
        file_path = f'{output_dir}/{symbol}_data.csv'
        row_df = pd.DataFrame([row])

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            row_df = row_df[existing_df.columns]
            if 'Date' in existing_df.columns:

                existing_df['Date'] = pd.to_datetime(existing_df['Date'])

                current_date = pd.to_datetime(row['Date']).date()

                date_rows = existing_df[existing_df['Date'].dt.date == current_date]

                if not (date_rows['Date'] == str(row['Date'])).any():
                    row_df.to_csv(file_path, mode='a', header=False, index=False)
            else:

                row_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            row_df.to_csv(file_path, mode='w', header=True, index=False)

class stock_dataform(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[DataRequired()])
    startdate = DateField('Start Date (YYYY-MM-DD)', validators=[DataRequired()])
    enddate = DateField('End Date (YYYY-MM-DD)', validators=[DataRequired()])
    interval1 = StringField('Interval (e.g., 1d, 1wk, 1mo)', validators=[DataRequired()])
    sma_ema  = SelectField('SMA/EMA', choices=[('SMA', 'SMA'), ('EMA', 'EMA')])
    sma1 = StringField('SMA Window-1')
    sma2 = StringField('SMA Window-2')
    ema1 = StringField('EMA Window-1')
    ema2 = StringField('EMA Window-2')
    submit = SubmitField('Get Data')


def sma_data(symbol, startdate, enddate, sma1, sma2):
    
    date_range = pd.date_range(start=startdate, end=enddate)
    dfs = []
    
    for date in date_range:
        year = date.year
        month = date.month
        day = date.day
        file_path = f'SE_Data/{symbol}_data/{year}/{month}/{day}/{symbol}_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
                df = df[df['Date'].dt.date == date.date()]
            dfs.append(df)

    
    if dfs:
        df1 = pd.concat(dfs, ignore_index=True)
        
        sma_1 = []
        sma_2 = []
        for x in range(len(df1)):
            if x < sma1 - 1:
                sma_1.append(np.nan)
            else:
                current_window = df1['Close'][x - (sma1 - 1):x + 1]
                sma_1.append(current_window.sum() / sma1)
        for x in range(len(df1)):
            if x < sma2 - 1:
                sma_2.append(np.nan)
            else:
                current_window = df1['Close'][x - (sma2 - 1):x + 1]
                sma_2.append(current_window.sum() / sma2)
        df1['SMA1'] = sma_1
        df1['SMA2'] = sma_2
        
        fig = plt.figure(figsize=(16, 8))
        plt.style.use('Solarize_Light2')
        plt.plot(df1['SMA1'], label=f'SMA_{sma1}')
        plt.plot(df1['SMA2'], label=f'SMA_{sma2}')
        plt.plot(df1['Close'], label='Original Price', color='k')
        plt.title('Simple Moving Average-1 vs. Simple Moving Average-2')
        plt.xlabel('Time Period')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64, df1
    else:
        return 'None'
    


def ema_data(symbol,startdate,enddate,ema1,ema2):
    date_range = pd.date_range(start=startdate, end=enddate)
    dfs = []
    for date in date_range:
        year = date.year
        month = date.month
        day = date.day
        file_path = f'SE_Data/{symbol}_data/{year}/{month}/{day}/{symbol}_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
                df = df[df['Date'].dt.date == date.date()]
            dfs.append(df)
    
    if dfs:
        df1 = pd.concat(dfs, ignore_index=True)
        df1.reset_index(inplace=True)
        df1[f'EMA_{ema1}'] = df1['Close'].ewm(span=ema1, adjust=False).mean()
        df1[f'EMA_{ema2}'] = df1['Close'].ewm(span=ema2, adjust=False).mean()
        df1['Signal'] = 0
        df1['Signal'] = np.where(df1[f'EMA_{ema1}'] > df1[f'EMA_{ema2}'], 1.0,0.0 )
        df1['Position'] = df1['Signal'].diff()
        
        df1['Date_1'] = np.where((df1['Position'] == 1) | (df1['Position'] == -1), df1['Date'],0.0 )
        df1['Buy_Data'] = np.where(df1['Position'] == 1, df1['Close'],0.0 )
        df1['Sell_Data'] = np.where(df1['Position'] == -1, df1['Close'], 0.0)
        conditions = [df1['Position'] == 1, (df1['Position'] == -1) & (df1['Position'] == -1), df1['Position'] == 0]
        choices = ('BUY', 'SELL', 'HOLD')
        df1['BUY_SELL_HOLD'] = np.select(conditions, choices, default='')
        
        fig = plt.figure(figsize=(27, 15))
        plt.style.use('Solarize_Light2')
        plt.plot(df1['Close'],color='k', label='Original')
        plt.plot(df1[f'EMA_{ema1}'],color='b', label=f'EMA_{ema1}')
        plt.plot(df1[f'EMA_{ema2}'],color='g', label=f'EMA_{ema2}')
        plt.plot(df1['Close'], label='Original Price', color='k')
        plt.plot(df1[df1['Position'] == 1].index,
            df1[f'EMA_{ema1}'][df1['Position'] == 1],
            '^', markersize=10, color='g', label='buy-short-term-bull')

        plt.plot(df1[df1['Position'] == -1].index,
            df1[f'EMA_{ema1}'][df1['Position'] == -1],
            'v', markersize=10, color='r', label='sell-short-term-bear')
        
        plt.title(''f'EMA_{ema1}'' vs. 'f'EMA_{ema2}')
        plt.xlabel('Time Period')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64, df1
        
    else:   
        return 'None'
    

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    message = db.Column(db.String(500), nullable=False)

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    phone = StringField('Phone', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    address = StringField('Address')
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    form = ContactForm()
    if form.validate_on_submit():
        contact = Contact(
            name=form.name.data,
            phone=form.phone.data,
            email=form.email.data,
            address=form.address.data,
            message=form.message.data
        )
        db.session.add(contact)
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('contactus.html', form=form)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    form = chatbot_form()
    
    if form.validate_on_submit():
        with open("instructions.txt", "r", encoding="utf-8") as f:
            instructions1 = f.read()
        
        prompt = form.prompt.data
        
        
        return Response(
            stream_chat_response(prompt, instructions1),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    return render_template('chatbot.html', form=form)

@app.route('/services', methods=['GET', 'POST'])
def services():
    form = stock_dataform()
    result_message = None
    plot_url = None
    table_data = None
    
    if form.validate_on_submit():
        symbol = form.symbol.data
        startdate = form.startdate.data
        enddate = form.enddate.data
        interval1 = form.interval1.data
        sma_ema = form.sma_ema.data
        save_data(symbol, startdate, enddate, interval1)
        if sma_ema == 'SMA':
            sma1 = int(form.sma1.data)
            sma2 = int(form.sma2.data)
            plot_result = sma_data(symbol, startdate, enddate, sma1, sma2)
            if plot_result and plot_result != 'None':
                plot_url, df1 = plot_result
                table_data = build_table(df1, 'blue_light')

        elif sma_ema == 'EMA':
            ema1 = int(form.ema1.data)
            ema2 = int(form.ema2.data)
            plot_result = ema_data(symbol, startdate, enddate, ema1, ema2)
            if plot_result and plot_result != 'None':
                plot_url, df1 = plot_result

                filtered_df = df1[(df1['Position'] == 1) | (df1['Position'] == -1)]
                table_data = build_table(filtered_df, 'red_dark', padding='20px', font_size='medium', width_dict='width_dict')

            else:
                result_message = "Please select either SMA or EMA."

    return render_template('services.html', stock_dataform=form, result_message=result_message, plot_url=plot_url, table_data=table_data)
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()