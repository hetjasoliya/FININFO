
#Stock Analysis Web App Backend & Frontend Guide

This is a Flask-based web application for stock data analysis and AI-powered chat, featuring SMA/EMA indicators, contact form storage, and more.

---

## Backend Features

- SMA & EMA based stock trend analysis
- AI-powered chatbot using OpenRouter (Mistral model)
- Stores contact queries to MySQL
- Streaming chat responses (SSE)
- Interactive data visualizations using Matplotlib
- Dynamic plotting and table rendering with `pretty_html_table`
- Bootstrap-enhanced templates via Flask-Bootstrap
- Modular FlaskForm setup with WTForms

## Tech Stack

Backend:
- Flask, Flask-WTF, Flask-Bootstrap, Flask-Modals
- WTForms, SQLAlchemy, PyMySQL
- yfinance, pandas, numpy, matplotlib
- pretty_html_table, OpenAI API (via OpenRouter)
- SSE for chatbot

Frontend:
- Jinja2 templating, Bootstrap
- Dynamic charts, styled tables

## Project Structure

.
├── app.py
├── templates/
├── static/
├── SE_Data/
├── instructions.txt
└── README.txt

## Setup Instructions

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create MySQL DB:

```sql
CREATE DATABASE fininfo;
```

3. Set `OPENROUTER_API_KEY` as an environment variable or in `app.py`.

4. Run the server:

```bash
python app.py
```

## requirements.txt

Flask==2.3.3
Flask-WTF==1.2.1
Flask-Bootstrap==3.3.7.1
Flask-SQLAlchemy==3.1.1
Flask-Modals==0.1.1
WTForms==3.1.2
pymysql==1.1.0
yfinance==0.2.36
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.4
pretty-html-table==0.9.16
openai==1.30.1
pydantic==2.7.1

---
# Route Summary

| Route        | Methods  | Description                         |
| ------------ | -------- | ----------------------------------- |
| `/`          | GET      | Home page                           |
| `/about`     | GET      | About page                          |
| `/contactus` | GET/POST | Contact form (stores to DB)         |
| `/chatbot`   | GET/POST | AI chatbot with SSE response        |
| `/services`  | GET/POST | Stock data input & SMA/EMA analysis |

# Frontend Guide

## Templates Overview

- `home.html`: Landing page with navigation
- `services.html`: Stock symbol/date range form with SMA/EMA charts and tables
- `chatbot.html`: AI chatbot interface using SSE
- `contactus.html`: Contact form storing entries in MySQL
- `about.html`: Static info page

## Static Assets

static/
├── css/
├── js/
└── images/

## Interactivity & Styling

- Bootstrap 5 styling (Flask-Bootstrap)
- Jinja2 templates
- Matplotlib chart themes
- `pretty_html_table` for styled data tables

## Testing URLs

- `/`: Home
- `/services`: Stock tool
- `/chatbot`: Chat interface
- `/contactus`: Contact form

---

## Optional Enhancements

- Chart.js or Plotly for interactivity
- Form field animations
- More visual themes

## Security

- CSRF protection via Flask-WTF
- Proper input validation
- Safe DB storage for contacts

---


