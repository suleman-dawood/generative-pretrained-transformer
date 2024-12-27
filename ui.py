from flask import Flask, render_template, request
from generate import generate_text

app = Flask(__name__)

@app.route('/')
def home():
    # renders the home page (index.html)
    return render_template('index.html')  # ensure this file exists in the templates folder

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    # handles post requests to generate text
    try:
        data = request.get_json()  # gets the json data from the request
        length = int(data.get('length', 100))  # defaults to a length of 100 if not provided

        result = generate_text(length)  # generates text based on the specified length

        return result  # returns the generated text as the response
    except Exception as e:
        # returns an error message with a 500 status code if something goes wrong
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    # starts the flask app in debug mode
    app.run(debug=True)
