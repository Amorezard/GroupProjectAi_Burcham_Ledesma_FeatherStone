from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hiro')
def hiro():
    return render_template('hiro.html')

@app.route('/test')
def test():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', ssl_context=('ssl_cert/cert.pem', 'ssl_cert/key.pem')) 