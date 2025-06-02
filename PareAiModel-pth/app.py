from flask import Flask
from routes.predict_routes import predict_bp

app = Flask(__name__)
app.register_blueprint(predict_bp)

@app.route('/', methods=['GET'])
def health_check():
    return {'message': 'API Active'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
