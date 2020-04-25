from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from utils.model import load_model

app = Flask(__name__)
api = Api(app)

model = load_model()


class SVMModel(Resource):
    def post(self):
        posted_data = request.get_json()

        assert 'x' in posted_data
        assert 'y' in posted_data

        pred = model.predict([
            list(posted_data.values())
        ])[0]

        return jsonify({'prediction': {'class': int(pred)}})


api.add_resource(SVMModel, '/clusterize')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
