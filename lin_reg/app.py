from flask import Flask, request, jsonify;

app = Flask(__name__);

@app.route('/')
def home():
	return "Hello World";


@app.route('/item', methods=["POST"])
def create_item():
	req_data = request.get_json();
	return jsonify({
			'name': req_data['name'],
			'status': 'Item created successfully.'
		})

app.run(port = 5000)
