import connexion
from swagger_server import encoder


def create_app():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Thalas AI APIs'}, pythonic_params=True)
    return app


app = create_app()


def main():
    app.run(port=8080, threaded=False)


if __name__ == '__main__':
    main()
