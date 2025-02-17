python inference_server/server.py


Test:

curl -X POST http://127.0.0.1:5888/reset

curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:5888/step

curl -X POST http://127.0.0.1:5888/simulate
