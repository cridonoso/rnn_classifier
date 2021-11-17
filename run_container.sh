docker run -it --rm \
  --mount "type=bind,src=$(pwd),dst=/home/" \
  --workdir /home/ \
  -p 8888:8888\
  rnn_clf bash
