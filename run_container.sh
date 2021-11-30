docker run -it --rm \
  --mount "type=bind,src=$(pwd),dst=/home/" \
  --workdir /home/ \
  --runtime=nvidia\
  -p 8888:8888\
  rnn_clf bash
