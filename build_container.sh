docker build -t rnn_clf \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .
