FROM tensorflow/tensorflow:2.7.0-gpu

# ==== UPDATE SISTEM ====
RUN apt-get update
RUN python -m pip install --upgrade pip
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
RUN apt install -y graphviz

# ==== BUILDING WORKING DIR ====
WORKDIR /home/
ADD ./requirements.txt ./requirements.txt

# ==== INSTALL PYTHON REQUIREMENTS ====
RUN pip install -r requirements.txt

# ==== EXPOSE PORTS ====
EXPOSE 8888
EXPOSE 6006
