FROM continuumio/miniconda3

# clean up impage
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Do not run as root
RUN groupadd -r myuser && useradd -r -g myuser myuser

WORKDIR /app

# Install requirements
COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -n AutoBoost -f environment.yml \
    && rm -rf /opt/conda/pkgs/* 
EXPOSE 5000 5000

# activate the classifierTools environment
ENV PATH /opt/conda/envs/AutoBoost/bin:$PATH

# copy over python files and dataset
COPY . .

CMD ['bash', 'conda activate AutoBoost']
CMD [ "python3", "src/model_builder/auto_builder.py" ]
CMD [ "python3", "src/flask_server/server.py" ]

#  docker build -t oli5679/autoboost .

# docker run  -p 5000:5000 oli5679/autoboost