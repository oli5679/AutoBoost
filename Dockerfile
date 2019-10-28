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
    && conda env create -n classifierTools -f environment.yml \
    && rm -rf /opt/conda/pkgs/*


# activate the classifierTools environment
ENV PATH /opt/conda/envs/classifierTools/bin:$PATH

# copy over python files and dataset
COPY . .

# run train_model script
RUN cd src && python3 train_model.py 
