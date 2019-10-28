FROM continuumio/miniconda3

# Install extra packages if required
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Add the user that will run the app (no need to run as root)
RUN groupadd -r myuser && useradd -r -g myuser myuser

WORKDIR /app

# Install myapp requirements
COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -n classifierTools -f environment.yml \
    && rm -rf /opt/conda/pkgs/*


# activate the myapp environment
ENV PATH /opt/conda/envs/classifierTools/bin:$PATH

COPY . .

RUN cd src && python3 train_model.py 
