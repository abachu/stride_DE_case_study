# Use python base image
FROM python:3.10

# Disable python log buffering
ENV PYTHONUNBUFFERED=1

# Update apt-get and install packages
RUN apt-get update && \
    apt-get install -y \
        python3-dev python3-pip \
        postgresql-client \
        vim

# Set working directory
WORKDIR /dash

# Copy all files from context
COPY . /dash

# Upgrade pip and install packages specified in requirements
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run dash.py --server.port 80
