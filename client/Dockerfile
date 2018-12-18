FROM anibali/pytorch

USER root
# Install vim for local development
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "ffmpeg"]

# Add requirement.txt first for caching purposes.
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Running a terminal lets you run any script.
CMD /bin/bash
