# Use OpenJDK base image with Java 11 pre-installed
FROM openjdk:11-slim

# Set environment variables for Spark
ENV SPARK_VERSION=3.4.1
ENV HADOOP_VERSION=2
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Install Python, pip, and curl
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install Apache Spark
RUN curl -fsSL https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | \
    tar -xz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME

# Install Python dependencies
RUN pip3 install --no-cache-dir pyspark==${SPARK_VERSION} spark-nlp

# Set the working directory
WORKDIR /app

# Copy your Python script
COPY spell_check.py .

# Default command to run the script
CMD ["python3", "spell_check.py"]
