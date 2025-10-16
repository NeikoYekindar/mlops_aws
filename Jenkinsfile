pipeline {
    agent any
    environment {
        // Thay thế bằng tên S3 bucket của bạn
        S3_BUCKET = 'my-weather-pipeline-artifacts'
        IMAGE_NAME = 'weather-tcn-trainer'
        // Tự động tạo tag image dựa trên số lần build của Jenkins
        IMAGE_TAG = "build-${BUILD_NUMBER}" 
    }

    stages {
        stage('Checkout Code'){
            steps{
                git branch: 'main', url: 'https://github.com/NeikoYekindar/mlops_aws.git'
            }
        }
        stage('Build Docker Image'){
            steps{
                script {
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
                }
            }
        }
        stage('Train model'){
            steps{
                script{
                    sh 'mkdir -p data'
                    sh 'mkdir -p models'
                    sh "aws s3 cp s3://my-weather-pipeline-artifacts/data/weather.csv data/weather.csv"


                    sh """
                    docker run --rm \\
                        -v \$(pwd)/data:/opt/ml/input/data \\
                        -v \$(pwd)/models:/opt/ml/models \\
                        ${IMAGE_NAME}:${IMAGE_TAG} \\
                        --data-path /opt/ml/input/data/weather.csv \\
                        --model-dir /opt/ml/models
                    """
                }
            }
        }
        stage('Upload model to s3'){
            steps{
                sh "awws s3 cp models/models.pth s3://${S3_BUCKET}/models/weather-tcn-jenkins-${IMAGE_TAG}.pth"
            }
        }
        
    }
    post {
        always {
            echo 'clearing ip workspace'
            cleanWs()
            script {
                sh "docker rmi ${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }
    }
}