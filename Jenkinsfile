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
                    sh 'mkdir -p model'
                    sh "aws s3 cp s3://my-weather-pipeline-artifacts/data/weather.csv data/weather.csv"


                    sh """
                    docker run --rm \\
                        -v \$(pwd)/data:/app/data \\
                        -v \$(pwd)/model:/app/model \\
                        ${IMAGE_NAME}:${IMAGE_TAG} \\
                        --data-path /app/data/weather.csv \\
                        --model-output-path /app/model/model.pth
                    """
                }
            }
        }
        stage('Upload model to s3'){
            steps{
                sh "aws s3 cp model/model.pth s3://my-weather-pipeline-artifacts/model/weather-tcn-jenkins.pth"
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
