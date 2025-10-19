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
                    // Build image; assume docker is available on the agent
                    sh "sudo apt install -y docker.io"
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
                }
            }
        }
        stage('Prepare Data & Folders'){
            steps{
                script{
                    sh 'mkdir -p data model'
                    // Tải dataset từ S3, nếu thất bại dùng file trong repo (nếu có)
                    sh '''
                        set -e
                        if aws s3 ls s3://${S3_BUCKET}/data/weather_dataset.csv >/dev/null 2>&1; then
                          aws s3 cp s3://${S3_BUCKET}/data/weather_dataset.csv data/weather_dataset.csv
                          aws s3 cp s3://${S3_BUCKET}/data/weather_test.csv data/weather_test.csv
                        else
                          echo "S3 dataset not found, trying local repo copy..."
                          test -f weather_dataset.csv && cp weather_dataset.csv data/weather_dataset.csv || (echo "Dataset missing" && exit 1)
                        fi
                    '''
                }
            }
        }
        stage('Train model 1'){
            steps{
                script{
                    sh """
                    docker run --rm \
                        -v \$(pwd)/data:/app/data \
                        -v \$(pwd)/model:/app/model \
                        ${IMAGE_NAME}:${IMAGE_TAG} \
                        weather_train_1.py \
                        --data-path /app/data/weather_dataset.csv \
                        --model-output-path /app/model/model_1.pth
                    """
                }
            }
        }
        stage('Train model 2'){
            steps{
                script{
                    sh """
                    docker run --rm \
                        -v \$(pwd)/data:/app/data \
                        -v \$(pwd)/model:/app/model \
                        ${IMAGE_NAME}:${IMAGE_TAG} \
                        weather_train_2.py \
                        --data-path /app/data/weather_dataset.csv \
                        --model-output-path /app/model/model_2.pth
                    """
                }
            }
        }
        stage('Evaluate & Select Best'){
            steps{
                script{
                    sh """
                    docker run --rm \
                        -v \$(pwd)/data:/app/data \
                        -v \$(pwd)/model:/app/model \
                        ${IMAGE_NAME}:${IMAGE_TAG} \
                        weather_test.py \
                        --data-path /app/data/weather_test.csv \
                        --model1-path /app/model/model_1.pth \
                        --model2-path /app/model/model_2.pth \
                        --out-dir /app/model
                    """
                }
            }
        }
        stage('Archive & Upload artifacts'){
            steps{
                script {
                    archiveArtifacts artifacts: 'model/**', fingerprint: true
                    sh "aws s3 cp model/best_model.pth s3://${S3_BUCKET}/model/best_model_${IMAGE_TAG}.pth"
                    sh "aws s3 cp model/results.json s3://${S3_BUCKET}/model/results_${IMAGE_TAG}.json"
                }
            }
        }
        
    }
    post {
        always {
            echo 'clearing ip workspace'
            cleanWs()
            script {
                sh "docker rmi -f ${IMAGE_NAME}:${IMAGE_TAG} || true"
            }
        }
    }
}
